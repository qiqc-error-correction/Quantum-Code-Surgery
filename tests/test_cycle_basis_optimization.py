"""Same-graph cycle optimization precedes expensive geometric sparsification."""
import itertools

import numpy as np
import pytest

from codesurgery.distance_preservation import ensure_distance
from codesurgery.distance_preservation.sparsify import sparsify_cycles
from codesurgery.examples import direct_sum, four_qubit_code, ibm_bb90_code
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank, solve
from codesurgery.symplectic import pauli_product
from codesurgery.universal_adapters.graphs import cycle_basis


def graph(n, edges):
    b = np.zeros((len(edges), n), dtype=np.uint8)
    for i, edge in enumerate(edges):
        b[i, list(edge)] = 1
    return b


# The degree-expanded graph from the seeded BB90 X1 X3 regression.  Keeping
# just its 42 endpoint pairs avoids a multi-megabyte saved deformed-code fixture.
BB90_EXPANDED_EDGES = [
    (1, 13), (2, 14), (5, 15), (4, 16), (3, 4), (5, 17),
    (7, 8), (6, 8), (6, 7), (9, 10), (0, 10), (9, 18),
    (11, 12), (1, 12), (2, 11), (13, 14), (14, 19), (3, 13),
    (16, 17), (15, 17), (15, 16), (4, 18), (5, 18), (6, 19),
    (0, 7), (8, 19), (1, 9), (2, 10), (0, 11), (3, 12),
    (5, 14), (1, 11), (10, 17), (13, 16), (3, 19), (6, 15),
    (7, 18), (2, 4), (0, 8), (9, 12), (2, 19), (5, 12),
]


def assert_complete_same_cycle_space(b, original, replacement):
    assert not np.any((replacement.astype(np.int64) @ b) % 2)
    assert rank(replacement) == len(b) - b.shape[1] + 1
    assert in_rowspace(original, replacement)
    assert in_rowspace(replacement, original)


def test_bb90_expanded_graph_avoids_seventeen_layers_by_default():
    b = graph(20, BB90_EXPANDED_EDGES)
    original = cycle_basis(b)
    matching = np.eye(len(b), dtype=np.uint8)[:3]
    saved = [a.copy() for a in (b, original, matching)]
    assert int(original.sum(axis=0).max()) == 13

    actual_b, actual_m, cycles, report = sparsify_cycles(
        b, matching=matching, cycles=original,
        max_cycle_weight=8, max_cycle_congestion=12,
    )
    np.testing.assert_array_equal(actual_b, b)
    np.testing.assert_array_equal(actual_m, matching)
    assert_complete_same_cycle_space(b, original, cycles)
    assert int(cycles.sum(axis=1).max()) <= 8
    assert int(cycles.sum(axis=0).max()) <= 12
    assert not report["applied"]  # No topology change, even though the basis changed.
    assert report["layers"] == 1
    assert report["cycle_basis_optimization"]["attempted"]
    assert report["cycle_basis_optimization"]["changed"]
    assert 90 + len(actual_b) == 132

    legacy_b, _, _, legacy_report = sparsify_cycles(
        b, matching=matching, cycles=original,
        max_cycle_weight=8, max_cycle_congestion=12,
        optimize_cycle_basis=False,
    )
    assert legacy_report["applied"] and legacy_report["layers"] == 17
    assert 90 + len(legacy_b) == 1124
    repeated = sparsify_cycles(b, matching=matching, cycles=original,
                               max_cycle_weight=8, max_cycle_congestion=12)
    np.testing.assert_array_equal(repeated[2], cycles)
    assert repeated[3]["cycle_basis_optimization"] == report["cycle_basis_optimization"]
    for current, before in zip((b, original, matching), saved):
        np.testing.assert_array_equal(current, before)


def test_redundant_parallel_edge_cycles_are_optimized_without_losing_digon():
    b = graph(3, [(0, 1), (0, 1), (1, 2), (2, 0)])
    supplied = np.array([[1, 1, 0, 0], [1, 0, 1, 1],
                         [0, 1, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
    assert rank(supplied) == 2 and supplied.sum(axis=0).max() == 3
    new_b, _, replacement, report = sparsify_cycles(
        b, cycles=supplied, max_cycle_congestion=2,
    )
    np.testing.assert_array_equal(new_b, b)
    assert_complete_same_cycle_space(b, supplied, replacement)
    assert len(replacement) == 2
    assert any(np.array_equal(row, [1, 1, 0, 0]) for row in replacement)
    assert replacement.sum(axis=0).max() <= 2
    assert report["cycle_basis_optimization"]["changed"]
    assert not report["applied"]


@pytest.mark.parametrize("b", [graph(1, []), graph(4, [(0, 1), (1, 2), (2, 3)])])
def test_tree_skips_optimizer_and_preserves_empty_cycle_shape(b):
    actual, matching, cycles, report = sparsify_cycles(b)
    np.testing.assert_array_equal(actual, b)
    assert actual is not b and matching is None
    assert cycles.shape == (0, len(b))
    assert not report["cycle_basis_optimization"]["attempted"]
    assert not report["cycle_basis_optimization"]["changed"]
    assert not report["applied"]


def test_already_sparse_user_generators_are_preserved_exactly():
    b = graph(3, [(0, 1), (1, 2), (2, 0)])
    supplied = np.ones((2, 3), dtype=np.uint8)
    _, _, cycles, report = sparsify_cycles(b, cycles=supplied)
    np.testing.assert_array_equal(cycles, supplied)
    assert not report["cycle_basis_optimization"]["attempted"]
    assert not report["cycle_basis_optimization"]["changed"]


@pytest.mark.parametrize("supplied", [
    [[1, 0, 0, 0, 0]],  # Not closed.
    [[1, 1, 1, 1, 0]],  # Closed but fails to span both cycles.
])
def test_optimizer_does_not_silently_repair_invalid_supplied_cycle_space(supplied):
    b = graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    with pytest.raises(ValueError, match="complete"):
        sparsify_cycles(b, cycles=supplied, max_cycle_congestion=1)


def test_candidate_budget_falls_back_to_valid_legacy_construction():
    b = graph(4, list(itertools.combinations(range(4), 2)))
    kwargs = dict(max_cycle_congestion=1, max_cycle_weight=4)
    actual = sparsify_cycles(b, cycle_basis_max_candidates=1, **kwargs)
    legacy = sparsify_cycles(b, optimize_cycle_basis=False, **kwargs)
    for new, old in zip((actual[0], actual[2]), (legacy[0], legacy[2])):
        np.testing.assert_array_equal(new, old)
    assert actual[3]["applied"]
    assert actual[3]["cycle_basis_optimization"]["attempted"]
    assert not actual[3]["cycle_basis_optimization"]["changed"]
    assert not np.any((actual[2].astype(np.int64) @ actual[0]) % 2)
    assert rank(actual[2]) == len(actual[0]) - actual[0].shape[1] + 1


@pytest.mark.parametrize("options", [
    {"optimize_cycle_basis": 1},
    {"cycle_basis_max_roots": 0},
    {"cycle_basis_max_roots": True},
    {"cycle_basis_max_candidates": 0},
    {"cycle_basis_max_candidates": 1.5},
])
@pytest.mark.parametrize("pipeline", [False, True])
def test_optimization_options_validate_before_already_sparse_return(options, pipeline):
    if pipeline:
        result = gauging_deform(four_qubit_code(), "XXII", require_distance=False)
        call = lambda: ensure_distance(result, **options)
    else:
        call = lambda: sparsify_cycles(graph(2, [(0, 1)]), **options)
    with pytest.raises(ValueError, match=next(iter(options))):
        call()


def test_triangle_weight_cap_can_be_met_without_unavailable_layer_squares():
    b = graph(3, [(0, 1), (1, 2), (2, 0)])
    matching = np.zeros((2, 3), dtype=np.uint8)
    matching[1, 0] = 1
    before = gauging_deform(four_qubit_code(), "-XXII", graph=b,
        matching=matching, ports=[0, 1], cycle_basis=np.ones((3, 3), dtype=np.uint8),
        require_distance=False)
    after = ensure_distance(before, max_cycle_weight=3, max_cycle_congestion=2)
    assert after.n == before.n == 7
    assert after.metadata["max_cycle_weight"] == 3
    assert after.metadata["max_cycle_congestion"] == 1
    sparse = after.metadata["distance_postprocessing"]["sparsification"]
    assert sparse["cycle_basis_optimization"]["changed"]
    assert not sparse["applied"] and sparse["layers"] == 1
    _assert_same_signed_group(before, after)
    with pytest.raises(ValueError, match="layer-square weight 4"):
        ensure_distance(before, max_cycle_weight=3, max_cycle_congestion=2,
                        optimize_cycle_basis=False)


def _assert_same_signed_group(before, after):
    old = np.hstack((before.check_x, before.check_z))
    new = np.hstack((after.check_x, after.check_z))
    assert rank(old) == rank(new)
    assert in_rowspace(old, new) and in_rowspace(new, old)
    transforms = solve(old.T, new.T).T
    for row, selected in enumerate(transforms):
        x, z, phase = pauli_product(before.check_x, before.check_z,
            before.check_signs, selected=np.flatnonzero(selected))
        np.testing.assert_array_equal(x, after.check_x[row])
        np.testing.assert_array_equal(z, after.check_z[row])
        assert phase == 2 * int(after.check_signs[row])


def test_bb90_four_factor_default_pipeline_preserves_signed_code_without_layers():
    code = ibm_bb90_code()
    # A fixed weight-20 representative of X1 X3 Z2 Z4.  Its Y positions make
    # phase preservation nontrivial; the negative sign is deliberate.
    x = np.zeros(90, dtype=np.uint8)
    z = np.zeros(90, dtype=np.uint8)
    x[[13, 16, 26, 28, 31, 39, 43, 46, 59, 62, 70, 75, 85, 87]] = 1
    z[[16, 27, 28, 29, 31, 45, 46, 47, 60, 62, 70, 75, 77, 85]] = 1
    target = dict(x=x, z=z, sign=1)
    before = gauging_deform(code, target, require_distance=False)
    assert before.n == 132 and before.metadata["max_cycle_weight"] == 8
    snapshot = before.check_x.copy(), before.check_z.copy(), before.check_signs.copy()
    # A published source distance and an independently checked graph bound
    # certify the lower bound. This test does not confuse RIS upper bounds
    # with an exhaustive proof of distance ten.
    after = ensure_distance(before, distance_method="ris", distance_ris_trials=16,
                            distance_max_candidates=10000, distance_time_limit=2)
    assert after.n == 132 and after.k_stabilizer == 7
    assert after.metadata["distance_preservation_guaranteed"]
    assert after.metadata["distance_lower_bound"] >= 10
    assert after.metadata["max_check_weight"] <= 7
    report = after.metadata["distance_postprocessing"]
    assert report["expansion_added_edges"] == 0
    assert report["sparsification"]["layers"] == 1
    assert not report["sparsification"]["applied"]
    assert report["sparsification"]["cycle_basis_optimization"]["changed"]
    assert after.measurement_sign == 1 and after.validate()
    np.testing.assert_array_equal(after.metadata["graph_incidence"], before.metadata["graph_incidence"])
    np.testing.assert_array_equal(after.metadata["matching_matrix"], before.metadata["matching_matrix"])
    _assert_same_signed_group(before, after)
    for current, saved in zip((before.check_x, before.check_z, before.check_signs), snapshot):
        np.testing.assert_array_equal(current, saved)

    # Eliminate auxiliary support independently: only the requested product
    # may be read out, even though it contains physical X, Y and Z factors.
    coefficients = nullspace(np.hstack((after.check_x[:, 90:], after.check_z[:, 90:])).T)
    read_x = (coefficients @ after.check_x[:, :90]) & 1
    read_z = (coefficients @ after.check_z[:, :90]) & 1
    basis = code.standard_logicals()
    signatures = np.hstack(((read_x @ basis.jz.T) & 1, (read_z @ basis.jx.T) & 1))
    expected = np.zeros(16, dtype=np.uint8)
    expected[[0, 2, 9, 11]] = 1
    assert rank(signatures) == 1
    assert in_rowspace(signatures, expected[None, :])
    assert in_rowspace(expected, signatures)


@pytest.mark.parametrize("pair", [False, True])
def test_basis_only_rebuild_reverifies_compositional_adapter_distance_proof(monkeypatch, pair):
    from codesurgery.gauging_measurement import components

    code = direct_sum(*(four_qubit_code() for _ in range(4)))
    exact_limit = 2 if pair else 4
    if pair:
        before = gauging_pair_deform(code, "-XXIIXXII" + "IIII" * 2,
            "IIII" * 2 + "ZZIIZZII", require_distance=False, exact_limit=exact_limit)
    else:
        before = gauging_deform(code, "-" + "XXII" * 4,
            require_distance=False, exact_limit=exact_limit)
    assert before.metadata["max_cycle_congestion"] == 4
    assert before.metadata["relative_expansion"]["condition_met"]
    assert not before.metadata["relative_expansion"]["exact"]
    checked_blocks = []
    original_verify = components.verify_component_adapter_expansion

    def verify_actual_blocks(b, *args, **kwargs):
        checked_blocks.append(b.copy())
        return original_verify(b, *args, **kwargs)

    monkeypatch.setattr(components, "verify_component_adapter_expansion", verify_actual_blocks)
    # No numerical candidates, and no full-graph expansion enumeration: the
    # result must retain and re-derive the actual component/adapter proof.
    after = ensure_distance(before, max_cycle_congestion=2,
        exact_limit=exact_limit, certificate_limit=exact_limit,
        distance_method="enumeration", distance_max_candidates=0)
    assert after.n == before.n == (28 if pair else 26)
    assert after.metadata["max_cycle_congestion"] == 2
    assert after.metadata["distance_preservation_guaranteed"]
    assert after.metadata["distance_lower_bound"] >= 2
    assert not after.metadata["relative_expansion"]["exact"]
    assert after.metadata["relative_expansion"]["condition_met"]
    assert after.metadata["distance_postprocessing"]["sparsification"]["cycle_basis_optimization"]["changed"]
    for key in ("graph_incidence", "matching_matrix", "ports"):
        np.testing.assert_array_equal(after.metadata[key], before.metadata[key])
    _assert_same_signed_group(before, after)
    assert after.validate()

    if pair:
        proof = after.metadata["adapter_distance_proof"]
        assert proof["condition_met"] and len(checked_blocks) >= 2
        for factor in proof["factor_relative_expansion"]:
            # Even factor graphs exceed exact_limit=2: nested component proofs
            # are needed, rather than just the outer adapter metadata.
            assert not factor["exact"] and factor["condition_met"]
            assert factor["component_adapter_proof"]["verified_against_actual_graph"]
            assert factor["component_adapter_proof"]["condition_met"]
    else:
        proof = after.metadata["component_adapter_distance_proof"]
        assert proof["component_count"] == 4
        assert proof["condition_met"] and proof["verified_against_actual_graph"]
        assert any(np.array_equal(block, after.metadata["graph_incidence"])
                   for block in checked_blocks)
