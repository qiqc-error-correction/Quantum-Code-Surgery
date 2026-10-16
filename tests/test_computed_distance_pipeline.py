"""Actual-distance-driven repair and the dressed-memory row-weight threshold.

The distance witnesses below are checked against the returned matrices, not
against the expansion certificate.  In particular, h < 1 is not a distance
failure: the Hamming Y example has h = 1/2 and distance exactly three.
"""
import itertools

import numpy as np
import pytest

from codesurgery.examples import direct_sum, four_qubit_code, steane_code
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli, pauli_product


SINGLES = [gadget_deform, gauging_deform]
PAIRS = [gadget_pair_deform, gauging_pair_deform]


def hamming_code(distance=3):
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)]
                  for bit in (3, 2, 1, 0)], dtype=np.uint8)
    return CSSCode(h, h, distance=distance)


def independently_check_distance(result, expected, witness):
    """Exhaust all lighter Pauli errors; validate an explicit logical witness."""
    x, z = result.check_x, result.check_z
    checks = np.hstack([x, z])
    errors = []
    for weight in range(1, expected):
        for support in itertools.combinations(range(result.n), weight):
            for paulis in itertools.product(((1, 0), (0, 1), (1, 1)), repeat=weight):
                row = np.zeros(2 * result.n, dtype=np.uint8)
                for qubit, (px, pz) in zip(support, paulis):
                    row[qubit], row[result.n + qubit] = px, pz
                errors.append(row)
    if errors:
        errors = np.asarray(errors)
        syndrome = (errors[:, :result.n] @ z.T + errors[:, result.n:] @ x.T) % 2
        assert in_rowspace(errors[~syndrome.any(axis=1)], checks)
    wx, wz, _ = parse_pauli(witness, result.n)
    assert np.count_nonzero(wx | wz) == expected
    assert not np.any((x @ wz + z @ wx) % 2)
    assert not in_rowspace(np.concatenate([wx, wz]), checks)


def independently_check_readout(code, result, excluded=()):
    assert result.validate()
    x, z = result.check_x, result.check_z
    cancel = nullspace(np.hstack([x[:, code.n:], z[:, code.n:]]).T)
    image = cancel @ np.hstack([x[:, :code.n], z[:, :code.n]]) % 2
    old = np.block([[code.hx, np.zeros_like(code.hx)],
                    [np.zeros_like(code.hz), code.hz]])
    target = np.concatenate([result.measurement_x, result.measurement_z])
    assert rank(np.vstack([old, image])) == rank(old) + 1
    assert in_rowspace(image, np.vstack([old, target]))
    assert in_rowspace(target, image)
    for word in excluded:
        ex, ez, _ = parse_pauli(word, code.n)
        assert not in_rowspace(np.concatenate([ex, ez]), image)
    wx, wz, phase = pauli_product(x, z, result.check_signs,
                                 np.flatnonzero(result.measurement_witness))
    assert np.array_equal(wx[:code.n], result.measurement_x)
    assert np.array_equal(wz[:code.n], result.measurement_z)
    assert not np.any(wx[code.n:] | wz[code.n:])
    assert phase == 2 * result.measurement_sign


def assert_final_row_threshold(result, expected):
    report = result.metadata["distance_postprocessing"]
    assert report["sparsity_threshold"] == expected
    assert report["sparsity_threshold_source"]
    actual = int((result.check_x | result.check_z).sum(axis=1).max(initial=0))
    assert actual <= expected
    assert report["sparsity_certificate"]["actual"]["max_check_weight"] == actual
    assert report["sparsity_certificate"]["verified_against_full_check_matrix"]


@pytest.mark.parametrize("route", SINGLES)
def test_actual_distance_passes_despite_failed_expansion_bound(route):
    code = hamming_code()
    basic = route(code, "-YXXZZ" + "I" * 10, require_distance=False)
    result = route(code, "-YXXZZ" + "I" * 10)
    assert result.n == basic.n == 19
    assert result.metadata["relative_expansion"]["beta"] == .5
    assert not result.metadata["relative_expansion"]["condition_met"]
    assert result.metadata["distance_calculation"]["status"] == "exact"
    assert result.metadata["distance_calculation"]["distance"] == 3
    assert result.metadata["source_distance_calculation"]["distance"] == 3
    post = result.metadata["distance_postprocessing"]
    assert post["initial_distance_calculation"]["distance"] == 3
    assert not post["changed"]
    assert post["expansion_added_edges"] == 0
    for key in ("check_x", "check_z", "check_signs", "measurement_witness"):
        assert np.array_equal(getattr(result, key), getattr(basic, key))
    assert result.metadata["distance_preservation_guaranteed"]
    assert_final_row_threshold(result, 9)
    independently_check_readout(code, result)
    independently_check_distance(result, 3, "YIIIIYY" + "I" * 12)


@pytest.mark.parametrize("route", PAIRS)
def test_actual_distance_failure_triggers_bounded_expansion(route):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    factors = ("XXIIIIII", "IIIIZZII")
    basic = route(code, *factors, adapter_size=1, require_distance=False)
    independently_check_distance(basic, 1, "IIIIIIIIIXI")
    result = route(code, *factors, adapter_size=1)
    post = result.metadata["distance_postprocessing"]
    assert basic.n == 11 and result.n == 12
    assert post["initial_distance_calculation"]["status"] == "exact"
    assert post["initial_distance_calculation"]["distance"] == 1
    assert post["expansion_added_edges"] == 1
    assert post["expansion"]["acceptance"] == "distance"
    assert result.metadata["distance_calculation"]["distance"] == 2
    assert_final_row_threshold(result, 5)
    independently_check_readout(code, result, excluded=factors)
    independently_check_distance(result, 2, "XX" + "I" * (result.n - 2))


@pytest.mark.parametrize("route", SINGLES)
def test_missing_source_distance_is_calculated_without_mutating_input(route):
    known = four_qubit_code()
    code = CSSCode(known.hx, known.hz)
    result = route(code, "XXII")
    assert code.distance is None
    source = result.metadata["source_distance_calculation"]
    assert source["status"] == "exact" and source["distance"] == 2
    assert result.metadata["distance_lower_bound"] >= 2
    independently_check_readout(code, result)


@pytest.mark.parametrize("route", SINGLES)
@pytest.mark.parametrize("word", ["-YXZI", "-XXII"])
def test_auto_mode_preserves_signed_single_measurements(route, word):
    code = four_qubit_code()
    result = route(code, word)
    assert result.measurement_sign == 1
    assert result.metadata["distance_postprocessing"]["required"]
    assert result.metadata["distance_calculation"]["distance"] >= 2
    assert_final_row_threshold(result, 5)
    independently_check_readout(code, result)


@pytest.mark.parametrize("route", PAIRS)
@pytest.mark.parametrize("left", ["-YXZIIIII", "-XXIIIIII"])
def test_auto_mode_preserves_signed_joint_measurement_only(route, left):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    right = "IIIIZZII"
    result = route(code, left, right)
    assert result.measurement_sign == 1
    assert result.metadata["distance_calculation"]["distance"] >= 2
    assert_final_row_threshold(result, 5)
    independently_check_readout(code, result, excluded=(left, right))


@pytest.mark.parametrize("route,pair", [(r, False) for r in SINGLES] + [(r, True) for r in PAIRS])
def test_explicit_false_skips_automatic_distance_work(route, pair):
    code = direct_sum(four_qubit_code(), four_qubit_code()) if pair else hamming_code()
    args = ("XXIIIIII", "IIIIZZII") if pair else ("-YXXZZ" + "I" * 10,)
    kwargs = {"adapter_size": 1} if pair else {}
    result = route(code, *args, require_distance=False,
                   distance_options={"distance_method": "enumeration", "distance_max_candidates": 1, "max_check_weight": 1}, **kwargs)
    assert "distance_postprocessing" not in result.metadata
    assert result.n == (11 if pair else 19)
    assert not result.metadata["distance_preservation_guaranteed"]
    independently_check_readout(code, result, excluded=args if pair else ())


@pytest.mark.parametrize("route", SINGLES)
def test_insufficient_calculation_budget_never_claims_a_failed_distance(route):
    # beta = 1/2 offers no theorem fallback; one candidate cannot even finish
    # the weight-one shell. The explicit legacy policy raises on uncertainty;
    # the default certificate-seeking policy has separate regression coverage.
    with pytest.raises(ValueError, match="(?i)distance"):
        route(hamming_code(), "-YXXZZ" + "I" * 10,
              distance_options={"distance_method": "enumeration", "distance_max_candidates": 1,
                                "unknown_distance_policy": "error"})


@pytest.mark.parametrize("route", SINGLES)
def test_declared_source_distance_above_computed_distance_is_rejected(route):
    known = four_qubit_code()
    code = CSSCode(known.hx, known.hz, distance=3)
    with pytest.raises(ValueError, match="(?i)distance"):
        route(code, "XXII")


@pytest.mark.parametrize("route", SINGLES)
def test_impossible_explicit_row_limit_is_not_silently_increased(route):
    with pytest.raises(ValueError, match="(?i)(spars|weight|limit|threshold)"):
        route(four_qubit_code(), "XXII", distance_options={"max_check_weight": 4})


def test_cycle_exceeding_memory_threshold_is_cellulated_without_expansion():
    code = four_qubit_code()
    graph = np.zeros((8, 8), dtype=np.uint8)
    for vertex in range(8):
        graph[vertex, [vertex, (vertex + 1) % 8]] = 1
    matching = np.zeros((2, 8), dtype=np.uint8)
    matching[1, 0] = 1
    kwargs = dict(graph=graph, ports=[0, 1], matching=matching)
    basic = gauging_deform(code, "XXII", require_distance=False, **kwargs)
    assert int((basic.check_x | basic.check_z).sum(axis=1).max()) == 8
    result = gauging_deform(code, "XXII", **kwargs)
    post = result.metadata["distance_postprocessing"]
    assert basic.n == 12 and result.n == 17
    assert post["initial_distance_calculation"]["distance"] == 2
    assert post["expansion_added_edges"] == 0
    assert post["sparsification"]["added_chord_count"] == 5
    assert result.metadata["max_cycle_weight"] <= 5
    assert_final_row_threshold(result, 5)
    independently_check_readout(code, result)
    independently_check_distance(result, 2, "XIX" + "I" * 14)


def test_measuring_only_logical_qubit_has_no_remaining_logicals():
    result = gauging_deform(steane_code())
    assert result.k_stabilizer == 0
    assert result.metadata["distance_calculation"]["status"] == "no_logicals"
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_postprocessing"]["expansion_added_edges"] == 0


def test_graph_bound_is_separate_from_an_incomplete_actual_distance_search():
    result = gauging_deform(four_qubit_code(), "XXII",
                           distance_options={"distance_method": "enumeration", "distance_max_candidates": 1})
    actual = result.metadata["distance_calculation"]
    assert actual["distance"] is None
    assert actual["proven_lower_bound"] == 1
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] == 2
    assert "graph lower-bound" in result.metadata["distance_verification_method"]
    assert result.metadata["distance_postprocessing"]["expansion_added_edges"] == 0


def test_existing_short_cycles_allow_a_bound_below_four():
    result = gauging_deform(four_qubit_code(), "XXII",
                           distance_options={"max_cycle_weight": 3})
    assert result.metadata["max_cycle_weight"] <= 3
    assert result.metadata["distance_calculation"]["distance"] == 2


def test_public_distance_helper_rejects_a_mutated_sealed_code():
    from codesurgery.distance_preservation import code_distance
    result = gauging_deform(four_qubit_code(), "XXII", require_distance=False)
    result.check_x[0, 0] ^= 1
    with pytest.raises(ValueError, match="changed"):
        code_distance(result)


@pytest.mark.parametrize("swap", [False, True])
def test_pure_target_threshold_does_not_silently_include_the_other_css_block(swap):
    hx, hz = [[1, 1, 0, 0]], [[1, 1, 1, 1]]
    code = CSSCode(hz, hx) if swap else CSSCode(hx, hz)
    logical = "XXII" if swap else "ZZII"
    # The relevant dressed block has row weight 3. The other memory block
    # already weighs 4 and graph-only cellulation cannot make it lighter.
    with pytest.raises(ValueError, match="sparsity"):
        gauging_deform(code, logical)
    result = gauging_deform(code, logical, distance_options={"max_check_weight": 4})
    post = result.metadata["distance_postprocessing"]
    assert post["sparsity_threshold"] == 3
    assert post["effective_row_threshold"] == 4
    assert post["sparsity_reference_rows"] == ([1] if swap else [0])
