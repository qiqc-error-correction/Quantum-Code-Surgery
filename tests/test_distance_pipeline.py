"""End-to-end signed readout, distance, and full-check sparsity verification."""
import copy
import itertools

import numpy as np
import pytest

from codesurgery.distance_preservation import ensure_distance
from codesurgery.examples import direct_sum, four_qubit_code
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.gauging_measurement.api import build_graph
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli, pauli_product


SINGLES = [gadget_deform, gauging_deform]
PAIRS = [gadget_pair_deform, gauging_pair_deform]


def hamming_code():
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)]
                  for bit in (3, 2, 1, 0)], dtype=np.uint8)
    return CSSCode(h, h, distance=3)


def graph(n, edges):
    b = np.zeros((len(edges), n), dtype=np.uint8)
    for e, endpoints in enumerate(edges):
        b[e, list(endpoints)] = 1
    return b


def assert_actual_measurement(code, result, excluded=()):
    """Compute the stabilizer image on memory, independently of certificates."""
    assert result.validate()
    x, z = result.check_x, result.check_z
    assert not np.any((x @ z.T + z @ x.T) % 2)
    cancellation = nullspace(np.hstack([x[:, code.n:], z[:, code.n:]]).T)
    image = cancellation @ np.hstack([x[:, :code.n], z[:, :code.n]]) % 2
    old = np.block([[code.hx, np.zeros_like(code.hx)],
                    [np.zeros_like(code.hz), code.hz]])
    target = np.concatenate([result.measurement_x, result.measurement_z])
    assert rank(np.vstack([old, image])) == rank(old) + 1
    assert in_rowspace(image, np.vstack([old, target]))
    assert in_rowspace(target, image)
    for word in excluded:
        fx, fz, _ = parse_pauli(word, code.n)
        assert not in_rowspace(np.concatenate([fx, fz]), image)
    px, pz, phase = pauli_product(x, z, result.check_signs,
                                 np.flatnonzero(result.measurement_witness))
    assert np.array_equal(px[:code.n], result.measurement_x)
    assert np.array_equal(pz[:code.n], result.measurement_z)
    assert not np.any(px[code.n:] | pz[code.n:])
    assert phase == 2 * result.measurement_sign
    assert result.k_stabilizer == code.k - 1
    post = result.metadata["distance_postprocessing"]
    assert post["required"] and post["static_distance_only"]
    assert post["distance_lower_bound"] == code.distance
    cert = post["sparsity_certificate"]
    assert cert["condition_met"] and cert["verified_against_full_check_matrix"]
    assert cert["actual"]["max_check_weight"] == int((x | z).sum(axis=1).max())
    assert cert["actual"]["max_qubit_degree"] == int((x | z).sum(axis=0).max())
    assert all(cert["actual"][name] <= bound for name, bound in cert["limits"].items())


def assert_no_logical_below_distance(result, d):
    errors = []
    for weight in range(1, d):
        for support in itertools.combinations(range(result.n), weight):
            for paulis in itertools.product(((1, 0), (0, 1), (1, 1)), repeat=weight):
                row = np.zeros(2 * result.n, dtype=np.uint8)
                for q, (x, z) in zip(support, paulis):
                    row[q], row[result.n + q] = x, z
                errors.append(row)
    if errors:
        errors = np.asarray(errors)
        syndrome = (errors[:, :result.n] @ result.check_z.T
                    + errors[:, result.n:] @ result.check_x.T) % 2
        assert in_rowspace(errors[~syndrome.any(axis=1)],
                           np.hstack([result.check_x, result.check_z]))


@pytest.mark.parametrize("route", SINGLES)
def test_single_signed_y_computed_distance_avoids_unnecessary_expansion(route):
    code = hamming_code()
    target = "-YXXZZ" + "I" * 10
    old = route(code, target, require_distance=False)
    assert not old.metadata["distance_preservation_guaranteed"]
    result = route(code, target, require_distance=True)
    assert not result.metadata["distance_postprocessing"]["changed"]
    assert result.metadata["distance_postprocessing"]["expansion_added_edges"] == 0
    assert result.measurement_sign == 1
    assert_actual_measurement(code, result)
    assert_no_logical_below_distance(result, 3)
    assert result.metadata["distance_calculation"]["distance"] == 3
    assert not result.metadata["relative_expansion"]["condition_met"]


@pytest.mark.parametrize("route", PAIRS)
@pytest.mark.parametrize("left", ["-XXIIIIII", "-YXZIIIII"])
def test_thin_adapter_repaired_without_reading_either_signed_factor(route, left):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    right = "IIIIZZII"
    old = route(code, left, right, adapter_size=1, require_distance=False)
    assert not old.metadata["distance_preservation_guaranteed"]
    result = route(code, left, right, adapter_size=1, require_distance=True)
    assert result.metadata["distance_postprocessing"]["changed"]
    assert result.metadata["distance_postprocessing"]["expansion_added_edges"] > 0
    assert result.measurement_sign == 1
    assert_actual_measurement(code, result, excluded=(left, right))
    assert_no_logical_below_distance(result, 2)
    assert result.metadata["initial_construction"]["adapter"]["adapter_size"] == 1


@pytest.mark.parametrize("route", SINGLES)
def test_already_valid_small_single_is_matrix_identical(route):
    code = four_qubit_code()
    old = route(code, "-YXZI", require_distance=False)
    result = route(code, "-YXZI", require_distance=True)
    assert not result.metadata["distance_postprocessing"]["changed"]
    for key in ("check_x", "check_z", "check_signs", "measurement_witness"):
        assert np.array_equal(getattr(result, key), getattr(old, key))
    assert "distance_postprocessing" not in old.metadata
    assert_actual_measurement(code, result)


@pytest.mark.parametrize("route", PAIRS)
def test_already_valid_adapter_is_matrix_identical(route):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    words = ("-XXIIIIII", "IIIIZZII")
    old = route(code, *words, require_distance=False)
    result = route(code, *words, require_distance=True)
    assert not result.metadata["distance_postprocessing"]["changed"]
    for key in ("check_x", "check_z", "check_signs", "measurement_witness"):
        assert np.array_equal(getattr(result, key), getattr(old, key))
    assert_actual_measurement(code, result, excluded=words)


@pytest.mark.parametrize("route,pair", [(r, False) for r in SINGLES] + [(r, True) for r in PAIRS])
def test_unknown_source_distance_is_computed_for_all_public_routes(route, pair):
    known = direct_sum(four_qubit_code(), four_qubit_code()) if pair else four_qubit_code()
    code = CSSCode(known.hx, known.hz)
    args = ("XXIIIIII", "IIIIZZII") if pair else ("XXII",)
    result = route(code, *args, require_distance=True)
    assert result.metadata["source_distance_calculation"]["distance"] == 2


@pytest.mark.parametrize("target", ["ZZZZI", "-YYYYI"])
def test_single_explicit_even_hyperedge_kept_when_actual_distance_is_enough(target):
    code = CSSCode([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0]],
                   np.zeros((0, 5), dtype=np.uint8), distance=1)
    old = gadget_deform(code, target, require_distance=False)
    assert old.metadata["graph_incidence"][0].sum() == 4
    result = gadget_deform(code, target, require_distance=True)
    assert not result.metadata["distance_postprocessing"]["changed"]
    assert np.array_equal(result.metadata["graph_incidence"], old.metadata["graph_incidence"])
    assert_actual_measurement(code, result)


def cellulation_example(**kwargs):
    code = four_qubit_code()
    edges = [(v, (v + 1) % 8) for v in range(8)] + [(0, 4)]
    b = graph(8, edges)
    matching = np.zeros((2, 9), dtype=np.uint8)
    matching[0, 8] = 1       # X memory check anticommutes at Y and Z.
    matching[1, [0, 1]] = 1  # Z memory check anticommutes at Y and X.
    result = gauging_deform(code, "-YXZI", graph=b, ports=[0, 2, 4],
        matching=matching, require_distance=True,
        distance_options=dict(max_cycle_weight=4, **kwargs))
    return code, b, matching, result


def test_cycle_layers_cellulation_and_port_relative_proof_preserve_distance():
    code, b, matching, result = cellulation_example()
    report = result.metadata["distance_postprocessing"]
    sparse = report["sparsification"]
    assert report["expansion_added_edges"] == 0
    assert sparse["layers"] == 2 and sparse["added_chord_count"] == 4
    assert sparse["triangle_count"] == 6
    assert result.metadata["max_cycle_weight"] <= 4
    assert result.metadata["max_cycle_congestion"] <= 3
    final_b, final_m = result.metadata["graph_incidence"], result.metadata["matching_matrix"]
    assert np.array_equal(final_b[:len(b), :8], b)
    assert not np.any(final_b[:len(b), 8:])
    assert np.array_equal(final_m[:, :len(b)], matching)
    assert not np.any(final_m[:, len(b):])
    assert_actual_measurement(code, result)
    assert_no_logical_below_distance(result, 2)


def test_embedded_base_certificate_is_used_when_final_graph_cannot_be_enumerated():
    code = hamming_code()
    b = graph(8, [(v, (v + 1) % 8) for v in range(8)] + [(0, 4)])
    # Five ports make the relative denominator potentially two, so ordinary
    # graph connectedness alone is insufficient.  The base has beta_3=1.
    result = gauging_deform(code, "-YXXZZ" + "I" * 10,
        graph=b, ports=[0, 1, 2, 3, 4], exact_limit=8, require_distance=True,
        distance_options={"max_cycle_weight": 4, "max_matching_congestion": 4})
    cert = result.metadata["relative_expansion"]
    assert result.metadata["graph_incidence"].shape[1] == 16
    assert not cert["exact"] and cert["condition_met"]
    assert cert["verified_from_actual_incidence"]
    assert cert["base_certificate"]["exact"]
    assert cert["base_certificate"]["beta"] == 1
    assert_actual_measurement(code, result)
    assert_no_logical_below_distance(result, 3)


def test_forced_decongestion_reduces_cycle_edge_participation():
    code = four_qubit_code()
    edges = [(v, (v + 1) % 8) for v in range(8)] + [(v, v + 4) for v in range(4)]
    b = graph(8, edges)
    matching = np.zeros((2, 12), dtype=np.uint8)
    matching[1, 8] = 1
    old = gauging_deform(code, "-XXII", graph=b, ports=[0, 4], matching=matching, require_distance=False)
    result = gauging_deform(code, "-XXII", graph=b, ports=[0, 4], matching=matching,
        require_distance=True, distance_options={"max_cycle_weight": 4, "max_cycle_congestion": 3, "max_check_weight": 7})
    sparse = result.metadata["distance_postprocessing"]["sparsification"]
    assert sparse["layers"] == 5 and sparse["added_chord_count"] == 6
    assert old.metadata["max_cycle_congestion"] > result.metadata["max_cycle_congestion"]
    assert result.metadata["max_cycle_congestion"] <= 3
    assert result.metadata["max_cycle_weight"] <= 4
    assert_actual_measurement(code, result)
    assert_no_logical_below_distance(result, 2)


@pytest.mark.parametrize("options,match", [
    ({"max_layers": 1}, "layers"),
    ({"max_cycle_congestion": 1}, "sparsity"),
    ({"max_check_weight": 2}, "sparsity"),
    ({"max_qubit_degree": 2}, "sparsity"),
])
def test_unattainable_final_limits_raise_without_partial_result(options, match):
    with pytest.raises(ValueError, match=match):
        cellulation_example(**options)


@pytest.mark.parametrize("kwargs", [
    {"require_distance": 1},
    {"require_distance": True, "distance_options": []},
    {"require_distance": True, "distance_options": {"max_base_degree": False}},
    {"require_distance": True, "distance_options": {"max_cycle_weight": 0}},
    {"require_distance": True, "distance_options": {"max_matching_weight": -1}},
    {"require_distance": True, "distance_options": {"max_qubit_degree": True}},
    {"require_distance": True, "distance_options": {"max_entries": False}},
])
def test_bad_options_are_rejected_even_for_existing_passing_gadget(kwargs):
    with pytest.raises(ValueError):
        gadget_deform(four_qubit_code(), "XXII", **kwargs)


def test_distance_opt_out_ignores_well_formed_options_but_rejects_wrong_type():
    code = hamming_code()
    target = "-YXXZZ" + "I" * 10
    old = gadget_deform(code, target, require_distance=False)
    result = gadget_deform(code, target, require_distance=False,
                            distance_options={"max_layers": 1, "max_base_degree": 1})
    assert np.array_equal(old.check_x, result.check_x)
    assert np.array_equal(old.check_z, result.check_z)
    assert "distance_postprocessing" not in result.metadata
    assert not result.metadata["distance_preservation_guaranteed"]
    with pytest.raises(ValueError, match="dictionary"):
        gadget_deform(code, target, require_distance=False, distance_options=[])


@pytest.mark.parametrize("mutation", ["boolean", "uncertified", "wrong_prefix", "wrong_ports", "wrong_distance"])
def test_renderer_rejects_forged_postprocessing_proofs(mutation):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    old = gadget_pair_deform(code, "XXIIIIII", "IIIIZZII", adapter_size=1, require_distance=False)
    md = old.metadata
    b = md["graph_incidence"]
    proof = dict(base_graph=b.copy(), base_ports=list(md["ports"]), distance=2,
                 condition_met=True, exact_limit=20, certificate_limit=128)
    if mutation == "boolean":
        proof = {"condition_met": True}
    elif mutation == "wrong_prefix":
        proof["base_graph"] = b[::-1].copy()
    elif mutation == "wrong_ports":
        proof["base_ports"] = list(reversed(proof["base_ports"]))
    elif mutation == "wrong_distance":
        proof["distance"] = 1
    with pytest.raises(ValueError, match="proof|certificate|base"):
        build_graph(code, (old.measurement_x, old.measurement_z, old.measurement_sign),
            md["logical_basis"], b, ports=md["ports"], matching=md["matching_matrix"],
            cycles=md["cycle_basis"], extra_metadata={"postprocessing_distance_proof": proof})


def test_ensure_distance_cannot_reuse_tampered_sealed_certificate():
    old = gadget_deform(four_qubit_code(), "XXII", require_distance=False)
    copy_result = copy.deepcopy(old)
    copy_result.metadata["distance_preservation_guaranteed"] = False
    with pytest.raises(ValueError, match="changed"):
        ensure_distance(copy_result)
    assert old.validate()


def test_reprocessing_certified_result_is_idempotent_and_does_not_mutate_it():
    old = gadget_deform(hamming_code(), "-YXXZZ" + "I" * 10, require_distance=True)
    before = old._fingerprint()
    result = ensure_distance(old)
    assert before == old._fingerprint()
    assert not result.metadata["distance_postprocessing"]["changed"]
    assert np.array_equal(old.check_x, result.check_x)
    assert np.array_equal(old.check_z, result.check_z)
    assert_actual_measurement(hamming_code(), result)


def test_postprocessing_budget_is_checked_even_on_fast_path():
    old = gadget_deform(four_qubit_code(), "XXII", require_distance=False)
    with pytest.raises(ValueError, match="max_entries"):
        ensure_distance(old, max_entries=1)
    assert "distance_postprocessing" not in old.metadata
