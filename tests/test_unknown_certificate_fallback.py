"""Continue finite graph certification after a distance search exhausts its budget.

The small signed mixed-Pauli examples use real distance searches with a one-
candidate budget, rather than pretending an unknown result is a distance loss.
The graph-cut oracle below is independent of the production certificate code.
"""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from codesurgery.distance_preservation import ensure_distance
from codesurgery.distance_preservation.expansion import ExpansionConstructionError
from codesurgery.examples import four_qubit_code
from codesurgery.explicit_gadgets import gadget_deform
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli


UNKNOWN_OPTIONS = dict(distance_method="enumeration", distance_ris_trials=0,
                       distance_max_candidates=1, attempts=1)
MIXED_TARGET = "-YXXZZ" + "I" * 10


def hamming_code(*, distance=3):
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)]
                  for bit in (3, 2, 1, 0)], dtype=np.uint8)
    return CSSCode(h, h, distance=distance)


def exact_global_h(graph):
    """Count crossing edges directly over every cut, including parallel edges."""
    edges = [set(np.flatnonzero(row)) for row in graph]
    return min(Fraction(sum(len(edge & set(subset)) == 1 for edge in edges), size)
               for size in range(1, graph.shape[1] // 2 + 1)
               for subset in combinations(range(graph.shape[1]), size))


def assert_graph_certificate_and_fixed_limits(result):
    assert result.validate()
    md = result.metadata
    report = md["distance_postprocessing"]
    assert report["initial_distance_state"] == "unknown"
    assert report["expansion_trigger"] == "distance certificate unavailable within search budget"
    assert report["expansion"]["acceptance"] == "expansion"
    assert report["expansion_added_edges"] > 0
    # Numerical search remains inconclusive. Only the separate graph theorem
    # establishes the output lower bound, conditional on trusted source d=3.
    for search in (report["initial_distance_calculation"], md["distance_calculation"]):
        assert search["target_met"] is None
        assert search["distance"] is None
        assert search["proven_lower_bound"] < 3
        assert search["upper_bound"] >= 3
    assert md["distance_preservation_guaranteed"]
    assert md["distance_lower_bound"] == 3
    assert md["relative_expansion"]["condition_met"]
    assert exact_global_h(md["graph_incidence"]) > 1
    assert report["sparsification"]["layers"] == 1

    # Independently inspect the complete output checks; a favorable graph
    # degree alone is insufficient to certify the full check weights.
    support = result.check_x | result.check_z
    limits = report["sparsity_certificate"]["limits"]
    assert limits["max_check_weight"] == report["sparsity_threshold"] == 9
    assert int(support.sum(axis=1).max()) <= limits["max_check_weight"]
    assert int(support.sum(axis=0).max()) <= limits["max_qubit_degree"]
    assert not np.any((result.check_x @ result.check_z.T
                       + result.check_z @ result.check_x.T) % 2)


@pytest.mark.parametrize("builder", [gauging_deform, gadget_deform])
def test_default_unknown_search_continues_to_a_verified_graph_certificate(builder):
    code = hamming_code()
    basic = builder(code, MIXED_TARGET, require_distance=False)
    assert not basic.metadata["relative_expansion"]["condition_met"]
    assert exact_global_h(basic.metadata["graph_incidence"]) < 1
    result = ensure_distance(basic, **UNKNOWN_OPTIONS)
    assert_graph_certificate_and_fixed_limits(result)
    assert result.metadata["distance_postprocessing"]["unknown_distance_policy"] == "certify"
    np.testing.assert_array_equal(result.measurement_x, basic.measurement_x)
    np.testing.assert_array_equal(result.measurement_z, basic.measurement_z)
    assert result.measurement_sign == basic.measurement_sign == 1
    # Successful processing is nonmutating even when the search is budgeted.
    assert basic.n == 19
    assert not basic.metadata["relative_expansion"]["condition_met"]


def test_reduced_mixed_target_continues_optimization_and_preserves_its_logical_class():
    base = hamming_code()
    # Fixed Z ancillas make weight 8 > 2d. Reduction removes them and retains
    # the weight-five target whose initial graph has beta_d=1/2.
    hx = np.pad(base.hx, ((0, 0), (0, 3)))
    hz = np.vstack([np.pad(base.hz, ((0, 0), (0, 3))),
                    np.pad(np.eye(3, dtype=np.uint8), ((0, 0), (15, 0)))])
    code = CSSCode(hx, hz, distance=3)
    target = MIXED_TARGET + "ZZZ"
    result = gauging_deform(code, target, optimize_logical=True,
        logical_optimization_options=dict(method="heuristic", max_trials=1, seed=0),
        distance_options=UNKNOWN_OPTIONS)
    assert_graph_certificate_and_fixed_limits(result)
    reduction = result.metadata["logical_optimization"]
    assert reduction["before_weight"] == 8
    assert reduction["after_weight"] == 5
    assert reduction["threshold"] == 6
    assert reduction["equivalence_verified"] and reduction["signature_verified"]
    original_x, original_z, original_sign = parse_pauli(target, code.n)
    basis = code.standard_logicals()
    np.testing.assert_array_equal(result.measurement_x @ basis.jz.T % 2,
                                  original_x @ basis.jz.T % 2)
    np.testing.assert_array_equal(result.measurement_z @ basis.jx.T % 2,
                                  original_z @ basis.jx.T % 2)
    # This representative change only drops the positive Z stabilizers of
    # fixed ancillas, so the requested minus sign stays a literal minus sign.
    assert result.measurement_sign == original_sign == 1
    assert not np.any(result.measurement_x[-3:] | result.measurement_z[-3:])


def test_unknown_search_with_existing_certificate_does_not_add_edges(monkeypatch):
    from codesurgery.distance_preservation import api

    def forbidden(*args, **kwargs):
        pytest.fail("An existing distance certificate needs no further expansion")

    monkeypatch.setattr(api, "enhance_expansion", forbidden)
    result = gauging_deform(four_qubit_code(), "XXII", distance_options=UNKNOWN_OPTIONS)
    assert result.metadata["distance_calculation"]["target_met"] is None
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_postprocessing"]["expansion_added_edges"] == 0
    assert result.metadata["distance_postprocessing"]["expansion_trigger"] == "none"


def test_unknown_search_cannot_override_an_exhausted_degree_budget():
    # A degree-two graph on five ports cannot satisfy the strict h>1 rule.
    # Search may try it, but must not increase the cap or return unverified.
    with pytest.raises(ExpansionConstructionError):
        gauging_deform(hamming_code(), MIXED_TARGET,
            distance_options=dict(UNKNOWN_OPTIONS, max_base_degree=2))


def test_undeclared_unknown_source_still_requires_a_trusted_distance(monkeypatch):
    from codesurgery.distance_preservation import api

    def forbidden(*args, **kwargs):
        pytest.fail("Graph expansion cannot establish the unknown source distance")

    monkeypatch.setattr(api, "enhance_expansion", forbidden)
    with pytest.raises(ValueError, match="Source distance is undetermined"):
        gauging_deform(hamming_code(distance=None), MIXED_TARGET,
                       distance_options=UNKNOWN_OPTIONS)


@pytest.mark.parametrize("policy", [None, True, 1, "", "continue", "unknown"])
def test_unknown_distance_policy_is_validated_even_for_already_certified_codes(policy):
    with pytest.raises(ValueError, match="unknown_distance_policy"):
        gauging_deform(four_qubit_code(), "XXII",
                       distance_options={"unknown_distance_policy": policy})
