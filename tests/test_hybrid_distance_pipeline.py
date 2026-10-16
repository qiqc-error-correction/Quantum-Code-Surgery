"""Distance backends in the automatic surgery pipeline."""
import numpy as np
import pytest

from codesurgery.examples import direct_sum, four_qubit_code
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.models import CSSCode


@pytest.mark.parametrize("route,pair", [
    (gadget_deform, False), (gauging_deform, False),
    (gadget_pair_deform, True), (gauging_pair_deform, True),
])
def test_backend_options_reach_source_and_every_final_check(monkeypatch, route, pair):
    from codesurgery.distance_preservation import distance

    calls = []
    real_verify = distance.verify_distance

    def tracked_verify(code, target_distance, **kwargs):
        calls.append((code.n, target_distance, dict(kwargs)))
        return real_verify(code, target_distance, **kwargs)

    monkeypatch.setattr(distance, "verify_distance", tracked_verify)
    code = direct_sum(four_qubit_code(), four_qubit_code()) if pair else four_qubit_code()
    targets = ("XXIIIIII", "IIIIZZII") if pair else ("XXII",)
    options = dict(distance_method="cluster", distance_ris_trials=3,
                   distance_seed=19, distance_time_limit=5.0,
                   distance_node_limit=1000, distance_max_candidates=20000,
                   distance_max_weight=7)
    result = route(code, *targets, distance_options=options)
    assert result.validate()
    assert len(calls) >= 2
    assert calls[0][0] == code.n
    assert calls[-1][0] == result.n
    assert all(target == code.distance for _, target, _ in calls)
    expected = dict(method="cluster", ris_trials=3, seed=19, time_limit=5.0,
                    node_limit=1000, max_candidates=20000, max_weight=7)
    assert all(kwargs == expected for _, _, kwargs in calls)
    report = result.metadata["distance_calculation"]
    assert report["target_met"] is True
    assert report["max_weight"] <= code.distance - 1
    assert result.metadata["distance_postprocessing"]["distance_search_options"] == expected


def test_threshold_certificate_does_not_claim_exact_distance():
    known = four_qubit_code()
    # A conservative trusted target deliberately below the actual d=2.
    code = CSSCode(known.hx, known.hz, distance=1)
    result = gauging_deform(code, "XXII", distance_options={
        "distance_method": "enumeration", "distance_ris_trials": 0,
        "distance_max_weight": 0,
    })
    for key in ("source_distance_calculation", "distance_calculation"):
        report = result.metadata[key]
        assert report["target_distance"] == 1
        assert report["target_met"] is True
        assert report["proven_lower_bound"] == 1
        assert report["upper_bound"] == 2
        assert report["distance"] is None
        assert report["max_weight"] == 0
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] == 1


def test_undeclared_source_requires_and_gets_exact_distance():
    known = four_qubit_code()
    code = CSSCode(known.hx, known.hz)
    result = gadget_deform(code, "XXII", distance_options={"distance_method": "cluster"})
    source = result.metadata["source_distance_calculation"]
    assert source["status"] == "exact"
    assert source["distance"] == 2
    assert code.distance is None
    assert result.metadata["distance_calculation"]["target_met"] is True


def test_explicit_error_policy_does_not_expand_an_unknown_distance(monkeypatch):
    from codesurgery.distance_preservation import api

    def forbidden_expansion(*args, **kwargs):
        pytest.fail("Unknown distance must not trigger expansion")

    monkeypatch.setattr(api, "enhance_expansion", forbidden_expansion)
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)]
                  for bit in (3, 2, 1, 0)], dtype=np.uint8)
    with pytest.raises(ValueError, match="undetermined"):
        gauging_deform(CSSCode(h, h, distance=3), "-YXXZZ" + "I" * 10,
            distance_options={"distance_method": "enumeration",
                              "distance_ris_trials": 0, "distance_max_candidates": 1,
                              "unknown_distance_policy": "error"})


@pytest.mark.parametrize("options", [
    {"distance_method": "typo"},
    {"distance_ris_trials": -1},
    {"distance_time_limit": -1},
    {"distance_node_limit": -1},
    {"distance_seed": True},
    {"distance_unused_option": 1},
])
def test_invalid_backend_options_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        gadget_deform(four_qubit_code(), "XXII", distance_options=options)


def test_zero_enumeration_budget_does_not_disable_milp():
    pytest.importorskip("scipy.optimize")
    result = gauging_deform(four_qubit_code(), "XXII", distance_options={
        "distance_method": "milp", "distance_max_candidates": 0})
    report = result.metadata["distance_calculation"]
    assert report["target_met"] is True
    assert report["checked_candidates"] == 0 and report["backend"] == "milp"
