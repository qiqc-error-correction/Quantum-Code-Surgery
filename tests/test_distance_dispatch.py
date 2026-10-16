"""Public backend dispatch, independently checked RIS witnesses and tri-state decisions."""
import json

import numpy as np
import pytest

from codesurgery.distance_preservation import code_distance, stabilizer_distance, verify_distance
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace
from codesurgery.symplectic import parse_pauli


def five_qubit_checks():
    rows = [parse_pauli(s, 5) for s in ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def check_witness(x, z, report):
    wx, wz = np.array(report["witness"]["x"]), np.array(report["witness"]["z"])
    assert not np.any((x @ wz + z @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack([x, z]))
    assert int((wx | wz).sum()) == report["upper_bound"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("seed", range(8))
def test_ris_mixed_witnesses_and_bounds(seed):
    x, z = five_qubit_checks()
    # Local Clifford gates preserve distance but introduce arbitrary Y entries.
    rng = np.random.default_rng(seed)
    for q in range(5):
        if rng.integers(2):
            x[:, q], z[:, q] = z[:, q].copy(), x[:, q].copy()
        if rng.integers(2):
            z[:, q] ^= x[:, q]
    a = stabilizer_distance(x, z, method="ris", ris_trials=20, seed=seed)
    assert a["proven_lower_bound"] == 1 and a["upper_bound"] >= 3
    assert a["distance"] is None and a["status"] == "bounded"
    assert a["ris"]["trials_completed"] == 20
    assert a == stabilizer_distance(x, z, method="ris", ris_trials=20, seed=seed)
    check_witness(x, z, a)


def test_ris_excludes_low_weight_stabilizers():
    code = four_qubit_code()
    x = np.pad(np.vstack([code.hx, np.zeros_like(code.hz)]), ((0, 1), (0, 1)))
    z = np.pad(np.vstack([np.zeros_like(code.hx), code.hz]), ((0, 1), (0, 1)))
    x[-1, -1] = z[-1, -1] = 1
    r = stabilizer_distance(x, z, method="ris", ris_trials=32)
    assert r["upper_bound"] == 2 and r["proven_lower_bound"] == 1
    check_witness(x, z, r)


def disable_optional_gurobi(monkeypatch):
    """Exercise the dependency-free fallback even on a licensed test host."""
    from codesurgery.distance_preservation import _distance_gurobi
    def unavailable(*args, **kwargs):
        raise ImportError("Gurobi unavailable for fallback-path test")
    monkeypatch.setattr(_distance_gurobi, "gurobi_distance", unavailable)


def test_auto_css_falls_back_to_clusters_with_separate_ris_budget(monkeypatch):
    disable_optional_gurobi(monkeypatch)
    r = code_distance(steane_code())
    assert r["backend"] == "cluster" and r["distance"] == 3
    assert r["requested_method"] == "auto" and r["ris"]["trials_completed"] == 64
    limited = code_distance(steane_code(), max_candidates=0, ris_trials=4)
    assert limited["checked_candidates"] == 0 and limited["ris"]["trials_completed"] == 4
    assert limited["proven_lower_bound"] == 1 and limited["distance"] is None


def test_missing_milp_auto_falls_back_but_explicit_choice_is_clear(monkeypatch):
    disable_optional_gurobi(monkeypatch)
    from codesurgery.distance_preservation import _distance_milp
    def unavailable(*args, **kwargs):
        raise ImportError("no SciPy")
    monkeypatch.setattr(_distance_milp, "milp_distance", unavailable)
    x, z = five_qubit_checks()
    r = stabilizer_distance(x, z, ris_trials=0)
    assert r["backend"] == "cluster" and r["distance"] == 3
    assert "unavailable" in r["fallback_reason"]
    with pytest.raises(ImportError, match="codesurgery\\[distance\\]"):
        stabilizer_distance(x, z, method="milp")


def test_incomplete_milp_auto_returns_bounds_without_another_search(monkeypatch):
    disable_optional_gurobi(monkeypatch)
    from codesurgery.distance_preservation import _distance_cluster, _distance_milp
    def incomplete(x, z, initial, **kwargs):
        return dict(initial, milp={"status": 1}, stop_reason="milp_time_or_node_limit")
    def forbidden(*args, **kwargs):
        raise AssertionError("A solver timeout must not start a second search")
    monkeypatch.setattr(_distance_milp, "milp_distance", incomplete)
    monkeypatch.setattr(_distance_cluster, "cluster_distance", forbidden)
    x, z = five_qubit_checks()
    r = stabilizer_distance(x, z, ris_trials=0)
    assert r["backend"] == "milp" and r["milp"]["status"] == 1
    assert r["distance"] is None and r["proven_lower_bound"] == 1
    assert r["stop_reason"] == "milp_time_or_node_limit"
    check_witness(x, z, r)


@pytest.mark.parametrize("css", [False, True])
@pytest.mark.parametrize("options,expected", [({}, (60.0, None)),
    ({"time_limit": 1.25, "node_limit": 17}, (1.25, 17)),
    ({"time_limit": None}, (None, None))])
def test_auto_prefers_gurobi_for_css_and_mixed_and_propagates_budget(monkeypatch, css, options, expected):
    from codesurgery.distance_preservation import _distance_cluster, _distance_gurobi, _distance_milp
    seen = []
    def limited(x, z, initial, *, time_limit, node_limit):
        seen.append((time_limit, node_limit))
        return dict(initial, status="bounded", proven_lower_bound=2,
                    gurobi={"status": "TIME_LIMIT"}, stop_reason="gurobi_time_or_node_limit")
    def forbidden(*args, **kwargs):
        raise AssertionError("A completed Gurobi call must return its bounds directly")
    monkeypatch.setattr(_distance_gurobi, "gurobi_distance", limited)
    monkeypatch.setattr(_distance_milp, "milp_distance", forbidden)
    monkeypatch.setattr(_distance_cluster, "cluster_distance", forbidden)
    if css:
        code = steane_code()
        x = np.vstack([code.hx, np.zeros_like(code.hz)])
        z = np.vstack([np.zeros_like(code.hx), code.hz])
    else:
        x, z = five_qubit_checks()
    r = stabilizer_distance(x, z, ris_trials=0, **options)
    assert seen == [expected]
    assert r["backend"] == "gurobi" and r["requested_method"] == "auto"
    assert r["gurobi"]["status"] == "TIME_LIMIT" and r["distance"] is None
    assert r["stop_reason"] == "gurobi_time_or_node_limit"
    assert r["distance_interval"] == dict(lower_bound=2, upper_bound=r["upper_bound"],
        lower_inclusive=True, upper_inclusive=True, distance_is_infinite=False)
    assert r["distance_statement"] == f"1 < d <= {r['upper_bound']}"
    check_witness(x, z, r)


def test_explicit_gurobi_unavailable_has_clear_error(monkeypatch):
    disable_optional_gurobi(monkeypatch)
    with pytest.raises(ImportError, match=r"codesurgery\[gurobi\].*method='milp'/'cluster'"):
        code_distance(steane_code(), method="gurobi", ris_trials=0)


def test_public_distance_intervals_preserve_exact_and_no_logical_cases():
    exact = code_distance(steane_code(), method="cluster", ris_trials=0)
    assert exact["distance_statement"] == "d = 3"
    assert exact["distance_interval"]["lower_bound"] == exact["distance_interval"]["upper_bound"] == 3
    from codesurgery.models import CSSCode
    state = code_distance(CSSCode([[1]], np.zeros((0, 1), dtype=int)), method="cluster")
    assert state["distance_interval"]["distance_is_infinite"] is True
    assert state["distance_interval"]["lower_bound"] is None
    assert state["distance_interval"]["upper_bound"] is None
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("method", ["auto", "enumeration", "cluster", "ris"])
def test_threshold_disproof_always_has_a_valid_witness(method):
    r = verify_distance(steane_code(), 4, method=method)
    assert r["target_met"] is False and r["upper_bound"] < 4
    assert r["target_status"] == "insufficient"


def test_threshold_success_is_distinct_from_exact_distance():
    # Explicitly test completed cluster shells, independently of solver choice.
    r = verify_distance(steane_code(), 2, method="cluster")
    assert r["target_met"] is True and r["target_status"] == "sufficient"
    assert r["proven_lower_bound"] == 2 and r["distance"] is None
    assert r["max_weight"] == 1
    r = verify_distance(steane_code(), 3, max_weight=0, method="cluster")
    assert r["target_met"] is None and r["target_status"] == "unknown"


def test_threshold_one_needs_no_enumeration_and_state_has_no_logicals():
    r = verify_distance(steane_code(), 1, ris_trials=0)
    assert r["target_met"] is True and r["checked_candidates"] == 0
    from codesurgery.models import CSSCode
    r = verify_distance(CSSCode([[1]], np.zeros((0, 1), dtype=int)), 100)
    assert r["target_met"] is True and r["distance_is_infinite"]


@pytest.mark.parametrize("kwargs,key", [({"method":"bad"}, "method"),
    ({"ris_trials":True}, "ris_trials"), ({"seed":-1}, "seed"),
    ({"time_limit":float("nan")}, "time_limit"), ({"time_limit":True}, "time_limit"),
    ({"node_limit":1.2}, "node_limit"), ({"max_candidates":False}, "max_candidates")])
def test_invalid_new_options_rejected_before_early_exit(kwargs, key):
    with pytest.raises(ValueError, match=key):
        stabilizer_distance(np.zeros((0, 1)), np.zeros((0, 1)), **kwargs)


@pytest.mark.parametrize("target", [0, -1, True, 1.5])
def test_bad_threshold(target):
    with pytest.raises(ValueError, match="target_distance"):
        verify_distance(steane_code(), target)


def test_threshold_counterexample_skips_all_unneeded_search(monkeypatch):
    from codesurgery.distance_preservation import _distance_ris, _distance_cluster, _distance_milp, _distance_gurobi
    def must_not_search(*args, **kwargs):
        raise AssertionError("A valid upper bound below the target already disproves preservation")
    monkeypatch.setattr(_distance_ris, "ris_distance", must_not_search)
    monkeypatch.setattr(_distance_cluster, "cluster_distance", must_not_search)
    monkeypatch.setattr(_distance_milp, "milp_distance", must_not_search)
    monkeypatch.setattr(_distance_gurobi, "gurobi_distance", must_not_search)
    for method in ("auto", "enumeration", "milp", "gurobi", "cluster", "ris"):
        r = verify_distance(steane_code(), 4, method=method)
        assert r["target_met"] is False and r["upper_bound"] == 3
        assert r["distance"] is None and r["backend"] == "linear_algebra"
