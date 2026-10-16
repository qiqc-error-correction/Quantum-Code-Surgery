"""MILP certificates, Pauli weight, degeneracy, and interrupted solver safety."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from codesurgery.distance_preservation._distance_milp import milp_distance
from codesurgery.distance_preservation.distance import _enumeration_distance
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace
from codesurgery.symplectic import parse_pauli

scipy = pytest.importorskip("scipy.optimize")


def checks(words):
    parsed = [parse_pauli(word, len(words[0])) for word in words]
    return np.array([row[0] for row in parsed]), np.array([row[1] for row in parsed])


def css_checks(code):
    return (np.vstack([code.hx, np.zeros_like(code.hz)]),
            np.vstack([np.zeros_like(code.hx), code.hz]))


def seed(x, z, **kwargs):
    return _enumeration_distance(x, z, max_candidates=0, **kwargs)


def weakened_seed(x, z, word):
    """Use a deliberately nonminimal but independently checked initial witness."""
    result = seed(x, z)
    wx, wz, _ = parse_pauli(word, x.shape[1])
    assert not np.any((x @ wz + z @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack([x, z]))
    weight = int((wx | wz).sum())
    result.update(status="unknown", distance=None, proven_lower_bound=1,
                  upper_bound=weight,
                  witness={"x": wx.tolist(), "z": wz.tolist(), "pauli": word, "weight": weight})
    return result


def verify(x, z, report):
    witness = report["witness"]
    wx, wz = np.array(witness["x"]), np.array(witness["z"])
    assert not np.any((x @ wz + z @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack([x, z]))
    assert int((wx | wz).sum()) == witness["weight"] == report["upper_bound"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("code", [four_qubit_code(), steane_code()])
def test_css_matches_complete_enumeration(code):
    x, z = css_checks(code)
    initial = seed(x, z)
    saved = deepcopy(initial)
    result = milp_distance(x, z, initial)
    reference = _enumeration_distance(x, z)
    assert result["status"] == "exact"
    assert result["distance"] == reference["distance"]
    assert result["milp"]["proof"] == "milp_infeasible_cutoff"
    assert initial == saved
    verify(x, z, result)


def test_non_css_optimum_improves_initial_upper_bound():
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    initial = weakened_seed(x, z, "XXXXX")
    result = milp_distance(x, z, initial)
    assert result["distance"] == 3 == _enumeration_distance(x, z)["distance"]
    assert result["milp"]["proof"] == "milp_optimal"
    assert result["milp"]["witness_validated"]
    assert result["milp"]["logical_basis_rank"] == 2
    verify(x, z, result)


def test_y_costs_one_physical_qubit():
    x, z = checks(["YY"])
    result = milp_distance(x, z, weakened_seed(x, z, "XX"))
    assert result["distance"] == 1
    assert "Y" in result["witness"]["pauli"]
    assert sum(result["witness"]["x"]) + sum(result["witness"]["z"]) == 2
    verify(x, z, result)


@pytest.mark.parametrize("ancilla", ["Z", "Y"])
def test_weight_one_ancilla_stabilizer_is_excluded(ancilla):
    x, z = checks(["XXXXI", "ZZZZI", "IIII" + ancilla])
    result = milp_distance(x, z, seed(x, z))
    assert result["distance"] == 2
    assert result["milp"]["logical_basis_rank"] == 4
    assert result["milp"]["proof"] == "milp_infeasible_cutoff"
    verify(x, z, result)


def test_final_code_basis_finds_free_auxiliary_logical():
    x, z = checks(["XXXXI", "ZZZZI"])
    result = milp_distance(x, z, weakened_seed(x, z, "XXIII"))
    assert result["distance"] == 1
    assert result["witness"]["pauli"][:4] == "IIII"
    assert result["milp"]["logical_basis_rank"] == 6
    verify(x, z, result)


def test_disconnected_code_and_redundant_checks():
    x, z = checks(["XXXXIIII", "ZZZZIIII", "IIIIXXXX", "IIIIZZZZ",
                   "XXXXIIII", "IIIIIIII"])
    result = milp_distance(x, z, seed(x, z))
    assert result["distance"] == 2
    assert result["milp"]["logical_basis_rank"] == 8
    verify(x, z, result)


def test_threshold_proves_only_requested_range():
    x, z = css_checks(steane_code())
    result = milp_distance(x, z, seed(x, z, max_weight=1))
    assert result["status"] == "bounded" and result["distance"] is None
    assert result["proven_lower_bound"] == 2 and result["upper_bound"] == 3
    assert result["milp"]["cutoff"] == 1
    verify(x, z, result)


@pytest.mark.parametrize("rng_seed", range(4))
def test_mixed_small_codes_crosscheck_enumeration(rng_seed):
    x, z = css_checks(four_qubit_code())
    rng = np.random.default_rng(rng_seed)
    for _ in range(5):
        a, b = rng.choice(4, size=2, replace=False)
        x[:, b] ^= x[:, a]
        z[:, a] ^= z[:, b]
        z[:, a] ^= x[:, a]
    result = milp_distance(x, z, seed(x, z))
    assert result["distance"] == _enumeration_distance(x, z)["distance"]
    verify(x, z, result)


def test_timeout_keeps_only_validated_upper_bound(monkeypatch):
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    initial = weakened_seed(x, z, "XXXXX")
    actual_solver = scipy.milp

    def timed_out(**kwargs):
        result = actual_solver(**kwargs)
        assert result.status == 0
        result.status = 1
        result.message = "Mock node limit after a feasible incumbent"
        result.mip_dual_bound = 99.99999
        return result

    monkeypatch.setattr(scipy, "milp", timed_out)
    result = milp_distance(x, z, initial, node_limit=10000)
    assert result["upper_bound"] == 3
    assert result["proven_lower_bound"] == 1
    assert result["distance"] is None and result["status"] == "unknown"
    assert result["milp"]["proof"] is None
    assert result["stop_reason"] == "milp_time_or_node_limit"
    verify(x, z, result)


@pytest.mark.parametrize("mode", ["zero", "fractional", "nan", "wrong_shape"])
def test_invalid_optimal_solver_output_cannot_certify(monkeypatch, mode):
    x, z = css_checks(steane_code())
    initial = seed(x, z)

    def invalid(**kwargs):
        vector = np.zeros_like(kwargs["c"])
        if mode == "fractional":
            vector[:] = 0.5
        elif mode == "nan":
            vector[:] = np.nan
        elif mode == "wrong_shape":
            vector = vector[:-1]
        return SimpleNamespace(status=0, x=vector, fun=0.0, message="Mock invalid optimum")

    monkeypatch.setattr(scipy, "milp", invalid)
    result = milp_distance(x, z, initial)
    assert result["status"] == "unknown" and result["distance"] is None
    assert result["upper_bound"] == initial["upper_bound"]
    assert result["proven_lower_bound"] == 1
    assert not result["milp"]["witness_validated"]
    assert result["stop_reason"] == "milp_invalid_solution"


def test_bad_solver_objective_retains_witness_without_exact_claim(monkeypatch):
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    actual_solver = scipy.milp

    def wrong_objective(**kwargs):
        result = actual_solver(**kwargs)
        result.fun = 1.0
        return result

    monkeypatch.setattr(scipy, "milp", wrong_objective)
    result = milp_distance(x, z, weakened_seed(x, z, "XXXXX"))
    assert result["upper_bound"] == 3 and result["proven_lower_bound"] == 1
    assert result["distance"] is None and result["milp"]["proof"] is None
    assert result["stop_reason"] == "milp_invalid_solution"
    verify(x, z, result)


def test_solver_exception_does_not_change_bounds(monkeypatch):
    x, z = css_checks(steane_code())
    initial = seed(x, z)

    def broken(**kwargs):
        raise RuntimeError("mock solver error")

    monkeypatch.setattr(scipy, "milp", broken)
    result = milp_distance(x, z, initial)
    assert result["upper_bound"] == 3 and result["proven_lower_bound"] == 1
    assert result["distance"] is None and result["milp"]["proof"] is None
    assert result["stop_reason"] == "milp_solver_error"


@pytest.mark.parametrize("kwargs", [dict(time_limit=-1), dict(time_limit=np.inf),
                                   dict(time_limit=True), dict(node_limit=-1),
                                   dict(node_limit=True), dict(node_limit=0.5)])
def test_bad_resource_limits(kwargs):
    x, z = css_checks(steane_code())
    with pytest.raises(ValueError):
        milp_distance(x, z, seed(x, z), **kwargs)
