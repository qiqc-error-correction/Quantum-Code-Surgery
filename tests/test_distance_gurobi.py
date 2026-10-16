"""Gurobi exact-distance controls and conservative interrupted-solve bounds."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from codesurgery.distance_preservation import _distance_gurobi as backend
from codesurgery.distance_preservation._distance_milp import milp_distance
from codesurgery.distance_preservation.distance import _enumeration_distance
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace
from codesurgery.symplectic import parse_pauli


def checks(words):
    parsed = [parse_pauli(word, len(words[0])) for word in words]
    return np.array([row[0] for row in parsed]), np.array([row[1] for row in parsed])


def css_checks(code):
    return (np.vstack([code.hx, np.zeros_like(code.hz)]),
            np.vstack([np.zeros_like(code.hx), code.hz]))


def seed(x, z, **kwargs):
    return _enumeration_distance(x, z, max_candidates=0, **kwargs)


def weakened_seed(x, z, word):
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


def real_solve(x, z, initial, **kwargs):
    pytest.importorskip("gurobipy")
    try:
        return backend.gurobi_distance(x, z, initial, **kwargs)
    except backend.GurobiUnavailableError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("code", [four_qubit_code(), steane_code()])
def test_real_css_matches_complete_enumeration(code):
    x, z = css_checks(code)
    initial = seed(x, z)
    saved = deepcopy(initial)
    result = real_solve(x, z, initial)
    assert result["distance"] == _enumeration_distance(x, z)["distance"]
    assert result["gurobi"]["proof"] == "gurobi_infeasible_cutoff"
    assert result["gurobi"]["time_limit"] == 60
    assert result["gurobi"]["status_name"] == "INFEASIBLE"
    assert initial == saved
    verify(x, z, result)


def test_real_non_css_optimum_improves_witness():
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    result = real_solve(x, z, weakened_seed(x, z, "XXXXX"))
    assert result["distance"] == 3 == _enumeration_distance(x, z)["distance"]
    assert result["gurobi"]["proof"] == "gurobi_optimal"
    assert result["gurobi"]["logical_basis_rank"] == 2
    assert result["gurobi"]["witness_validated"]
    verify(x, z, result)


def test_real_y_costs_one_qubit():
    x, z = checks(["YY"])
    result = real_solve(x, z, weakened_seed(x, z, "XX"))
    assert result["distance"] == 1 and "Y" in result["witness"]["pauli"]
    verify(x, z, result)


def test_real_unlimited_limits_can_be_requested():
    x, z = css_checks(four_qubit_code())
    result = real_solve(x, z, seed(x, z), time_limit=None, node_limit=None)
    assert result["distance"] == 2
    assert result["gurobi"]["time_limit"] is None and result["gurobi"]["node_limit"] is None


@pytest.mark.parametrize("ancilla", ["Z", "Y"])
def test_real_weight_one_ancilla_stabilizer_excluded(ancilla):
    x, z = checks(["XXXXI", "ZZZZI", "IIII" + ancilla])
    result = real_solve(x, z, seed(x, z))
    assert result["distance"] == 2
    assert result["gurobi"]["logical_basis_rank"] == 4
    verify(x, z, result)


def test_real_complete_basis_finds_free_auxiliary_logical():
    x, z = checks(["XXXXI", "ZZZZI"])
    result = real_solve(x, z, weakened_seed(x, z, "XXIII"))
    assert result["distance"] == 1
    assert result["witness"]["pauli"][:4] == "IIII"
    assert result["gurobi"]["logical_basis_rank"] == 6
    verify(x, z, result)


def test_real_threshold_only_certifies_requested_range():
    x, z = css_checks(steane_code())
    result = real_solve(x, z, seed(x, z, max_weight=1))
    assert result["status"] == "bounded" and result["distance"] is None
    assert result["proven_lower_bound"] == 2 and result["upper_bound"] == 3
    verify(x, z, result)


def test_model_identical_to_existing_scipy_builder(monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    initial = weakened_seed(x, z, "XXXXX")
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status=1, x=None, message="captured without solving")

    monkeypatch.setattr(scipy, "milp", capture)
    milp_distance(x, z, initial)
    actual = backend._model_data(x, z, cutoff=4)
    dense = np.zeros(captured["constraints"].A.shape)
    for i, row in enumerate(actual["rows"]):
        for column, coefficient in row:
            dense[i, column] += coefficient
    np.testing.assert_array_equal(dense, captured["constraints"].A.toarray())
    for ours, theirs in [("lower", captured["bounds"].lb), ("upper", captured["bounds"].ub),
                         ("objective", captured["c"]), ("row_lower", captured["constraints"].lb),
                         ("row_upper", captured["constraints"].ub)]:
        np.testing.assert_array_equal(actual[ours], theirs)


@pytest.fixture
def distance_ten_input():
    # Generalized Shor [[100,1,10]]: genuine weight-ten upper witnesses, no
    # expensive solve. Deterministic mocked outcomes isolate bound semantics.
    side, n = 10, 100
    xrows, zrows = [], []
    for block in range(side):
        for j in range(side - 1):
            zrow = np.zeros(n, np.uint8)
            zrow[block * side + j:block * side + j + 2] = 1
            xrows.append(np.zeros(n, np.uint8)); zrows.append(zrow)
    for block in range(side - 1):
        xrow = np.zeros(n, np.uint8)
        xrow[block * side:(block + 2) * side] = 1
        xrows.append(xrow); zrows.append(np.zeros(n, np.uint8))
    x, z = np.asarray(xrows), np.asarray(zrows)
    initial = seed(x, z)
    assert initial["upper_bound"] == 10 and initial["k"] == 1
    return x, z, initial


@pytest.mark.parametrize("status", [8, 9])
@pytest.mark.parametrize("bound,expected", [(9.0, 9), (8.000000001, 8), (8.00001, 9),
                                           (10.0, 10), (0.0, 1), (-2.0, 1)])
def test_limit_uses_conservative_integer_bound(monkeypatch, distance_ten_input, status, bound, expected):
    x, z, initial = distance_ten_input
    monkeypatch.setattr(backend, "_solve_gurobi", lambda *a, **kw:
                        dict(status=status, x=None, solution_count=0, ObjBoundC=bound))
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == expected
    assert result["upper_bound"] == 10
    assert result["distance"] == (10 if expected == 10 else None)
    assert result["gurobi"]["bound_source"] == "ObjBoundC"
    assert result["stop_reason"] == ("gurobi_node_limit" if status == 8 else "gurobi_time_limit")
    verify(x, z, result)


def test_bound_is_capped_at_searched_weight_range(monkeypatch, distance_ten_input):
    x, z, initial = distance_ten_input
    initial["max_weight"] = 7
    monkeypatch.setattr(backend, "_solve_gurobi", lambda *a, **kw:
                        dict(status=9, x=None, solution_count=0, ObjBoundC=20.0))
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == 8 and result["distance"] is None


def test_objbound_fallback(monkeypatch, distance_ten_input):
    x, z, initial = distance_ten_input
    monkeypatch.setattr(backend, "_solve_gurobi", lambda *a, **kw:
                        dict(status=9, x=None, solution_count=0, ObjBound=9.0))
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == 9 and result["gurobi"]["bound_source"] == "ObjBound"


@pytest.mark.parametrize("bound", [np.inf, -np.inf, np.nan, 1e100, -1e100])
def test_nonfinite_and_gurobi_infinity_do_not_certify(monkeypatch, distance_ten_input, bound):
    x, z, initial = distance_ten_input
    monkeypatch.setattr(backend, "_solve_gurobi", lambda *a, **kw:
                        dict(status=9, x=None, solution_count=0, ObjBoundC=bound))
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == 1 and result["distance"] is None
    assert result["gurobi"]["proof"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("status", [4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, None])
def test_ambiguous_numeric_or_other_status_never_promotes_bound(monkeypatch, status):
    x, z = css_checks(steane_code())
    initial = seed(x, z)
    monkeypatch.setattr(backend, "_solve_gurobi", lambda *a, **kw:
                        dict(status=status, x=None, solution_count=0, ObjBoundC=3.0))
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == 1 and result["distance"] is None
    assert result["gurobi"]["proof"] is None


@pytest.mark.parametrize("mode", ["zero", "fractional", "nan", "wrong_shape"])
def test_invalid_optimal_solution_cannot_certify(monkeypatch, mode):
    x, z = css_checks(steane_code())

    def invalid(data, **kwargs):
        vector = np.zeros_like(data["objective"])
        if mode == "fractional":
            vector[:] = .5
        elif mode == "nan":
            vector[:] = np.nan
        elif mode == "wrong_shape":
            vector = vector[:-1]
        return dict(status=2, x=vector, objective=0, solution_count=1, ObjBoundC=3)

    monkeypatch.setattr(backend, "_solve_gurobi", invalid)
    result = backend.gurobi_distance(x, z, seed(x, z))
    assert result["proven_lower_bound"] == 1 and result["distance"] is None
    assert result["stop_reason"] == "gurobi_invalid_solution"


def test_timeout_retains_validated_incumbent(monkeypatch):
    pytest.importorskip("gurobipy")
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    actual = backend._solve_gurobi

    def interrupted(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.update(status=9, ObjBoundC=2.0)
        return result

    monkeypatch.setattr(backend, "_solve_gurobi", interrupted)
    result = real_solve(x, z, weakened_seed(x, z, "XXXXX"))
    assert result["upper_bound"] == 3 and result["proven_lower_bound"] == 2
    assert result["distance"] is None
    verify(x, z, result)


def test_contradictory_bound_rejected_with_valid_incumbent(monkeypatch):
    pytest.importorskip("gurobipy")
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    actual = backend._solve_gurobi

    def contradictory(*args, **kwargs):
        result = actual(*args, **kwargs)
        result.update(status=9, ObjBoundC=99.0)
        return result

    monkeypatch.setattr(backend, "_solve_gurobi", contradictory)
    result = real_solve(x, z, weakened_seed(x, z, "XXXXX"))
    assert result["upper_bound"] == 3 and result["proven_lower_bound"] == 1
    assert result["distance"] is None and result["gurobi"]["proof"] is None


def test_missing_solver_is_explicitly_unavailable(monkeypatch):
    x, z = css_checks(steane_code())
    monkeypatch.setitem(sys.modules, "gurobipy", None)
    with pytest.raises(backend.GurobiUnavailableError, match="gurobipy"):
        backend.gurobi_distance(x, z, seed(x, z))


@pytest.mark.parametrize("errno", [10009, 10010])
def test_license_failures_are_explicitly_unavailable(monkeypatch, errno):
    class FakeError(Exception):
        def __init__(self):
            self.errno = errno

    class FakeEnv:
        def __init__(self, **kwargs): pass
        def setParam(self, *args): pass
        def start(self): raise FakeError()
        def dispose(self): pass

    monkeypatch.setitem(sys.modules, "gurobipy", SimpleNamespace(Env=FakeEnv, GurobiError=FakeError))
    x, z = css_checks(steane_code())
    with pytest.raises(backend.GurobiUnavailableError, match="license"):
        backend.gurobi_distance(x, z, seed(x, z))


def test_unexpected_solver_failure_preserves_existing_bounds(monkeypatch):
    x, z = css_checks(steane_code())

    def broken(*args, **kwargs):
        raise RuntimeError("mock numerical failure")

    monkeypatch.setattr(backend, "_solve_gurobi", broken)
    initial = seed(x, z)
    initial["proven_lower_bound"] = 2
    result = backend.gurobi_distance(x, z, initial)
    assert result["proven_lower_bound"] == 2 and result["upper_bound"] == 3
    assert result["distance"] is None and result["stop_reason"] == "gurobi_solver_error"


@pytest.mark.parametrize("kwargs", [dict(time_limit=-1), dict(time_limit=np.inf),
                                   dict(time_limit=True), dict(node_limit=-1),
                                   dict(node_limit=True), dict(node_limit=.5)])
def test_invalid_resource_limits(kwargs):
    x, z = css_checks(steane_code())
    with pytest.raises(ValueError):
        backend.gurobi_distance(x, z, seed(x, z), **kwargs)
