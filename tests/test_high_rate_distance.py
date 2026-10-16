"""Protected subsystem pairing and independent finite distance certificates."""
from itertools import product
import json

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.high_rate_surgery import distance as hd


def fixture_with_gauge(*, eliminated=False):
    code = four_qubit_code()
    basis = code.standard_logicals()
    b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    r = np.eye(4, dtype=np.uint8)[:, :2]
    mx = np.vstack((np.pad(code.hx, ((0, 0), (0, 2))), np.hstack((r.T, b.T))))
    if eliminated:
        hz = np.vstack((code.hz, code.hz))
        mz = np.hstack((hz, np.array([[1, 0], [1, 1]], dtype=np.uint8)))
    else:
        hz = code.hz
        mz = np.hstack((hz, np.array([[1, 1]], dtype=np.uint8)))
    targets = (nullspace(b) @ r.T @ basis.jz.T) & 1
    args = (code.hx, hz, basis.jx, basis.jz, mx, mz, b, r, targets)
    return args


def exhaustive(h, detector, gauge, stabilizers):
    minimum = h.shape[1] + 1
    for bits in product((0, 1), repeat=h.shape[1]):
        v = np.asarray(bits, dtype=np.uint8)
        if np.any((h @ v) & 1):
            continue
        detected = bool(np.any((detector @ v) & 1))
        assert detected == (not in_rowspace(v, np.vstack((stabilizers, gauge))))
        if detected:
            minimum = min(minimum, int(v.sum()))
    return minimum


def test_gauge_pairing_and_exhaustive_quotient_distance():
    args = fixture_with_gauge()
    info = hd.build_css_protected_basis(*args)
    mx, mz = args[4:6]
    gx, gz, lx, lz = (info[k] for k in ("gauge_x", "gauge_z", "bare_x", "bare_z"))
    assert info["protected_logicals"] == 1
    assert info["gauge_qubits"] == 1
    assert info["merged_logicals"] == 2
    np.testing.assert_array_equal((gx @ gz.T) & 1, [[1]])
    np.testing.assert_array_equal((lx @ lz.T) & 1, [[1]])
    assert not np.any((gx[:, :4] @ args[3].T) & 1)
    assert not np.any((gx @ lz.T) & 1)
    dx = exhaustive(mz, lz, gx, mx)
    dz = exhaustive(mx, lx, gz, mz)
    report = hd.css_dressed_distance(mx, mz, lx, lz, method="enum", trials=0)
    assert report["x"]["distance"] == dx
    assert report["z"]["distance"] == dz
    assert report["distance"] == min(dx, dz) == 2
    assert not report["measurement_distance_certified"]


def test_data_metacheck_eliminates_ancilla_gauge():
    args = fixture_with_gauge(eliminated=True)
    info = hd.build_css_protected_basis(*args)
    assert info["gauge_qubits"] == 0
    assert info["protected_logicals"] == info["merged_logicals"] == 1
    assert info["gauge_x"].shape == info["gauge_z"].shape == (0, 6)


def test_free_ancilla_weight_one_does_not_count_as_protected_logical():
    code = four_qubit_code()
    logical = code.standard_logicals()
    mx, mz = (np.pad(h, ((0, 0), (0, 1))) for h in (code.hx, code.hz))
    info = hd.build_css_protected_basis(code.hx, code.hz, logical.jx, logical.jz,
                                       mx, mz, np.zeros((1, 0), np.uint8),
                                       np.zeros((4, 0), np.uint8), np.zeros((0, 2), np.uint8))
    assert info["gauge_qubits"] == 1
    report = hd.css_dressed_distance(mx, mz, info["bare_x"], info["bare_z"],
                                    method="enum", trials=0)
    assert report["distance"] == 2
    assert all(v[-1] == 0 for v in report["low_weight_x_witnesses"])


def test_all_logicals_measured_is_not_a_measurement_distance_certificate():
    hx = hz = np.zeros((0, 2), np.uint8)
    eye = np.eye(2, dtype=np.uint8)
    info = hd.build_css_protected_basis(hx, hz, eye, eye, eye, hz,
                                       np.zeros((0, 2), np.uint8), eye, eye)
    assert info["protected_logicals"] == 0
    report = hd.css_dressed_distance(eye, hz, info["bare_x"], info["bare_z"])
    assert report["status"] == "no_protected_logicals"
    assert report["lower_bound"] is report["upper_bound"] is report["distance"] is None
    assert report["measurement_distance_certified"] is False


@pytest.mark.parametrize("tamper", ["target", "R", "check", "source_basis", "projection"])
def test_reject_wrong_measurement_or_inconsistent_identification(tamper):
    args = list(fixture_with_gauge())
    if tamper == "target":
        args[8] = np.array([[1, 0]], np.uint8)
    elif tamper == "R":
        args[7] = args[7].copy()
        args[7][2, 0] ^= 1
    elif tamper == "check":
        args[4] = args[4].copy()
        args[4][0, 0] ^= 1
    elif tamper == "source_basis":
        args[2] = np.zeros_like(args[2])
    else:
        args[5] = np.zeros((0, 6), np.uint8)
    with pytest.raises(ValueError):
        hd.build_css_protected_basis(*args)


def test_source_distance_enumeration_steane():
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="enum", trials=0)
    assert report["distance"] == 3
    assert report["x"]["witness_validated"] and report["z"]["witness_validated"]


def test_candidate_limit_does_not_certify_unfinished_weight_shell():
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="enum", trials=0, max_candidates=3)
    assert report["lower_bound"] == 1
    assert report["enumerated_candidates"] == 3
    assert report["distance"] is None
    assert report["x"]["stop_reason"] == report["z"]["stop_reason"] == "candidate_limit"


def test_completed_weight_shell_certifies_bound_with_cutoff():
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="enum", trials=0, max_weight=1)
    assert report["lower_bound"] == 2
    assert report["upper_bound"] >= 3
    assert report["distance"] is None
    assert report["enumerated_candidates"] == 14
    assert report["low_weight_x_witnesses"] == []


def test_heuristic_gives_only_upper_bound_and_is_reproducible():
    code = steane_code()
    basis = code.standard_logicals()
    args = (code.hx, code.hz, basis.jx, basis.jz)
    a = hd.css_dressed_distance(*args, method="heuristic", trials=5, seed=4)
    b = hd.css_dressed_distance(*args, method="heuristic", trials=5, seed=4)
    assert a["lower_bound"] == 1 and a["distance"] is None
    assert a["upper_bound"] == b["upper_bound"] == 3
    assert a["low_weight_x_witnesses"] == b["low_weight_x_witnesses"]
    assert a["x"]["heuristic_trials"] == 5


def test_auto_falls_back_when_gurobi_unavailable(monkeypatch):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {"status": "unavailable"})
    code = four_qubit_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="auto", trials=0)
    assert report["distance"] == 2
    assert report["x"]["solver"]["status"] == "unavailable"
    assert report["x"]["method"] == "enumeration"


def test_explicit_missing_gurobi_reports_unknown_bounds_without_raising(monkeypatch):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {"status": "unavailable"})
    code = four_qubit_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0)
    assert report["lower_bound"] == 1
    assert report["upper_bound"] == 2
    assert report["distance"] is None


def test_gurobi_timeout_uses_valid_objective_lower_bound(monkeypatch):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {
        "status": 9, "objective_bound": 2.0, "candidate": None, "solution_count": 0})
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0)
    assert report["lower_bound"] == 2
    assert report["x"]["lower_bound_certificate"].startswith("gurobi_objective_bound")


@pytest.mark.parametrize("status,bound", [(12, 10), (4, 10), (9, float("nan")), (9, 100)])
def test_ambiguous_or_contradictory_gurobi_bound_rejected(monkeypatch, status, bound):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {
        "status": status, "objective_bound": bound, "candidate": None, "solution_count": 0})
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0)
    assert report["lower_bound"] == 1
    assert report["distance"] is None


def test_invalid_gurobi_witness_never_creates_exact_answer(monkeypatch):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {
        "status": 2, "objective": 1, "candidate": [1, 0, 0, 0], "solution_count": 1})
    code = four_qubit_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0)
    assert report["distance"] is None
    assert report["upper_bound"] == 2


def test_infeasible_solver_report_remains_strict_json(monkeypatch):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **k: {
        "status": 3, "candidate": None, "solution_count": 0,
        "objective_bound": float("inf")})
    code = four_qubit_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0)
    assert report["distance"] == 2
    assert report["x"]["solver"]["objective_bound"] is None
    json.dumps(report, allow_nan=False)


def test_gurobi_directions_share_a_single_deadline(monkeypatch):
    clock = [100.0]
    deadlines = []
    monkeypatch.setattr(hd, "perf_counter", lambda: clock[0])
    def fake_solve(h, detector, cutoff, deadline, initial, seed):
        deadlines.append(deadline)
        clock[0] = deadline
        return {"status": 9, "candidate": None}
    monkeypatch.setattr(hd, "_solve_gurobi", fake_solve)
    code = steane_code()
    basis = code.standard_logicals()
    report = hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz,
                                    method="gurobi", trials=0, time_limit=60)
    assert deadlines == [130.0, 160.0]
    assert report["runtime_seconds"] == 60.0


@pytest.mark.parametrize("kwargs", [{"time_limit": 0}, {"time_limit": None},
                                     {"time_limit": True}, {"max_candidates": -1},
                                     {"trials": 1.5}, {"max_weight": -1},
                                     {"method": "not_a_method"}, {"seed": -1}])
def test_invalid_search_controls(kwargs):
    code = four_qubit_code()
    basis = code.standard_logicals()
    with pytest.raises(ValueError):
        hd.css_dressed_distance(code.hx, code.hz, basis.jx, basis.jz, **kwargs)


def test_noncanonical_bare_pairing_rejected():
    code = four_qubit_code()
    basis = code.standard_logicals()
    with pytest.raises(ValueError):
        hd.css_dressed_distance(code.hx, code.hz, basis.jx, np.zeros_like(basis.jz))
