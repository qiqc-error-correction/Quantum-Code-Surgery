"""Optional HiGHS MILP backend for the static stabilizer-code distance.

The objective is the Pauli weight, ``sum(x OR z)``.  A full basis of the
centralizer modulo the stabilizer is computed from the supplied checks;
requiring a nonzero symplectic signature against that basis excludes every
stabilizer, including low-weight and auxiliary-qubit stabilizers.

Optimal/infeasible HiGHS statuses supply the lower-bound conclusion.  Every
returned upper-bound witness is independently checked using integer GF(2)
arithmetic.  Floating-point dual estimates on an interrupted run are only
diagnostics, never promoted to a distance certificate.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np


def _limits(time_limit, node_limit):
    if time_limit is not None:
        if (isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, Real)
                or not np.isfinite(time_limit) or time_limit < 0):
            raise ValueError("milp time_limit must be finite and >= 0, or None")
        time_limit = float(time_limit)
    if node_limit is not None:
        if (isinstance(node_limit, (bool, np.bool_))
                or not isinstance(node_limit, (int, np.integer)) or node_limit < 0):
            raise ValueError("milp node_limit must be an integer >= 0, or None")
        node_limit = int(node_limit)
    return time_limit, node_limit


def _logical_basis(checks, n):
    """Return independent checks and a full (not necessarily paired) quotient basis."""
    from .distance import _basis, _kernel_rows

    stabilizers = _basis(checks)
    span = dict(stabilizers)
    mask = (1 << n) - 1
    commutation = [(value >> n) | ((value & mask) << n) for value in stabilizers.values()]
    logicals = []
    for original in _kernel_rows(commutation, 2 * n):
        value = original
        while value:
            pivot = value.bit_length() - 1
            if pivot in span:
                value ^= span[pivot]
            else:
                span[pivot] = value
                logicals.append(original)
                break
    if len(logicals) != 2 * (n - len(stabilizers)):
        raise RuntimeError("Could not compute the complete stabilizer logical quotient")
    return stabilizers, logicals


def milp_distance(check_x, check_z, initial, *, time_limit=30.0, node_limit=None):
    """Improve a validated distance report using optional SciPy/HiGHS MILP.

    ``check_x`` and ``check_z`` must already be equally shaped normalized,
    commuting binary matrices.  ``initial`` is a report from the distance
    enumerator, optionally improved by a validated random search.  Searching
    only weights up to ``min(max_weight, upper_bound - 1)`` permits threshold
    certification without requiring an exact distance.  An interrupted or
    failed solve preserves all previously established bounds.
    """
    from .distance import _bits, _contains, _witness

    time_limit, node_limit = _limits(time_limit, node_limit)
    report = deepcopy(initial)
    report["method"] = "symplectic_milp"
    info = report["milp"] = {
        "solver": "scipy.optimize.milp/HiGHS", "time_limit": time_limit,
        "node_limit": node_limit, "status": None, "message": None,
        "cutoff": None, "witness_validated": False, "proof": None,
        "logical_basis_rank": None,
        "lower_bound_policy": "optimal_or_infeasible_status_only",
    }
    if report["status"] in {"exact", "no_logicals"}:
        info["message"] = "Initial report already resolves the distance"
        return report
    n = int(check_x.shape[1])
    cutoff = min(int(report["max_weight"]), int(report["upper_bound"]) - 1)
    info["cutoff"] = cutoff
    if cutoff < int(report["proven_lower_bound"]):
        report.update(status="bounded", stop_reason="milp_weight_limit")
        info["message"] = "Requested weight range is already excluded"
        return report

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise ImportError("MILP distance needs scipy>=1.9; install codesurgery[distance] "
                          "or select an available distance backend") from error

    checks = [_bits(x) | (_bits(z) << n) for x, z in zip(check_x, check_z)]
    stabilizers, logicals = _logical_basis(checks, n)
    independent = list(stabilizers.values())
    r, ell = len(independent), len(logicals)
    info["logical_basis_rank"] = ell

    # Variables: x[n], z[n], t[n], signature[2k], check_parity[r], logical_parity[2k].
    sig_start, parity_start = 3 * n, 3 * n + ell
    logical_parity_start = parity_start + r
    size = logical_parity_start + ell
    lower, upper = np.zeros(size), np.ones(size)
    objective = np.zeros(size)
    objective[2 * n:3 * n] = 1
    rows, cols, data, low, high = [], [], [], [], []

    def constraint(entries, lb, ub):
        index = len(low)
        for column, coefficient in entries:
            rows.append(index)
            cols.append(column)
            data.append(coefficient)
        low.append(lb)
        high.append(ub)

    def symplectic_entries(value):
        for q in range(n):
            if (value >> (n + q)) & 1:
                yield q, 1
            if (value >> q) & 1:
                yield n + q, 1

    for j, value in enumerate(independent):
        terms = list(symplectic_entries(value))
        upper[parity_start + j] = len(terms) // 2
        constraint(terms + [(parity_start + j, -2)], 0, 0)
    for j, value in enumerate(logicals):
        terms = list(symplectic_entries(value))
        upper[logical_parity_start + j] = len(terms) // 2
        constraint(terms + [(sig_start + j, -1), (logical_parity_start + j, -2)], 0, 0)
    constraint([(sig_start + j, 1) for j in range(ell)], 1, np.inf)
    for q in range(n):
        constraint([(2 * n + q, 1), (q, -1)], 0, np.inf)
        constraint([(2 * n + q, 1), (n + q, -1)], 0, np.inf)
        constraint([(2 * n + q, 1), (q, -1), (n + q, -1)], -np.inf, 0)
    constraint([(2 * n + q, 1) for q in range(n)], -np.inf, cutoff)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(low), size), dtype=float).tocsc()
    low, high = np.asarray(low), np.asarray(high)
    options = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if node_limit is not None:
        options["node_limit"] = node_limit
    try:
        result = milp(c=objective, integrality=np.ones(size, dtype=np.uint8),
                      bounds=Bounds(lower, upper), constraints=LinearConstraint(matrix, low, high),
                      options=options)
    except Exception as error:
        info.update(message=f"{type(error).__name__}: {error}", proof=None)
        report.update(status="bounded" if report["proven_lower_bound"] > 1 else "unknown",
                      distance=None, stop_reason="milp_solver_error")
        return report

    status = getattr(result, "status", None)
    info["status"] = int(status) if isinstance(status, (int, np.integer)) else None
    info["message"] = str(getattr(result, "message", ""))
    for attr in ("mip_node_count", "mip_dual_bound", "mip_gap"):
        value = getattr(result, attr, None)
        if isinstance(value, Real) and np.isfinite(value):
            info[attr] = int(value) if attr == "mip_node_count" else float(value)

    # Never trust an incumbent's objective or support without checking its Pauli.
    candidate = getattr(result, "x", None)
    value = None
    model_valid = False
    if candidate is not None:
        try:
            candidate = np.asarray(candidate, dtype=float)
            if candidate.shape == (size,) and np.all(np.isfinite(candidate)):
                rounded = np.rint(candidate)
                if np.all(np.abs(candidate - rounded) <= 1e-6):
                    action = matrix @ rounded
                    model_valid = bool(np.all(rounded >= lower) and np.all(rounded <= upper)
                                       and np.all(action >= low) and np.all(action <= high))
                    if model_valid:
                        xb, zb = _bits(rounded[:n]), _bits(rounded[n:2 * n])
                        proposed = xb | (zb << n)
                        commutes = all(((xb & (check >> n)) ^ (zb & check)).bit_count() % 2 == 0
                                       for check in checks)
                        if commutes and not _contains(proposed, stabilizers):
                            witness = _witness(proposed, n)
                            if witness["weight"] <= cutoff:
                                value = proposed
                                info["witness_validated"] = True
                                if witness["weight"] < report["upper_bound"]:
                                    report.update(upper_bound=witness["weight"], witness=witness)
        except (TypeError, ValueError, OverflowError):
            model_valid = False

    if status == 2 and candidate is None:
        # All nontrivial logicals through the cutoff were excluded.
        report["proven_lower_bound"] = max(report["proven_lower_bound"], cutoff + 1)
        info["proof"] = "milp_infeasible_cutoff"
        report["stop_reason"] = "milp_weight_limit"
    elif status == 0 and value is not None:
        weight = _witness(value, n)["weight"]
        fun = getattr(result, "fun", None)
        if isinstance(fun, Real) and np.isfinite(fun) and abs(float(fun) - weight) <= 1e-6:
            report["proven_lower_bound"] = max(report["proven_lower_bound"], weight)
            info["proof"] = "milp_optimal"
            report["stop_reason"] = "milp_optimal"
        else:
            report["stop_reason"] = "milp_invalid_solution"
    elif status == 1:
        report["stop_reason"] = "milp_time_or_node_limit"
    elif status in (0, 2):
        report["stop_reason"] = "milp_invalid_solution"
    else:
        report["stop_reason"] = "milp_solver_error"

    if report["proven_lower_bound"] == report["upper_bound"]:
        report.update(status="exact", distance=report["upper_bound"])
    else:
        report.update(status="bounded" if report["proven_lower_bound"] > 1 else "unknown",
                      distance=None)
    return report


__all__ = ["milp_distance"]
