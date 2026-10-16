"""Optional Gurobi backend for static stabilizer distance and certified bounds.

The model searches every nontrivial logical Pauli up to the requested cutoff.
An incumbent is accepted only after exact GF(2) and integer-model checks.
On a time/node limit, a finite Gurobi MIP objective bound can still establish
an integer distance lower bound. This is a solver certificate under Gurobi's
numerical tolerances, not an independently checked exact-arithmetic proof.
"""
from __future__ import annotations

from copy import deepcopy
from math import ceil
from numbers import Real

import numpy as np

from ._distance_milp import _limits, _logical_basis


class GurobiUnavailableError(ImportError):
    """The optional solver is missing or its license cannot solve this model."""


_STATUS_NAMES = {
    2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
    6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    14: "INPROGRESS", 15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}
_BOUND_TOLERANCE = 1e-6


def _model_data(check_x, check_z, cutoff):
    """Construct the same integer model as the SciPy/HiGHS backend."""
    from .distance import _bits

    n = check_x.shape[1]
    checks = [_bits(x) | (_bits(z) << n) for x, z in zip(check_x, check_z)]
    stabilizers, logicals = _logical_basis(checks, n)
    independent = list(stabilizers.values())
    r, ell = len(independent), len(logicals)
    sig_start, parity_start = 3 * n, 3 * n + ell
    logical_parity_start = parity_start + r
    size = logical_parity_start + ell
    lower, upper, objective = np.zeros(size), np.ones(size), np.zeros(size)
    objective[2 * n:3 * n] = 1
    rows, lows, highs = [], [], []

    def constraint(entries, lo, hi):
        rows.append(list(entries))
        lows.append(lo)
        highs.append(hi)

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
    return dict(n=n, checks=checks, stabilizers=stabilizers, logical_basis_rank=ell,
                lower=lower, upper=upper, objective=objective, rows=rows,
                row_lower=np.asarray(lows), row_upper=np.asarray(highs))


def _solve_gurobi(data, *, time_limit, node_limit):
    """Keep solver imports and license ownership local to a single solve."""
    try:
        import gurobipy as gp
    except (ImportError, OSError) as error:
        raise GurobiUnavailableError(
            "Gurobi distance requires gurobipy and a usable Gurobi license; "
            "install codesurgery[gurobi] or select method='milp' for HiGHS") from error

    env = model = None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("codesurgery_static_distance", env=env)
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.Params.TimeLimit = gp.GRB.INFINITY if time_limit is None else time_limit
        model.Params.NodeLimit = gp.GRB.INFINITY if node_limit is None else node_limit
        variables = [model.addVar(lb=float(lo), ub=float(hi), vtype=gp.GRB.INTEGER,
                                  name=f"v{i}")
                     for i, (lo, hi) in enumerate(zip(data["lower"], data["upper"]))]
        for row, lo, hi in zip(data["rows"], data["row_lower"], data["row_upper"]):
            expr = gp.LinExpr([value for _, value in row], [variables[j] for j, _ in row])
            if lo == hi:
                model.addConstr(expr == float(lo))
            else:
                if np.isfinite(lo):
                    model.addConstr(expr >= float(lo))
                if np.isfinite(hi):
                    model.addConstr(expr <= float(hi))
        model.setObjective(gp.LinExpr(data["objective"].tolist(), variables), gp.GRB.MINIMIZE)
        model.optimize()
        result = dict(status=int(model.Status), solution_count=int(model.SolCount),
                      runtime_seconds=float(model.Runtime), node_count=float(model.NodeCount),
                      x=None, objective=None, version=list(gp.gurobi.version()))
        if model.SolCount > 0:
            result.update(x=np.asarray(model.getAttr("X", variables)), objective=float(model.ObjVal))
        for name in ("ObjBoundC", "ObjBound"):
            try:
                result[name] = float(getattr(model, name))
            except (AttributeError, gp.GurobiError):
                pass
        return result
    except gp.GurobiError as error:
        if error.errno in (10009, 10010):  # NO_LICENSE / SIZE_LIMIT_EXCEEDED
            raise GurobiUnavailableError(
                f"Gurobi cannot solve this model with the current license (error {error.errno}); "
                "configure a suitable license or select method='milp' for HiGHS") from error
        raise
    finally:
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()


def _finite_number(value):
    # Gurobi also uses 1e100 as its infinity sentinel in some attributes.
    return (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and abs(value) < 1e100)


def _validated_candidate(data, candidate, cutoff):
    from .distance import _bits, _contains, _witness

    if candidate is None:
        return None
    try:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != data["lower"].shape or not np.all(np.isfinite(candidate)):
            return None
        rounded = np.rint(candidate)
        if (not np.all(np.abs(candidate - rounded) <= _BOUND_TOLERANCE)
                or np.any(rounded < data["lower"]) or np.any(rounded > data["upper"])):
            return None
        # The model coefficients and rounded variable values are all integers.
        values = [int(value) for value in rounded]
        for row, lo, hi in zip(data["rows"], data["row_lower"], data["row_upper"]):
            activity = sum(values[j] * coefficient for j, coefficient in row)
            if not lo <= activity <= hi:
                return None
        n = data["n"]
        xb, zb = _bits(values[:n]), _bits(values[n:2 * n])
        if any(((xb & (check >> n)) ^ (zb & check)).bit_count() % 2 for check in data["checks"]):
            return None
        value = xb | (zb << n)
        if _contains(value, data["stabilizers"]):
            return None
        witness = _witness(value, n)
        return witness if witness["weight"] <= cutoff else None
    except (TypeError, ValueError, OverflowError):
        return None


def gurobi_distance(check_x, check_z, initial, *, time_limit=60.0, node_limit=None):
    """Improve a validated distance report with Gurobi, defaulting to 60 seconds.

    For TIME_LIMIT/NODE_LIMIT only, a finite ObjBoundC (or ObjBound fallback)
    establishes ``ceil(bound - 1e-6)``. Capping at cutoff+1 accounts for the
    model's weight restriction; capping at the validated upper bound cannot
    strengthen this conclusion. Ambiguous, numeric, or failed solves never
    supply a lower-bound certificate. The input report is not mutated.
    """
    time_limit, node_limit = _limits(time_limit, node_limit)
    report = deepcopy(initial)
    report["method"] = "symplectic_gurobi"
    info = report["gurobi"] = dict(
        solver="gurobipy/Gurobi", time_limit=time_limit, node_limit=node_limit,
        status=None, status_name=None, cutoff=None, witness_validated=False,
        proof=None, logical_basis_rank=None, bound_source=None,
        bound_rounding_tolerance=_BOUND_TOLERANCE,
        lower_bound_policy="completed_status_or_finite_time_node_limit_mip_bound",
        numerical_certificate="Gurobi floating-point MIP certificate; witnesses checked in exact GF(2)")
    if report["status"] in {"exact", "no_logicals"}:
        info["message"] = "Initial report already resolves the distance"
        return report
    cutoff = min(int(report["max_weight"]), int(report["upper_bound"]) - 1)
    info["cutoff"] = cutoff
    if cutoff < int(report["proven_lower_bound"]):
        report.update(status="bounded", stop_reason="gurobi_weight_limit")
        info["message"] = "Requested weight range is already excluded"
        return report
    data = _model_data(check_x, check_z, cutoff)
    info["logical_basis_rank"] = data["logical_basis_rank"]
    try:
        result = _solve_gurobi(data, time_limit=time_limit, node_limit=node_limit)
    except GurobiUnavailableError:
        raise
    except Exception as error:
        info["message"] = f"{type(error).__name__}: {error}"
        report.update(status="bounded" if report["proven_lower_bound"] > 1 else "unknown",
                      distance=None, stop_reason="gurobi_solver_error")
        return report

    status = result.get("status")
    info["status"] = int(status) if isinstance(status, (int, np.integer)) else None
    info["status_name"] = _STATUS_NAMES.get(status, "UNKNOWN")
    for name in ("runtime_seconds", "node_count", "ObjBoundC", "ObjBound"):
        if _finite_number(result.get(name)):
            info[name] = float(result[name])
    if "version" in result:
        info["version"] = result["version"]
    witness = _validated_candidate(data, result.get("x"), cutoff)
    if witness is not None:
        info["witness_validated"] = True
        if witness["weight"] < report["upper_bound"]:
            report.update(upper_bound=witness["weight"], witness=witness)

    if status == 3 and result.get("solution_count", 0) == 0 and result.get("x") is None:
        report["proven_lower_bound"] = max(report["proven_lower_bound"], cutoff + 1)
        info["proof"] = "gurobi_infeasible_cutoff"
        report["stop_reason"] = "gurobi_weight_limit"
    elif status == 2:
        objective = result.get("objective")
        if (witness is not None and _finite_number(objective)
                and abs(objective - witness["weight"]) <= _BOUND_TOLERANCE):
            report["proven_lower_bound"] = max(report["proven_lower_bound"], witness["weight"])
            info["proof"] = "gurobi_optimal"
            report["stop_reason"] = "gurobi_optimal"
        else:
            report["stop_reason"] = "gurobi_invalid_solution"
    elif status in (8, 9):
        report["stop_reason"] = "gurobi_node_limit" if status == 8 else "gurobi_time_limit"
        for name in ("ObjBoundC", "ObjBound"):
            bound = result.get(name)
            if not _finite_number(bound):
                continue
            # A contradictory solver bound must not be hidden by clipping.
            if witness is not None and bound > witness["weight"] + _BOUND_TOLERANCE:
                info["bound_rejected"] = "objective bound exceeds independently verified incumbent"
                break
            lower = min(ceil(float(bound) - _BOUND_TOLERANCE), cutoff + 1, int(report["upper_bound"]))
            report["proven_lower_bound"] = max(report["proven_lower_bound"], lower)
            info.update(bound_source=name, proof="gurobi_mip_objective_bound",
                        integer_lower_bound=lower)
            break
    else:
        report["stop_reason"] = "gurobi_solver_error"

    if report["proven_lower_bound"] == report["upper_bound"]:
        report.update(status="exact", distance=report["upper_bound"])
    else:
        report.update(status="bounded" if report["proven_lower_bound"] > 1 else "unknown", distance=None)
    return report


__all__ = ["GurobiUnavailableError", "gurobi_distance"]
