"""Resource-bounded distance of a binary stabilizer code.

The searched distance is ``min wt(x | z)`` over the symplectic centralizer
outside the stabilizer row space.  Generator signs do not affect this value
for a valid stabilizer code.  This is a static stabilizer distance, not a
circuit distance or a subsystem's dressed/protected distance.

Enumeration and cluster searches visit complete weight shells in ascending
order. Stopping halfway through a shell never certifies that shell. The CSS shortcut is exact:
every mixed logical has a nontrivial pure X or Z component of no larger
weight.  General mixed-Pauli checks use all three Paulis at each support site.
"""
from __future__ import annotations

from itertools import combinations, product

import numpy as np

from ..gf2 import binary_matrix


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bits(row):
    value = 0
    for q in np.flatnonzero(row):
        value |= 1 << int(q)
    return value


def _basis(rows):
    """Echelon basis indexed by highest set bit, for integer row reduction."""
    pivots = {}
    for value in rows:
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return pivots


def _contains(value, pivots):
    while value:
        row = pivots.get(value.bit_length() - 1)
        if row is None:
            return False
        value ^= row
    return True


def _kernel_rows(rows, width):
    pivots = _basis(rows)
    ordered = sorted(pivots.items())
    for column in range(width):
        if column in pivots:
            continue
        vector = 1 << column
        for pivot, row in ordered:
            if (row & vector).bit_count() & 1:
                vector |= 1 << pivot
        yield vector


def _witness(value, n):
    mask = (1 << n) - 1
    x, z = value & mask, value >> n
    return {
        "x": [(x >> q) & 1 for q in range(n)],
        "z": [(z >> q) & 1 for q in range(n)],
        "pauli": "".join("IXZY"[((x >> q) & 1) + 2 * ((z >> q) & 1)] for q in range(n)),
        "weight": (x | z).bit_count(),
    }


def _enumeration_distance(check_x, check_z, *, max_weight=None, max_candidates=2_000_000):
    """Compute or rigorously bound the distance of commuting Pauli checks.

    ``check_x`` and ``check_z`` are equally shaped binary arrays, one Pauli
    generator per row.  Redundant and zero generators are supported.
    ``max_weight=d-1`` searches just far enough to certify distance >= d.
    ``max_candidates`` bounds the number of enumerated Pauli candidates;
    linear algebra used to validate checks and find an upper-bound witness
    is not charged against this enumeration budget.

    The JSON-safe result has status ``exact``, ``bounded``, ``unknown`` or
    ``no_logicals``.  For a code with logical qubits, the lower bound is
    always rigorous, including on budget exhaustion.  ``unknown`` means no
    nonzero weight shell was completed before budget exhaustion, not that
    the reported trivial lower bound is invalid.  A stabilizer state has
    no nontrivial logical operator: report ``distance=None`` together with
    ``distance_is_infinite=True``, avoiding nonstandard JSON Infinity.

    The search is exponential; no numerical heuristic is promoted to an
    exact distance or a lower-bound certificate.
    """
    x = binary_matrix(check_x, name="check_x")
    z = binary_matrix(check_z, name="check_z")
    if x.shape == (0, 0) and z.shape[0] == 0:
        x = np.zeros_like(z)
    if z.shape == (0, 0) and x.shape[0] == 0:
        z = np.zeros_like(x)
    if x.shape != z.shape:
        raise ValueError("check_x and check_z must have the same shape")
    n = int(x.shape[1])
    max_candidates = _integer(max_candidates, "max_candidates")
    if max_weight is not None:
        max_weight = _integer(max_weight, "max_weight")
    limit = n if max_weight is None else min(max_weight, n)
    xs, zs = [_bits(row) for row in x], [_bits(row) for row in z]
    for index, (left_x, left_z) in enumerate(zip(xs, zs)):
        for right_x, right_z in zip(xs[:index], zs[:index]):
            if ((left_x & right_z) ^ (left_z & right_x)).bit_count() & 1:
                raise ValueError("Stabilizer generators do not commute")
    stabilizers = _basis(a | (b << n) for a, b in zip(xs, zs))
    k = n - len(stabilizers)
    css = all(not a or not b for a, b in zip(xs, zs))
    report = {
        "status": "unknown", "distance": None,
        "distance_is_infinite": False,
        "proven_lower_bound": 1, "upper_bound": None, "witness": None,
        "n": n, "k": k, "stabilizer_rank": len(stabilizers),
        "method": "css_pure_pauli_weight_enumeration" if css else "symplectic_pauli_weight_enumeration",
        "distance_kind": "static_stabilizer",
        "max_weight": limit, "max_candidates": max_candidates,
        "checked_candidates": 0, "completed_weight": 0,
        "stopped_at_weight": None, "stop_reason": None,
    }
    if k == 0:
        report.update(status="no_logicals", distance_is_infinite=True,
                      proven_lower_bound=None, stop_reason="no_logical_qubits")
        return report

    # Every centralizer basis spans all logical cosets with stabilizers.
    # At least one basis row is outside the stabilizer when k>0; any such
    # row supplies a valid upper-bound witness even with zero search budget.
    if css:
        upper_candidates = list(_kernel_rows(zs, n))
        upper_candidates += [value << n for value in _kernel_rows(xs, n)]
    else:
        upper_candidates = _kernel_rows((b | (a << n) for a, b in zip(xs, zs)), 2 * n)
    upper_value = None
    for value in upper_candidates:
        if not _contains(value, stabilizers):
            witness = _witness(value, n)
            if report["upper_bound"] is None or witness["weight"] < report["upper_bound"]:
                upper_value = value
                report["upper_bound"] = witness["weight"]
                report["witness"] = witness
    if upper_value is None:
        raise RuntimeError("No logical witness despite positive logical dimension")

    # A weight-one witness meets the universal lower bound without search.
    if report["upper_bound"] == 1:
        report.update(status="exact", distance=1, stop_reason="bounds_meet")
        return report

    syndrome_x = [_bits(z[:, q]) for q in range(n)]
    syndrome_z = [_bits(x[:, q]) for q in range(n)]

    def candidates(weight):
        for support in combinations(range(n), weight):
            if css:
                mask = sx = sz = 0
                for q in support:
                    mask |= 1 << q
                    sx ^= syndrome_x[q]
                    sz ^= syndrome_z[q]
                yield mask, sx
                yield mask << n, sz
            else:
                options = [((1 << q, syndrome_x[q]),
                            (1 << (n + q), syndrome_z[q]),
                            ((1 << q) | (1 << (n + q)), syndrome_x[q] ^ syndrome_z[q]))
                           for q in support]
                for assignments in product(*options):
                    value = syndrome = 0
                    for local_value, local_syndrome in assignments:
                        value |= local_value
                        syndrome ^= local_syndrome
                    yield value, syndrome

    for weight in range(1, min(limit, report["upper_bound"] - 1) + 1):
        for value, syndrome in candidates(weight):
            if report["checked_candidates"] == max_candidates:
                report.update(status="bounded" if report["completed_weight"] else "unknown",
                              stopped_at_weight=weight, stop_reason="candidate_budget")
                return report
            report["checked_candidates"] += 1
            if syndrome == 0 and not _contains(value, stabilizers):
                report.update(status="exact", distance=weight,
                              proven_lower_bound=weight, upper_bound=weight,
                              witness=_witness(value, n), stopped_at_weight=weight,
                              stop_reason="logical_witness_found")
                return report
        report["completed_weight"] = weight
        report["proven_lower_bound"] = weight + 1
    if report["proven_lower_bound"] == report["upper_bound"]:
        report.update(status="exact", distance=report["upper_bound"], stop_reason="bounds_meet")
    else:
        report.update(status="bounded", stop_reason="weight_limit")
    return report


def code_distance(code, **kwargs):
    """Distance report for a CSSCode or a symplectic deformed-code object.

    An object's declared ``distance`` is never used as evidence.  For an
    object with gauge qubits this still computes its stabilizer-code
    distance, counting gauge logicals; it is not a dressed-distance solver.
    """
    if hasattr(code, "check_x") and hasattr(code, "check_z"):
        if callable(getattr(code, "validate", None)):
            code.validate()
        report = stabilizer_distance(code.check_x, code.check_z, **kwargs)
        if hasattr(code, "gauge_qubits"):
            report["gauge_qubits"] = int(code.gauge_qubits)
        return report
    if hasattr(code, "hx") and hasattr(code, "hz"):
        hx = binary_matrix(code.hx, name="hx")
        hz = binary_matrix(code.hz, name="hz")
        if hx.shape == (0, 0):
            hx = np.zeros((0, hz.shape[1]), dtype=np.uint8)
        if hz.shape == (0, 0):
            hz = np.zeros((0, hx.shape[1]), dtype=np.uint8)
        if hx.shape[1] != hz.shape[1]:
            raise ValueError("hx and hz must have the same column count")
        return stabilizer_distance(np.vstack([hx, np.zeros_like(hz)]),
                                   np.vstack([np.zeros_like(hx), hz]), **kwargs)
    raise TypeError("code must expose hx/hz or check_x/check_z matrices")


def _float_limit(value, name):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be None or a finite nonnegative number")
    return float(value)


def _distance_interval(report):
    """Expose inclusive integer bounds and an unambiguous readable statement."""
    lower, upper = report["proven_lower_bound"], report["upper_bound"]
    infinite = bool(report["distance_is_infinite"])
    report["distance_interval"] = dict(lower_bound=lower, upper_bound=upper,
        lower_inclusive=True, upper_inclusive=True, distance_is_infinite=infinite)
    if infinite:
        statement = "d = infinity (no logical qubits)"
    elif report["status"] == "exact":
        statement = f"d = {report['distance']}"
    elif upper is None:
        statement = f"d > {lower - 1}"
    else:
        statement = f"{lower - 1} < d <= {upper}"
    report["distance_statement"] = statement
    return report


def stabilizer_distance(check_x, check_z, *, max_weight=None, max_candidates=2_000_000,
                        method="auto", ris_trials=64, seed=0, time_limit=60.0,
                        node_limit=None, _stop_below=None):
    """Compute or bound static distance, with independently verifiable witnesses.

    ``auto`` first uses randomized information sets (RIS) for an upper bound,
    then prefers optional Gurobi for both CSS and mixed checks. When Gurobi is
    unavailable, use connected clusters for CSS or SciPy/HiGHS for mixed checks.
    A completed solver call, including a timeout, returns its bounds directly;
    it never starts an additional cluster search after exhausting that budget.
    ``enumeration`` retains the original exhaustive small-code reference.
    ``ris`` returns an upper bound only (except when it meets an existing
    rigorous lower bound); ``cluster``, ``milp`` (HiGHS), and ``gurobi`` select
    backends that can establish lower bounds or the exact distance.

    max_weight limits the weight whose absence needs proving. Independent
    budgets are ris_trials, max_candidates (enumeration/cluster states), and
    time_limit/node_limit (MILP solve only, excluding model construction).
    The default MILP solve limit is 60 seconds. The overall surgery call has
    no global wall-clock limit. Incomplete solves return ``distance=None`` and
    inclusive ``distance_interval`` bounds (e.g. ``8 < d <= 10`` means 9 or 10).
    """
    return _distance_interval(_stabilizer_distance(check_x, check_z,
        max_weight=max_weight, max_candidates=max_candidates, method=method,
        ris_trials=ris_trials, seed=seed, time_limit=time_limit,
        node_limit=node_limit, _stop_below=_stop_below))


def _stabilizer_distance(check_x, check_z, *, max_weight, max_candidates,
                         method, ris_trials, seed, time_limit, node_limit, _stop_below):
    if not isinstance(method, str) or method not in {"auto", "enumeration", "ris", "cluster", "milp", "gurobi"}:
        raise ValueError("method must be auto, enumeration, ris, cluster, milp, or gurobi")
    max_candidates = _integer(max_candidates, "max_candidates")
    ris_trials = _integer(ris_trials, "ris_trials")
    seed = _integer(seed, "seed")
    time_limit = _float_limit(time_limit, "time_limit")
    if node_limit is not None:
        node_limit = _integer(node_limit, "node_limit")
    if _stop_below is not None:
        _stop_below = _integer(_stop_below, "target_distance", minimum=1)

    def counterexample(report):
        if (_stop_below is not None and report["upper_bound"] is not None
                and report["upper_bound"] < _stop_below):
            if report["status"] != "exact":
                report.update(status="bounded", stop_reason="target_counterexample")
            return True
        return False

    if method == "enumeration":
        if _stop_below is not None:
            report = _enumeration_distance(check_x, check_z, max_weight=max_weight, max_candidates=0)
            if counterexample(report):
                report.update(max_candidates=max_candidates, requested_method=method,
                              backend="linear_algebra", candidate_count_kind="none")
                return report
        report = _enumeration_distance(check_x, check_z, max_weight=max_weight,
                                       max_candidates=max_candidates)
        report.update(requested_method=method, backend="enumeration",
                      candidate_count_kind="complete_pauli_assignments")
        return report

    # This validates all check matrices and supplies a deterministic witness.
    report = _enumeration_distance(check_x, check_z, max_weight=max_weight, max_candidates=0)
    report.update(max_candidates=max_candidates, requested_method=method,
                  backend="linear_algebra", candidate_count_kind="none")
    x, z = binary_matrix(check_x), binary_matrix(check_z)
    if x.shape == (0, 0) and z.shape[0] == 0:
        x = np.zeros_like(z)
    if z.shape == (0, 0) and x.shape[0] == 0:
        z = np.zeros_like(x)
    if report["status"] in {"exact", "no_logicals"} or counterexample(report):
        return report

    from ._distance_ris import ris_distance
    report = ris_distance(x, z, report, trials=ris_trials, seed=seed)
    if report["status"] == "exact":
        report["backend"] = "ris"
        return report
    if counterexample(report):
        report.update(method="random_information_sets", backend="ris")
        return report
    if method == "ris":
        report.update(method="random_information_sets", backend="ris",
                      status="bounded", stop_reason="heuristic_budget")
        return report
    if report["max_weight"] == 0:
        report.update(status="bounded", backend="linear_algebra+ris", stop_reason="weight_limit")
        return report

    if method in {"auto", "gurobi"}:
        try:
            from ._distance_gurobi import gurobi_distance
            report = gurobi_distance(x, z, report, time_limit=time_limit, node_limit=node_limit)
        except ImportError as exc:
            if method == "gurobi":
                raise ImportError("Gurobi distance requires gurobipy and a usable license; "
                                  "install codesurgery[gurobi] or choose method='milp'/'cluster'") from exc
            report["gurobi_unavailable_reason"] = str(exc)
        else:
            report["backend"] = "gurobi"
            return report

    css = bool(np.all(~np.any(x, axis=1) | ~np.any(z, axis=1)))
    selected = ("cluster" if css else "milp") if method == "auto" else method
    if selected == "milp":
        try:
            from ._distance_milp import milp_distance
            report = milp_distance(x, z, report, time_limit=time_limit, node_limit=node_limit)
            report["backend"] = "milp"
        except ImportError as exc:
            if method == "milp":
                raise ImportError("MILP distance requires scipy>=1.9; install codesurgery[distance] "
                                  "or choose method='cluster'") from exc
            report["fallback_reason"] = "SciPy MILP unavailable"
        else:
            return report
    from ._distance_cluster import cluster_distance
    report = cluster_distance(x, z, report, max_candidates=max_candidates)
    report["backend"] = "milp+cluster" if "milp" in report else "cluster"
    return report


def verify_distance(code, target_distance, **kwargs):
    """Decide d >= target_distance; target_met is True, False, or None.

    Searches only weights <= D-1 (or a smaller caller-specified max_weight).
    The claimed target is not itself used as evidence. A witness below D
    disproves it; a rigorous lower bound >= D proves it; otherwise unknown.
    """
    target = _integer(target_distance, "target_distance", minimum=1)
    supplied = kwargs.pop("max_weight", None)
    limit = target - 1 if supplied is None else min(target - 1, _integer(supplied, "max_weight"))
    kwargs["_stop_below"] = target
    report = code_distance(code, max_weight=limit, **kwargs)
    if report["distance_is_infinite"]:
        met = True
    elif report["upper_bound"] is not None and report["upper_bound"] < target:
        met = False
    elif report["proven_lower_bound"] >= target:
        met = True
    else:
        met = None
    report.update(target_distance=target, target_met=met,
                  target_status="sufficient" if met is True else "insufficient" if met is False else "unknown")
    return report


__all__ = ["stabilizer_distance", "code_distance", "verify_distance"]
