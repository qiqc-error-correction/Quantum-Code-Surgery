"""Choose which compatible targets need branching before paired gauging.

The finite-size policy bounds *each original* check's weight increase and each
original physical qubit's degree increase. New auxiliary checks have no input
baseline; their actual weights are audited separately. This planner does not
perform measurements, prove graph expansion, or impose a new distance claim.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..gf2 import binary_matrix
from .direct_attachment import direct_loads


class BranchingBudgetError(ValueError):
    """No candidate routing can satisfy the configured original-code budget."""


def _limit(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _roots(value, target_count):
    if not isinstance(value, (list, tuple)):
        raise ValueError("root_groups must be a list of nonempty target-index groups")
    result, seen = [], set()
    for group in value:
        if (not isinstance(group, (list, tuple, np.ndarray))
                or isinstance(group, np.ndarray) and group.ndim != 1
                or len(group) == 0):
            raise ValueError("Every root group must be a nonempty target-index group")
        normalized = []
        for index in group:
            if (isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    or not 0 <= index < target_count):
                raise ValueError("Root groups contain an invalid target index")
            index = int(index)
            if index in seen:
                raise ValueError("Root groups must contain unique, disjoint target indices")
            seen.add(index)
            normalized.append(index)
        result.append(tuple(normalized))
    return tuple(result)


class _Routing:
    def __init__(self, check_x, check_z, target_x, target_z):
        self.rows, self.columns = direct_loads(check_x, check_z, target_x, target_z)
        self.t, self.m = self.rows.shape
        if self.t == 0:
            raise ValueError("At least one nontrivial logical target is required")
        self.n = self.columns.shape[1]
        self.total_rows = self.rows.sum(axis=0)
        self.total_columns = self.columns.sum(axis=0)
        checks = binary_matrix(check_x) | binary_matrix(check_z)
        self.original_rows = checks.sum(axis=1, dtype=np.int64)
        self.original_columns = checks.sum(axis=0, dtype=np.int64)
        self.group = lru_cache(maxsize=16_384)(self._group)
        self.descendants = lru_cache(maxsize=16_384)(self._descendants)

    def _group(self, group):
        # QWC means a union has precisely the OR of the targets' local
        # anticommutation incidences. The union itself need not normalize Q.
        incident = np.any(self.rows[list(group)] > 0, axis=0).astype(np.int64)
        support = np.any(self.columns[list(group)], axis=0).astype(np.int64)
        return incident, support, int(incident.sum() + support.sum())

    def _descendants(self, group):
        if len(group) < 2:
            return 0
        mid = len(group) // 2
        return sum(self.group(child)[2] + self.descendants(child)
                   for child in (group[:mid], group[mid:]))

    def evaluate(self, groups):
        branched = tuple(sorted(index for group in groups for index in group))
        marked = set(branched)
        direct = [i for i in range(self.t) if i not in marked]
        rows, columns = self.total_rows.copy(), self.total_columns.copy()
        if branched:
            rows -= self.rows[list(branched)].sum(axis=0)
            columns -= self.columns[list(branched)].sum(axis=0)
        ancillas = 0
        for group in groups:
            incident, support, size = self.group(group)
            rows += incident
            columns += support
            ancillas += size + self.descendants(group)
        return dict(ancillas=int(ancillas), direct_targets=direct,
                    branched_targets=list(branched), root_groups=[list(g) for g in groups],
                    projected_original_row_growth=rows.tolist(),
                    projected_original_column_growth=columns.tolist())

    def candidates(self, branched):
        if not branched:
            return [self.evaluate(())]
        if len(branched) == 1:
            return [self.evaluate((branched,))]
        mid = len(branched) // 2
        # Two roots recover the old tree and use fewer stickers. A shared
        # root costs one extra sticker but consumes less memory connectivity.
        return [self.evaluate((branched[:mid], branched[mid:])),
                self.evaluate((branched,))]


def evaluate_routing(check_x, check_z, target_x, target_z, root_groups):
    """Audit a fixed root layout without searching for a new layout.

    Every root is a two-layer sticker followed by the existing balanced tree
    for its group. Targets absent from all roots are measured directly on the
    input code. Returned growth includes these future paired-graph attachments
    as well as the root stickers. It is not just the branching matrix weight.
    """
    routing = _Routing(check_x, check_z, target_x, target_z)
    return routing.evaluate(_roots(root_groups, routing.t))


def _overload(route, row_limit, column_limit):
    rows = np.maximum(0, np.asarray(route["projected_original_row_growth"]) - row_limit)
    columns = np.maximum(0, np.asarray(route["projected_original_column_growth"]) - column_limit)
    return (int(np.count_nonzero(rows) + np.count_nonzero(columns)),
            int(rows.sum() + columns.sum()),
            int(max(rows.max(initial=0), columns.max(initial=0))))


def _cost(route):
    return (route["ancillas"], len(route["branched_targets"]),
            tuple(route["branched_targets"]), tuple(tuple(g) for g in route["root_groups"]))


def plan_branching(check_x, check_z, target_x, target_z, *, branching_mode="adaptive",
                   max_check_increase=2, max_qubit_increase=2, force_copy=False):
    """Minimize stickers subject to cumulative original-row/column budgets.

    Up to twelve targets, enumerate every direct/branched subset and two root
    layouts per subset: balanced roots and one shared root. Above twelve use
    a bounded deterministic greedy search, with an all-branched fallback and
    removal of redundant branches. Optimality, where reported, concerns this
    candidate family only, not all possible trees or physical representatives.

    ``always`` preserves the legacy balanced tree and only reports budgets.
    For a single target ``force_copy`` forces one sticker in either mode. It
    has no additional effect for multiple targets. No recursion imposes a
    fictitious original-weight baseline on newly introduced auxiliary checks.
    """
    if not isinstance(branching_mode, str) or branching_mode not in ("adaptive", "always"):
        raise ValueError("branching_mode must be adaptive or always")
    row_limit = _limit(max_check_increase, "max_check_increase")
    column_limit = _limit(max_qubit_increase, "max_qubit_increase")
    if not isinstance(force_copy, (bool, np.bool_)):
        raise ValueError("force_copy must be a boolean")
    ctx = _Routing(check_x, check_z, target_x, target_z)
    all_direct = ctx.evaluate(())
    selection, optimal, examined = "all_direct", True, 1
    forced = bool(force_copy and ctx.t == 1)

    def feasible(route):
        return _overload(route, row_limit, column_limit)[0] == 0

    if branching_mode == "always":
        if ctx.t > 1:
            indices = tuple(range(ctx.t))
            mid = ctx.t // 2
            best = ctx.evaluate((indices[:mid], indices[mid:]))
        else:
            best = ctx.evaluate(((0,),) if forced else ())
        selection, optimal = "legacy_balanced", False
    elif forced:
        best = ctx.evaluate(((0,),))
        selection = "forced_copy"
    elif feasible(all_direct):
        best = all_direct
    elif ctx.t <= 12:
        best = None
        selection = "exact_subset_enumeration"
        for mask in range(1, 1 << ctx.t):
            branched = tuple(i for i in range(ctx.t) if mask & (1 << i))
            for route in ctx.candidates(branched):
                examined += 1
                if feasible(route) and (best is None or _cost(route) < _cost(best)):
                    best = route
    else:
        selection, optimal = "bounded_greedy", False
        all_branched = ctx.candidates(tuple(range(ctx.t)))
        examined += len(all_branched)
        feasible_routes = [r for r in all_branched if feasible(r)]
        best = min(feasible_routes, key=_cost) if feasible_routes else None
        chosen = ()
        # Adding one target can leave the overload unchanged until another
        # overlapping target joins its root. Never stop merely for that reason.
        for _ in range(min(ctx.t, 64)):
            trials = []
            for i in range(ctx.t):
                if i not in chosen:
                    trials.extend(ctx.candidates(tuple(sorted((*chosen, i)))))
            examined += len(trials)
            if not trials:
                break
            step = min(trials, key=lambda r: (*_overload(r, row_limit, column_limit), *_cost(r)))
            chosen = tuple(step["branched_targets"])
            if feasible(step):
                if best is None or _cost(step) < _cost(best):
                    best = step
                break
        # A reverse pass can remove more than one branch without requiring an
        # exponential subset search. Repeat at most t times, strictly shrinking.
        if best is not None:
            for _ in range(ctx.t):
                trials = []
                chosen = tuple(best["branched_targets"])
                for removed in chosen:
                    trials.extend(ctx.candidates(tuple(i for i in chosen if i != removed)))
                examined += len(trials)
                better = [r for r in trials if feasible(r) and _cost(r) < _cost(best)]
                if not better:
                    break
                best = min(better, key=_cost)

    if branching_mode == "adaptive" and (best is None or not feasible(best)):
        rows = np.flatnonzero(np.asarray(all_direct["projected_original_row_growth"]) > row_limit).tolist()
        columns = np.flatnonzero(np.asarray(all_direct["projected_original_column_growth"]) > column_limit).tolist()
        raise BranchingBudgetError("No branching route satisfies the original per-check/per-qubit growth limits "
                         f"(+{row_limit}, +{column_limit}); direct attachment exceeds original check rows "
                         f"{rows} and physical qubits {columns}. A nonempty attachment requires at least "
                         "one connection on each touched original item; increase the permitted budget.")

    report = dict(best)
    report.update(mode=branching_mode,
        max_check_increase=row_limit, max_qubit_increase=column_limit,
        limits=dict(max_check_increase=row_limit, max_qubit_increase=column_limit,
                    scope="original_checks_and_physical_qubits"),
        selection_method=selection, candidates_evaluated=examined,
        optimal_within_candidate_family=optimal,
        original_limits_satisfied=feasible(best),
        original_row_weights=ctx.original_rows.tolist(),
        original_column_degrees=ctx.original_columns.tolist(),
        projected_original_row_weights=(ctx.original_rows + np.asarray(best["projected_original_row_growth"])).tolist(),
        projected_original_column_degrees=(ctx.original_columns + np.asarray(best["projected_original_column_growth"])).tolist(),
        direct_all_row_growth=all_direct["projected_original_row_growth"],
        direct_all_column_growth=all_direct["projected_original_column_growth"],
        measurement_model="paired_graph: exact original support growth; expansion and auxiliary sparsity separate",
        auxiliary_limits_enforced=False,
        events=[dict(kind="direct_attachment_overflow",
                     check_rows=np.flatnonzero(ctx.total_rows > row_limit).tolist(),
                     physical_qubits=np.flatnonzero(ctx.total_columns > column_limit).tolist()),
                dict(kind="selected_roots", groups=best["root_groups"])])
    return report


__all__ = ["BranchingBudgetError", "plan_branching", "evaluate_routing"]
