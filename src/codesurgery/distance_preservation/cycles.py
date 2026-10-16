"""Bounded, deterministic cycle-basis search before changing graph topology.

This is a finite-size heuristic, not a minimum-congestion or minimum-cycle-
basis solver. A replacement is accepted only when it spans the same complete
cycle space and meets both requested bounds. Otherwise callers keep their
existing decongestion/cellulation fallback.
"""
from collections import deque

import numpy as np

from ..universal_adapters.graphs import adjacency, cycle_basis


def optimization_options(enabled, max_roots, max_candidates):
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("optimize_cycle_basis must be a boolean")
    for name, value in (("cycle_basis_max_roots", max_roots),
                        ("cycle_basis_max_candidates", max_candidates)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    return bool(enabled), int(max_roots), int(max_candidates)


def _metrics(cycles):
    return dict(cycle_generators=int(len(cycles)),
                max_cycle_weight=int(cycles.sum(axis=1).max(initial=0)),
                max_cycle_congestion=int(cycles.sum(axis=0).max(initial=0)))


def _independent_rows(ordered, dimension):
    pivots, selected = {}, []
    for row in ordered:
        remainder = row
        while remainder:
            pivot = remainder.bit_length() - 1
            if pivot in pivots:
                remainder ^= pivots[pivot]
            else:
                pivots[pivot] = remainder
                selected.append(row)
                break
        if len(selected) == dimension:
            break
    return selected


def _matrix(rows, edges):
    width = (edges + 7) // 8
    packed = b''.join(row.to_bytes(width, 'little') for row in rows)
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8).reshape(len(rows), width),
                         axis=1, bitorder='little')[:, :edges].copy()


def optimize_cycles(b, cycles, *, max_cycle_weight, max_cycle_congestion,
                    enabled=True, max_roots=64, max_candidates=10_000,
                    max_entries=50_000_000):
    """Try replacing a caller-validated complete cycle generating matrix.

    The graph and input cycle matrix are validated by the calling sparsifier.
    Root count and the number of distinct retained candidates are bounded;
    candidate storage also respects max_entries in dense-equivalent entries.
    Existing generators are candidates even when redundant or non-simple.
    """
    enabled, max_roots, max_candidates = optimization_options(enabled, max_roots, max_candidates)
    edges, vertices = b.shape
    dimension = edges - vertices + 1
    before = _metrics(cycles)
    report = dict(enabled=enabled, attempted=False, changed=False,
                  method='bounded multi-root BFS cycles and greedy GF(2) independence',
                  heuristic=True, optimality_guaranteed=False,
                  before=before, after=dict(before), cycle_rank=dimension,
                  graph_unchanged=True, matching_unchanged=True,
                  max_roots=max_roots, max_candidates=max_candidates,
                  roots_tried=0, candidate_count=0, search_truncated=False,
                  cycle_space_verified=True)

    def fits(metrics):
        return (metrics['max_cycle_weight'] <= max_cycle_weight
                and metrics['max_cycle_congestion'] <= max_cycle_congestion)

    if not enabled or fits(before):
        report['reason'] = 'disabled' if not enabled else 'existing generators already meet bounds'
        return cycles, report
    report['attempted'] = True
    limit = min(max_candidates, max_entries // max(1, edges))
    report['effective_candidate_limit'] = limit
    if limit < dimension:
        report.update(reason='candidate budget cannot contain a complete basis', search_truncated=True)
        return cycles, report

    candidates = set()
    truncated = False

    def add(row):
        nonlocal truncated
        if not row or row in candidates:
            return True
        if len(candidates) >= limit:
            truncated = True
            return False
        candidates.add(row)
        return True

    for row in cycles:
        if not add(int.from_bytes(np.packbits(row, bitorder='little').tobytes(), 'little')):
            break
    adj = adjacency(b)
    endpoints = [tuple(map(int, np.flatnonzero(row))) for row in b]
    roots = np.linspace(0, vertices - 1, min(vertices, max_roots), dtype=int).tolist()
    for root in roots:
        if truncated:
            break
        paths = [None] * vertices
        paths[root] = 0
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, edge in adj[u]:
                if paths[v] is None:
                    paths[v] = paths[u] ^ (1 << edge)
                    queue.append(v)
        report['roots_tried'] += 1
        for edge, (u, v) in enumerate(endpoints):
            if not add(paths[u] ^ paths[v] ^ (1 << edge)):
                break
    report.update(candidate_count=len(candidates),
                  search_truncated=truncated or len(roots) < vertices)
    # Exclude rows that cannot meet the fixed row-weight cap. If these cannot
    # span, changing the order cannot succeed and topology repair is needed.
    eligible = sorted((row for row in candidates if row.bit_count() <= max_cycle_weight),
                      key=lambda row: (row.bit_count(), row))
    selected = _independent_rows(eligible, dimension)
    if len(selected) != dimension:
        report.update(reason='bounded candidate pool has no complete basis within weight cap',
                      eligible_rank=len(selected))
        return cycles, report

    def accept(rows, strategy):
        proposed = _matrix(rows, edges)
        metrics = _metrics(proposed)
        report.setdefault('strategies', []).append(dict(strategy=strategy, **metrics))
        if not fits(metrics):
            return None
        # Kernel inclusion and full cycle rank prove equality with the validated
        # input space, including parallel-edge cycles and redundant inputs.
        proposed = cycle_basis(b, proposed, max_entries=max_entries)
        report.update(changed=not np.array_equal(proposed, cycles), after=metrics,
                      reason='complete equivalent cycle basis meets both bounds',
                      selected_strategy=strategy)
        return proposed

    proposed = accept(selected, 'shortest_first')
    if proposed is not None:
        return proposed, report
    # Two additional deterministic orderings discourage heavily reused edges.
    # No unbounded backtracking; failure keeps the original fallback intact.
    frequency = [0] * edges
    for row in eligible:
        remaining = row
        while remaining:
            bit = remaining & -remaining
            edge = bit.bit_length() - 1
            frequency[edge] += 1
            remaining ^= bit
    costs = {}
    for row in eligible:
        remaining, cost = row, 0
        while remaining:
            bit = remaining & -remaining
            cost += frequency[bit.bit_length() - 1]
            remaining ^= bit
        costs[row] = cost
    for strategy, key in (
        ('shortest_then_edge_frequency', lambda row: (row.bit_count(), costs[row], row)),
        ('edge_frequency_first', lambda row: (costs[row], row.bit_count(), row)),
    ):
        proposed = accept(_independent_rows(sorted(eligible, key=key), dimension), strategy)
        if proposed is not None:
            return proposed, report
    report['reason'] = 'no tested equivalent basis meets both bounds; use topology fallback'
    return cycles, report
