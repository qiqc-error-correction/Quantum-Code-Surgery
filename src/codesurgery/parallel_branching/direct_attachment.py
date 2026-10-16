"""Exact support preview for direct, locally compatible gauging attachments.

This deterministic preview pairs each check's even local boundary, joins the
resulting components, and includes a complete cycle basis. It measures the
requested logical span, unlike a branching sticker. It does not certify graph
expansion, distance, or fault tolerance and is used only to budget row/column
growth before deciding which targets need branching.
"""
from __future__ import annotations

import numpy as np

from ..gf2 import binary_matrix, binary_vector, in_rowspace, solve
from ..symplectic import pauli_product
from .construction import _signed_check_rank


def _validate_targets(x, z, tx, tz):
    if x.shape != z.shape or x.shape[1] == 0:
        raise ValueError("check_x/check_z must have equal shapes and positive width")
    if tx.shape != tz.shape:
        raise ValueError("target_x/target_z must have equal shapes")
    old_generators = np.hstack((x, z))
    for i, (xx, zz) in enumerate(zip(tx, tz)):
        if in_rowspace(np.r_[xx, zz], old_generators):
            raise ValueError(f"Target {i} is trivial modulo input stabilizers")
        if np.any((x @ zz + z @ xx) % 2):
            raise ValueError(f"Target {i} must commute with every input check")
        for j in range(i):
            if np.any((xx * tz[j]) ^ (zz * tx[j])):
                raise ValueError("Targets must be qubit-wise Pauli compatible")


def direct_loads(check_x, check_z, target_x, target_z):
    """Return per-target row and column growth, shaped ``(t,m)`` and ``(t,n)``.

    Row increments are half the local anticommutation boundary weight. Column
    increments are the physical target support. These are the exact increments
    on old checks and qubits for :func:`preview_direct_attachment`; added graph
    connection edges and cycle checks do not affect either memory increment.
    This helper has no phase arguments and validates binary commutation only.
    """
    x = binary_matrix(check_x, name="check_x")
    z = binary_matrix(check_z, name="check_z")
    if x.shape != z.shape or x.shape[1] == 0:
        raise ValueError("check_x/check_z must have equal shapes and positive width")
    tx = binary_matrix(target_x, n_cols=x.shape[1], name="target_x")
    tz = binary_matrix(target_z, n_cols=x.shape[1], name="target_z")
    # Check input commutation row-by-row to avoid allocating an m-by-m matrix.
    for i in range(len(x)):
        if np.any((x[:i] @ z[i] + z[:i] @ x[i]) % 2):
            raise ValueError("Stabilizer checks do not commute")
    _validate_targets(x, z, tx, tz)
    loads = np.zeros((len(tx), len(x)), np.int64)
    for i, (xx, zz) in enumerate(zip(tx, tz)):
        support = np.flatnonzero(xx | zz)
        boundary = (x[:, support] * zz[support]) ^ (z[:, support] * xx[support])
        loads[i] = boundary.sum(axis=1, dtype=np.int64) // 2
    return loads, (tx | tz).astype(np.int64)


def _sector(x, z, signs, tx, tz, target_signs):
    """Select one consistent outcome sector, including dependent observables."""
    ax, az, ass = x.copy(), z.copy(), signs.copy()
    outcomes = np.zeros(len(tx), np.uint8)
    independent = []
    for i, (xx, zz, sign) in enumerate(zip(tx, tz, target_signs)):
        generators = np.hstack((ax, az))
        if in_rowspace(np.r_[xx, zz], generators):
            coeff = solve(generators.T, np.r_[xx, zz])
            _, _, phase = pauli_product(ax, az, ass, np.flatnonzero(coeff))
            if phase & 1:
                raise ValueError("Target relation has a non-Hermitian phase")
            outcomes[i] = (phase // 2) ^ int(sign)
        else:
            # A +1 outcome of the signed independent observable is a valid
            # arbitrary preview sector; dependent outcomes are then forced.
            ax = np.vstack((ax, xx))
            az = np.vstack((az, zz))
            ass = np.r_[ass, np.uint8(sign)]
            independent.append(i)
    return outcomes, independent


def _graph_plan(x, z, tx, tz):
    support = np.flatnonzero(tx | tz)
    w = len(support)
    pairs, row_edges, owners = [], [], []
    parent = list(range(w))

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def add_edge(u, v, owner=-1):
        edge = (min(u, v), max(u, v))
        # The default gauging paired_graph gives every check boundary pair
        # its own edge. Equal endpoints owned by different checks therefore
        # remain parallel edges, with their own cycle relation and qubit.
        index = len(pairs)
        pairs.append(edge)
        owners.append(owner)
        a, b = root(u), root(v)
        if a != b:
            parent[b] = a
        return index

    # Do not first allocate the full restriction: the budget is checked before
    # any of the returned F, B, M or cycle matrices are allocated.
    for i, (xr, zr) in enumerate(zip(x, z)):
        boundary = np.flatnonzero((xr[support] * tz[support])
                                 ^ (zr[support] * tx[support]))
        if len(boundary) % 2:
            raise ValueError("Each target must commute with every input check")
        row_edges.append([add_edge(int(u), int(v), i)
                          for u, v in zip(boundary[::2], boundary[1::2])])
    components_by_root = {}
    degrees = [0] * w
    for u, v in pairs:
        degrees[u] += 1
        degrees[v] += 1
    for v in range(w):
        components_by_root.setdefault(root(v), []).append(v)
    # Match paired_graph's deterministic chain: visit components by their
    # smallest vertex, and choose the least-degree vertex within each one.
    components = [min(vertices, key=lambda v: (degrees[v], v))
                  for vertices in components_by_root.values()]
    connecting_edges = [add_edge(u, v) for u, v in zip(components, components[1:])]
    return dict(support=support, edges=pairs, row_edges=row_edges, edge_owners=owners,
                connecting_edges=connecting_edges,
                components_before_connecting=len(components),
                cycle_count=len(pairs) - w + 1)


def _cycle_basis(edges, vertex_count):
    """A full fundamental cycle basis, without a dense nullspace workspace."""
    parent = list(range(vertex_count))
    adjacent = [[] for _ in parent]
    chords = []

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e, (u, v) in enumerate(edges):
        a, b = root(u), root(v)
        if a == b:
            chords.append((e, u, v))
        else:
            parent[b] = a
            adjacent[u].append((v, e))
            adjacent[v].append((u, e))
    cycles = np.zeros((len(chords), len(edges)), np.uint8)
    for i, (chord, start, end) in enumerate(chords):
        previous = {start: (-1, -1)}
        queue = [start]
        for u in queue:
            if u == end:
                break
            for v, e in adjacent[u]:
                if v not in previous:
                    previous[v] = (u, e)
                    queue.append(v)
        cycles[i, chord] = 1
        v = end
        while v != start:
            v, e = previous[v]
            cycles[i, e] = 1
    return cycles


def preview_direct_attachment(check_x, check_z, target_x, target_z, *,
                              check_signs=None, target_signs=None,
                              max_entries=50_000_000):
    """Build a direct-gauging preview and its exact memory support increments.

    Inputs are binary row matrices. Targets must be nontrivial normalizers and
    qubit-wise compatible (common nonidentity Pauli wherever they overlap).
    Empty target batches are allowed. Returned ``memory_row_growth`` has one
    entry per original check and ``memory_column_growth`` per physical qubit.

    ``check_signs`` describes one consistent *preview* outcome sector, not
    observed measurement outcomes. ``target_outcome_signs`` contains the
    corresponding +1/-1 bits for the signed targets. In particular, opposite
    signed copies cannot both be assigned +1. All added cycle checks are +X.

    Each graph is separate, even when targets overlap. ``B`` is edge-by-vertex,
    ``F`` is old-check-by-vertex and ``M`` is old-check-by-edge, with MB=F.
    Like the existing default paired_graph builder, equal endpoint pairs
    owned by distinct old checks remain distinct parallel edges. New vertex
    checks always use Z on graph edges; old-check dressing and cycles use X.
    This fixed ancilla convention also covers arbitrary mixed/Y memory axes.
    """
    if (isinstance(max_entries, (bool, np.bool_))
            or not isinstance(max_entries, (int, np.integer)) or max_entries < 1):
        raise ValueError("max_entries must be a positive integer")
    x = binary_matrix(check_x, name="check_x")
    z = binary_matrix(check_z, name="check_z")
    if x.shape != z.shape or x.shape[1] == 0:
        raise ValueError("check_x/check_z must have equal shapes and positive width")
    m, n = x.shape
    signs = (np.zeros(m, np.uint8) if check_signs is None else
             binary_vector(check_signs, length=m, name="check_signs"))
    tx = binary_matrix(target_x, n_cols=n, name="target_x")
    tz = binary_matrix(target_z, n_cols=n, name="target_z")
    if tx.shape != tz.shape:
        raise ValueError("target_x/target_z must have equal shapes")
    ts = (np.zeros(len(tx), np.uint8) if target_signs is None else
          binary_vector(target_signs, length=len(tx), name="target_signs"))
    old_rank = _signed_check_rank(x, z, signs)
    _validate_targets(x, z, tx, tz)
    plans = [_graph_plan(x, z, xx, zz) for xx, zz in zip(tx, tz)]
    new_n = n + sum(len(p["edges"]) for p in plans)
    new_m = m + sum(len(p["support"]) + p["cycle_count"] for p in plans)
    graph_entries = sum(m * len(p["support"]) + m * len(p["edges"])
                        + len(p["edges"]) * len(p["support"])
                        + p["cycle_count"] * len(p["edges"]) for p in plans)
    entries = 2 * new_m * new_n + new_m + graph_entries + len(tx) * (m + n)
    if entries > int(max_entries):
        raise ValueError(f"Direct attachment needs {entries} dense entries, above max_entries={max_entries}")
    outcomes, independent = _sector(x, z, signs, tx, tz, ts)
    ox, oz = np.zeros((new_m, new_n), np.uint8), np.zeros((new_m, new_n), np.uint8)
    os = np.zeros(new_m, np.uint8)
    ox[:m, :n], oz[:m, :n], os[:m] = x, z, signs
    target_row_growth = np.zeros((len(tx), m), np.int64)
    target_column_growth = (tx | tz).astype(np.int64)
    graphs, col, row = [], n, m
    for i, plan in enumerate(plans):
        support, edges = plan["support"], plan["edges"]
        w, e = len(support), len(edges)
        b = np.zeros((e, w), np.uint8)
        for j, (u, v) in enumerate(edges):
            b[j, [u, v]] = 1
        f = (x[:, support] * tz[i, support]) ^ (z[:, support] * tx[i, support])
        matching = np.zeros((m, e), np.uint8)
        for j, selected in enumerate(plan["row_edges"]):
            matching[j, selected] = 1
        cycles = _cycle_basis(edges, w)
        edge_columns = np.arange(col, col + e)
        vertex_rows = np.arange(row, row + w)
        cycle_rows = np.arange(row + w, row + w + len(cycles))
        ox[:m, col:col + e] = matching
        ox[vertex_rows, support] = tx[i, support]
        oz[vertex_rows, support] = tz[i, support]
        oz[row:row + w, col:col + e] = b.T
        ox[row + w:row + w + len(cycles), col:col + e] = cycles
        os[row] = ts[i] ^ outcomes[i]
        target_row_growth[i] = np.sum(matching, axis=1)
        graphs.append(dict(target_index=i, support=support, edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                           edge_owners=np.asarray(plan["edge_owners"], dtype=np.int64),
                           B=b, F=f, M=matching, cycle_basis=cycles,
                           edge_columns=edge_columns, vertex_rows=vertex_rows, cycle_rows=cycle_rows,
                           connecting_edges=np.asarray(plan["connecting_edges"], dtype=np.int64),
                           components_before_connecting=plan["components_before_connecting"]))
        col += e
        row += w + len(cycles)
    new_rank = _signed_check_rank(ox, oz, os)
    logical_rank = len(independent)
    if new_n - new_rank != n - old_rank - logical_rank:
        raise ValueError("Direct preview changed more logical information than requested")
    old_row_weights = np.sum(x | z, axis=1, dtype=np.int64)
    old_column_weights = np.sum(x | z, axis=0, dtype=np.int64)
    row_weights = np.sum(ox | oz, axis=1, dtype=np.int64)
    column_weights = np.sum(ox | oz, axis=0, dtype=np.int64)
    return dict(kind="direct_paired_gauging_preview", check_x=ox, check_z=oz, check_signs=os,
                old_n=n, old_checks=m, n=new_n, k=new_n - new_rank,
                graphs=graphs, matchings=[g["M"] for g in graphs],
                target_x=tx, target_z=tz, target_signs=ts, target_outcome_signs=outcomes,
                independent_target_indices=np.asarray(independent, dtype=np.int64),
                target_logical_rank=logical_rank,
                target_memory_row_growth=target_row_growth,
                target_memory_column_growth=target_column_growth,
                memory_row_growth=row_weights[:m] - old_row_weights,
                memory_column_growth=column_weights[:n] - old_column_weights,
                original_row_weights=old_row_weights, original_column_weights=old_column_weights,
                row_weights=row_weights, column_weights=column_weights,
                outcome_sector="arbitrary_consistent_preview_assignment",
                distance_certificate=None, expansion_certificate=None, dense_entries=entries)


__all__ = ["direct_loads", "preview_direct_attachment"]
