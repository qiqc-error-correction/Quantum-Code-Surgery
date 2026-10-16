"""Ordinary multigraph incidence matrices; no implicit expansion repair."""
from collections import deque
import numpy as np
from ..gf2 import binary_matrix, nullspace, rank


def budget_check(max_entries, *sizes):
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    if any(int(size) > max_entries for size in sizes):
        raise ValueError("Dense graph construction exceeds max_entries")


def adjacency(b):
    adj = [[] for _ in range(b.shape[1])]
    for e, row in enumerate(b):
        u, v = map(int, np.flatnonzero(row))
        adj[u].append((v, e))
        adj[v].append((u, e))
    return adj


def tree_data(b, root=0):
    adj = adjacency(b)
    parent = [-1] * len(adj)
    edge = [-1] * len(adj)
    children = [[] for _ in adj]
    order = [root]
    parent[root] = root
    for u in order:
        for v, e in adj[u]:
            if parent[v] == -1:
                parent[v], edge[v] = u, e
                children[u].append(v)
                order.append(v)
    if len(order) != len(adj):
        raise ValueError("Auxiliary graph must be connected")
    return parent, edge, children, order


def graph_matrix(value, *, max_entries=50_000_000):
    a = np.asarray(value)
    budget_check(max_entries, a.size)
    b = binary_matrix(value, name="graph incidence")
    if b.shape[1] < 1:
        raise ValueError("Graph must have at least one vertex; use shape (0,1) for one isolated vertex")
    if np.any(b.sum(axis=1) != 2):
        raise ValueError("Every graph incidence row must have weight exactly 2 (no loops or hyperedges)")
    tree_data(b)
    return b


def tree_path(u, v, parent, edges):
    ancestors = set()
    p = u
    while True:
        ancestors.add(p)
        if parent[p] == p:
            break
        p = parent[p]
    path = []
    p = v
    while p not in ancestors:
        path.append(edges[p])
        p = parent[p]
    lca = p
    p = u
    while p != lca:
        path.append(edges[p])
        p = parent[p]
    return path


def cycle_basis(b, supplied=None, *, max_entries=50_000_000):
    b = graph_matrix(b, max_entries=max_entries)
    m, n = b.shape
    budget_check(max_entries, m * m)
    if supplied is None:
        parent, edge, _, _ = tree_data(b)
        tree_edges = set(edge[1:])
        others = [e for e in range(m) if e not in tree_edges]
        cycles = np.zeros((len(others), m), dtype=np.uint8)
        for j, e in enumerate(others):
            u, v = map(int, np.flatnonzero(b[e]))
            cycles[j, e] = 1
            cycles[j, tree_path(u, v, parent, edge)] ^= 1
    else:
        budget_check(max_entries, np.asarray(supplied).size)
        cycles = binary_matrix(supplied, n_cols=m, name="cycle_basis")
    if np.any((cycles @ b) % 2) or rank(cycles) != m - n + 1:
        raise ValueError("cycle_basis must span the complete graph cycle space")
    return cycles


def relative_expansion(b, ports, *, distance=None, exact_limit=20, max_entries=50_000_000):
    """Definition 2 of arXiv:2410.03628: beta_t relative to a port subset.

    Complement symmetry permits enumeration up to half the graph vertices,
    while the denominator counts port vertices. Never sample or add edges.
    """
    b = graph_matrix(b, max_entries=max_entries)
    n = b.shape[1]
    ports = list(ports)
    if not ports or len(set(ports)) != len(ports) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 or v >= n for v in ports):
        raise ValueError("ports must be distinct graph vertex indices")
    t = len(ports) if distance is None else distance
    if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("distance must be a positive integer")
    if isinstance(exact_limit, bool) or not isinstance(exact_limit, (int, np.integer)) or not 1 <= exact_limit <= 24:
        raise ValueError("exact_limit must be an integer in [1,24]")
    report = dict(parameter_t=int(t), port_vertices=ports, exact=False, beta=None,
                  lower_bound=0.0, condition_met=False, certificate="not enumerated")
    if len(ports) == 1:
        report.update(exact=True, lower_bound=1.0, condition_met=True,
                      certificate="vacuous one-port relative expansion")
        return report
    if n > exact_limit or 12 * (1 << n) > max_entries:
        if min(t, len(ports) // 2) == 1:
            report.update(lower_bound=1.0, condition_met=True,
                          certificate="connected graph and relative denominator at most one")
        return report
    # This helper enumerates all eligible masks and preserves parallel edges.
    from ..explicit_gadgets.expansion import _enumerate_boundaries
    masks, _, boundaries = _enumerate_boundaries(b, max_entries=max_entries)
    inside = np.zeros(len(masks), dtype=np.uint8)
    for vertex in ports:
        inside += ((masks >> vertex) & 1).astype(np.uint8)
    denominator = np.minimum(int(t), np.minimum(inside, len(ports) - inside))
    valid = denominator > 0
    best_num, best_den, best_mask = None, 1, 0
    for den in np.unique(denominator[valid]):
        indices = np.flatnonzero(denominator == den)
        idx = int(indices[np.argmin(boundaries[indices])])
        num = int(boundaries[idx])
        if best_num is None or num * best_den < best_num * int(den):
            best_num, best_den, best_mask = num, int(den), int(masks[idx])
    report.update(exact=True, beta=best_num / best_den, lower_bound=best_num / best_den,
                  condition_met=best_num >= best_den, certificate="exhaustive relative parity-boundary enumeration",
                  worst_cut=[v for v in range(n) if (best_mask >> v) & 1],
                  boundary_weight=best_num, denominator=best_den, eligible_cut_count=int(valid.sum()))
    return report
