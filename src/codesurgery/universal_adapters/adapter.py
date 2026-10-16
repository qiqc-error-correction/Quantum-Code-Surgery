"""Algorithm 1 (SkipTree) and the repetition adapter of arXiv:2410.03628.

The cyclic connector cycle generators are retained as [T_l H_C T_r].
They are not replaced by an arbitrary nullspace basis.
"""
import numpy as np
from ..gf2 import rank
from .graphs import graph_matrix, cycle_basis, tree_data, tree_path, budget_check


def skip_tree(incidence, *, max_entries=50_000_000):
    b = graph_matrix(incidence, max_entries=max_entries)
    m, n = b.shape
    budget_check(max_entries, n * m, n * n)
    parent, edge, children, _ = tree_data(b)
    # Iterative version of mutually recursive LabelFirst / LabelLast.
    order = []
    stack = [(0, True, False)]
    while stack:
        v, first, emit = stack.pop()
        if emit:
            order.append(v)
            continue
        if first:
            order.append(v)
        else:
            stack.append((v, first, True))
        for child in reversed(children[v]):
            stack.append((child, not first, False))
    p = np.zeros((n, n), dtype=np.uint8)
    p[order, np.arange(n)] = 1
    h = np.eye(n, dtype=np.uint8) ^ np.roll(np.eye(n, dtype=np.uint8), 1, axis=1)
    t = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        t[i, tree_path(order[i], order[(i + 1) % n], parent, edge)] = 1
    if not np.array_equal((t @ b @ p) % 2, h):
        raise RuntimeError("SkipTree T F P = H_C invariant failed")
    if np.any(t.sum(axis=1) > 3) or np.any(t.sum(axis=0) > 2):
        raise RuntimeError("SkipTree (3,2)-sparsity invariant failed")
    return dict(T=t, P=p, H=h, order=order)


def _embedded_skip(b, size, vertices, max_entries):
    if vertices is None:
        vertices = tree_data(b)[3][:size]  # Connected BFS prefix.
    else:
        vertices = list(vertices)
        if len(vertices) != size or len(set(vertices)) != size or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v < b.shape[1] for v in vertices):
            raise ValueError("Adapter vertices must be distinct valid indices of adapter_size")
    rows = np.flatnonzero(b[:, vertices].sum(axis=1) == 2)
    sub = b[rows][:, vertices]
    small = skip_tree(sub, max_entries=max_entries)
    t = np.zeros((size, len(b)), dtype=np.uint8)
    t[:, rows] = small["T"]
    p = np.zeros((b.shape[1], size), dtype=np.uint8)
    p[vertices] = small["P"]
    if not np.array_equal((t @ b) % 2, (small["H"] @ p.T) % 2):
        raise RuntimeError("Embedded SkipTree invariant failed")
    return t, p, small["H"], vertices


def repetition_adapter(left, right, *, left_cycles=None, right_cycles=None,
                       adapter_size=None, left_vertices=None, right_vertices=None,
                       adapter_shift=0, max_entries=50_000_000):
    """Join two graphs, optionally rotating the right cyclic port labels.

    ``adapter_shift`` changes the cyclic origin of the right SkipTree output.
    It rotates T rows and P columns together, preserving T F P = H_C and
    SkipTree's (3,2)-sparsity. The connector generators remain [T_l H_C T_r];
    this option does not substitute a different cycle basis.
    """
    left, right = graph_matrix(left, max_entries=max_entries), graph_matrix(right, max_entries=max_entries)
    ml, nl = left.shape
    mr, nr = right.shape
    size = min(nl, nr) if adapter_size is None else adapter_size
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or not 1 <= size <= min(nl, nr):
        raise ValueError("adapter_size must be between 1 and min(vertex counts)")
    size = int(size)
    if (isinstance(adapter_shift, (bool, np.bool_))
            or not isinstance(adapter_shift, (int, np.integer))
            or not 0 <= adapter_shift < size):
        raise ValueError("adapter_shift must be an integer in [0,adapter_size)")
    adapter_shift = int(adapter_shift)
    m = ml + size + mr
    budget_check(max_entries, m * (nl + nr), m * m, size * max(ml, mr, nl, nr))
    cl = cycle_basis(left, left_cycles, max_entries=max_entries)
    cr = cycle_basis(right, right_cycles, max_entries=max_entries)
    tl, pl, hc, vl = _embedded_skip(left, size, left_vertices, max_entries)
    tr, pr, _, vr = _embedded_skip(right, size, right_vertices, max_entries)
    tr = np.roll(tr, adapter_shift, axis=0)
    pr = np.roll(pr, adapter_shift, axis=1)
    if not np.array_equal((tr @ right @ pr) % 2, hc):
        raise RuntimeError("Rotated right SkipTree T F P = H_C invariant failed")
    b = np.zeros((m, nl + nr), dtype=np.uint8)
    b[:ml, :nl] = left
    b[ml:ml + size, :nl], b[ml:ml + size, nl:] = pl.T, pr.T
    b[ml + size:, nl:] = right
    connector_count = size if size > 1 else 0  # H_C for A=1 is a zero/identity check.
    budget_check(max_entries, (len(cl) + connector_count + len(cr)) * m)
    cycles = np.zeros((len(cl) + connector_count + len(cr), m), dtype=np.uint8)
    cycles[:len(cl), :ml] = cl
    middle = slice(len(cl), len(cl) + connector_count)
    if connector_count:
        cycles[middle, :ml], cycles[middle, ml:ml + size], cycles[middle, ml + size:] = tl, hc, tr
    cycles[len(cl) + connector_count:, ml + size:] = cr
    if np.any((cycles @ b) % 2) or rank(cycles) != m - nl - nr + 1:
        raise RuntimeError("Adapter cycle space invariant failed")
    if np.any(cycles[middle].sum(axis=1) > 8):
        raise RuntimeError("Adapter introduced a cycle heavier than eight")
    return dict(incidence=b, cycle_basis=cycles, T_left=tl, T_right=tr, P_left=pl,
                P_right=pr, H_C=hc, adapter_size=size, vertices_left=vl, vertices_right=vr,
                adapter_shift=adapter_shift,
                matching=[[int(np.argmax(pl[:, i])), int(np.argmax(pr[:, i]))] for i in range(size)],
                edge_layout=dict(left=list(range(ml)), adapter=list(range(ml, ml + size)),
                                 right=list(range(ml + size, m))),
                connector_cycle_rows=list(range(len(cl), len(cl) + connector_count)),
                connector_max_weight=int(cycles[middle].sum(axis=1).max(initial=0)),
                source="Swaroop--Jochym-O'Connor--Yoder 2024, SkipTree and repetition adapter")
