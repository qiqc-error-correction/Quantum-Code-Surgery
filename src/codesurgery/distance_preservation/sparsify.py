"""Finite-size cycle decongestion and cellulation for ordinary multigraphs.

This implements the graph-product and cellulation idea of Sec. 2.3 of
arXiv:2410.03628v1.  The deterministic vertex-conflict coloring used here is
deliberately simpler than Freedman--Hastings decongestion: it gives explicit
sparsity bounds, but does not claim the paper's polylogarithmic overhead.

Original vertices and edges remain the first vertices/edges of layer zero.
Thus a verified relative-expansion bound at ports in that layer is preserved:
every final cut already contains the boundary of its layer-zero restriction.
This function does not itself certify the expansion of the input graph.
"""
from collections import deque

import numpy as np

from ..gf2 import binary_matrix, rank
from ..universal_adapters.graphs import budget_check, cycle_basis, graph_matrix
from .cycles import optimization_options, optimize_cycles


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _maximum(values):
    return int(np.max(values)) if np.size(values) else 0


def _metrics(b, cycles):
    return {
        "vertices": int(b.shape[1]),
        "edges": int(b.shape[0]),
        "cycle_generators": int(cycles.shape[0]),
        "cycle_rank": int(b.shape[0] - b.shape[1] + 1),
        "max_degree": _maximum(b.sum(axis=0)),
        "max_cycle_weight": _maximum(cycles.sum(axis=1)),
        "max_cycle_congestion": _maximum(cycles.sum(axis=0)),
    }


def _ordered_cycle(b, vector):
    """Order a simple fundamental cycle, retaining parallel-edge identities."""
    edge_ids = list(map(int, np.flatnonzero(vector)))
    adjacency = {}
    for e in edge_ids:
        u, v = map(int, np.flatnonzero(b[e]))
        adjacency.setdefault(u, []).append((v, e))
        adjacency.setdefault(v, []).append((u, e))
    if not edge_ids or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("A fundamental cycle must be a simple cycle, including a parallel-edge digon")
    start = min(adjacency)
    current, used, vertices, ordered_edges = start, set(), [], []
    while True:
        vertices.append(current)
        candidates = [(e, v) for v, e in adjacency[current] if e not in used]
        if not candidates:
            raise RuntimeError("Incomplete fundamental cycle traversal")
        e, following = min(candidates)
        used.add(e)
        ordered_edges.append(e)
        current = following
        if current == start:
            break
        if current in vertices:
            raise RuntimeError("A fundamental cycle revisited a vertex")
    if len(used) != len(edge_ids) or len(vertices) != len(edge_ids):
        raise RuntimeError("A fundamental cycle has multiple components")
    return vertices, ordered_edges


def _color_cycles(ordered_cycles, n_vertices):
    """Greedy coloring: cycles sharing ANY vertex must occupy different layers."""
    used_at_vertex = [set() for _ in range(n_vertices)]
    colors = [0] * len(ordered_cycles)
    # Largest supports first; original indices break ties and remain in reports.
    order = sorted(range(len(ordered_cycles)), key=lambda i: (-len(ordered_cycles[i][0]), i))
    for i in order:
        forbidden = set()
        for v in ordered_cycles[i][0]:
            forbidden.update(used_at_vertex[v])
        layer = 0
        while layer in forbidden:
            layer += 1
        colors[i] = layer
        for v in ordered_cycles[i][0]:
            used_at_vertex[v].add(layer)
    return colors


def _triangulate(vertices, boundary_edges, endpoints):
    """Alternating ears add at most two fresh chords per polygon vertex.

    A fresh edge is allocated even if the endpoint pair is already adjacent.
    This permits independent cycle replacement in a multigraph without
    increasing the incidence of an existing edge in multiple cellulations.
    """
    remaining = deque(vertices)
    edges = deque(boundary_edges)
    triangles, remove_left = [], True
    while len(remaining) > 3:
        if remove_left:
            chord = len(endpoints)
            endpoints.append((remaining[-1], remaining[1]))
            triangles.append([edges[-1], edges[0], chord])
            remaining.popleft()
            edges.popleft()
            edges.pop()
            edges.append(chord)
        else:
            chord = len(endpoints)
            endpoints.append((remaining[-2], remaining[0]))
            triangles.append([edges[-2], edges[-1], chord])
            remaining.pop()
            edges.pop()
            edges.pop()
            edges.append(chord)
        remove_left = not remove_left
    triangles.append(list(edges))
    return triangles


def sparsify_cycles(
    b, *, matching=None, cycles=None, max_cycle_weight=8,
    max_cycle_congestion=4, max_layers=256, max_entries=50_000_000,
    optimize_cycle_basis=True, cycle_basis_max_roots=64,
    cycle_basis_max_candidates=10_000,
):
    """Return ``(incidence, matching, cycle_generators, report)``.

    An already sufficiently sparse complete cycle generating set is kept.
    Otherwise, first try a bounded multi-root BFS cycle-basis search on the
    unchanged graph. Only a complete equivalent basis satisfying both bounds
    is accepted. Set optimize_cycle_basis=False for the original fallback.
    If no acceptable basis is found, a simple fundamental basis is colored by vertex conflicts,
    lifted to separate layers of ``B square P_L``, and supplemented by every
    interlayer square.  Lifted cycles longer than ``max_cycle_weight`` are
    triangulated by alternating ears; shorter cycles (including digons) stay.

    ``max_cycle_congestion`` is the trigger/desired threshold.  After the
    transformation the rigorous guaranteed bound is ``max(3, degree(B))``;
    the report explicitly records whether the desired threshold was attained.
    ``matching=None`` returns ``None``; otherwise the matching is extended by
    zero columns, preserving its boundary and both its row/column weights.
    """
    max_cycle_weight = _integer(max_cycle_weight, "max_cycle_weight", 4)
    max_cycle_congestion = _integer(max_cycle_congestion, "max_cycle_congestion", 1)
    max_layers = _integer(max_layers, "max_layers", 1)
    optimize_cycle_basis, cycle_basis_max_roots, cycle_basis_max_candidates = optimization_options(
        optimize_cycle_basis, cycle_basis_max_roots, cycle_basis_max_candidates)
    b = graph_matrix(b, max_entries=max_entries)
    m, n = b.shape
    if cycles is not None:
        budget_check(max_entries, np.asarray(cycles).size)
        raw_cycles = np.asarray(cycles)
        if raw_cycles.ndim == 2:
            budget_check(max_entries, raw_cycles.shape[0] * n)
    original_cycles = cycle_basis(b, cycles, max_entries=max_entries)
    if matching is not None:
        budget_check(max_entries, np.asarray(matching).size)
        matching = binary_matrix(matching, n_cols=m, name="matching")
        budget_check(max_entries, matching.shape[0] * n)
    before = _metrics(b, original_cycles)
    optimized_cycles, basis_report = optimize_cycles(b, original_cycles,
        max_cycle_weight=max_cycle_weight, max_cycle_congestion=max_cycle_congestion,
        enabled=optimize_cycle_basis, max_roots=cycle_basis_max_roots,
        max_candidates=cycle_basis_max_candidates, max_entries=max_entries)
    report = {
        "applied": False,
        "method": "existing complete cycle generators meet requested bounds",
        "reference": "https://arxiv.org/html/2410.03628v1#S2.SS3",
        "overhead_guarantee": "finite-size deterministic construction; no Freedman-Hastings polylogarithmic overhead claim",
        "base_vertices": n,
        "base_edges": m,
        "layers": 1,
        "base_prefix_preserved": True,
        "matching_supported_on_layer_zero": True,
        "relative_expansion_effect": "preserves any verified bound for ports contained in original layer zero; input bound must be checked separately",
        "requested_max_cycle_weight": max_cycle_weight,
        "requested_max_cycle_congestion": max_cycle_congestion,
        "before": before,
        "cycle_assignments": [],
        "interlayer_square_count": 0,
        "added_chord_count": 0,
        "triangle_count": 0,
        "cycle_basis_optimization": basis_report,
    }
    optimized_metrics = _metrics(b, optimized_cycles)
    if optimized_metrics["max_cycle_weight"] <= max_cycle_weight and optimized_metrics["max_cycle_congestion"] <= max_cycle_congestion:
        report.update(after=optimized_metrics, requested_cycle_weight_met=True, requested_cycle_congestion_met=True)
        if basis_report['changed']:
            report['method'] = 'equivalent cycle-basis optimization; graph unchanged'
        return b, matching, optimized_cycles, report

    # A user-supplied basis may contain non-simple or redundant generators.
    # Validate it above, then choose a simple independent basis for coloring.
    fundamental = cycle_basis(b, max_entries=max_entries)
    ordered = [_ordered_cycle(b, row) for row in fundamental]
    colors = _color_cycles(ordered, n)
    layers = max(colors, default=0) + 1
    if layers > max_layers:
        raise ValueError(f"Cycle decongestion needs {layers} layers, exceeding max_layers={max_layers}")
    chord_count = sum(len(vertices) - 3 for vertices, _ in ordered if len(vertices) > max_cycle_weight)
    horizontal_count = layers * m
    vertical_count = (layers - 1) * n
    total_vertices = layers * n
    total_edges = horizontal_count + vertical_count + chord_count
    total_cycles = (layers - 1) * m + len(fundamental) + chord_count
    budget_check(max_entries, total_edges * total_vertices, total_cycles * total_edges,
                 total_cycles * total_vertices,
                 0 if matching is None else matching.shape[0] * total_edges,
                 0 if matching is None else matching.shape[0] * total_vertices)

    base_endpoints = [tuple(map(int, np.flatnonzero(row))) for row in b]
    endpoints = [(u + layer * n, v + layer * n)
                 for layer in range(layers) for u, v in base_endpoints]
    endpoints.extend((layer * n + v, (layer + 1) * n + v)
                     for layer in range(layers - 1) for v in range(n))
    cycle_edges = []
    for layer in range(layers - 1):
        for e, (u, v) in enumerate(base_endpoints):
            cycle_edges.append([layer * m + e, (layer + 1) * m + e,
                                horizontal_count + layer * n + u,
                                horizontal_count + layer * n + v])
    triangle_count = 0
    assignments = []
    for i, ((vertices, edges), layer) in enumerate(zip(ordered, colors)):
        lifted_vertices = [v + layer * n for v in vertices]
        lifted_edges = [e + layer * m for e in edges]
        triangulated = len(vertices) > max_cycle_weight
        assignments.append({"basis_cycle": i, "layer": layer,
                            "base_vertices": vertices, "base_edge_ids": edges,
                            "triangulated": triangulated})
        if triangulated:
            triangles = _triangulate(lifted_vertices, lifted_edges, endpoints)
            triangle_count += len(triangles)
            cycle_edges.extend(triangles)
        else:
            cycle_edges.append(lifted_edges)
    if len(endpoints) != total_edges or len(cycle_edges) != total_cycles:
        raise RuntimeError("Unexpected cellulation edge/cycle count")
    final_b = np.zeros((total_edges, total_vertices), dtype=np.uint8)
    for e, pair in enumerate(endpoints):
        final_b[e, list(pair)] = 1
    final_cycles = np.zeros((total_cycles, total_edges), dtype=np.uint8)
    for row, edge_ids in enumerate(cycle_edges):
        final_cycles[row, edge_ids] = 1
    final_matching = None
    if matching is not None:
        final_matching = np.zeros((len(matching), total_edges), dtype=np.uint8)
        final_matching[:, :m] = matching

    # Independent algebra checks guard both topology and edge indexing.
    graph_matrix(final_b, max_entries=max_entries)
    if np.any((final_cycles @ final_b) & 1) or rank(final_cycles) != total_edges - total_vertices + 1:
        raise RuntimeError("Cellulated generators do not span the complete graph cycle space")
    if not np.array_equal(final_b[:m, :n], b) or np.any(final_b[:m, n:]):
        raise RuntimeError("Original layer-zero graph was not preserved")
    if final_matching is not None:
        boundary = (final_matching @ final_b) & 1
        if not np.array_equal(boundary[:, :n], (matching @ b) & 1) or np.any(boundary[:, n:]):
            raise RuntimeError("Decongestion changed the original matching boundary")
    after = _metrics(final_b, final_cycles)
    bounds = {"max_degree": before["max_degree"] + 4,
              "max_cycle_weight": max_cycle_weight,
              "max_cycle_congestion": max(3, before["max_degree"])}
    if any(after[key] > value for key, value in bounds.items()):
        raise RuntimeError("Cycle decongestion violated a proved sparsity bound")
    report.update(applied=True, method="vertex-disjoint cycle layers and alternating-ear cellulation",
                  layers=layers, horizontal_edge_count=horizontal_count,
                  vertical_edge_count=vertical_count, cycle_assignments=assignments,
                  interlayer_square_count=(layers - 1) * m,
                  added_chord_count=chord_count, triangle_count=triangle_count,
                  after=after, theorem_bounds=bounds,
                  requested_cycle_weight_met=after["max_cycle_weight"] <= max_cycle_weight,
                  requested_cycle_congestion_met=after["max_cycle_congestion"] <= max_cycle_congestion)
    return final_b, final_matching, final_cycles, report
