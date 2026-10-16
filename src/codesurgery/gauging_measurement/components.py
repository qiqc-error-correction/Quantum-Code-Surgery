"""Connect paired-boundary components with repetition-code adapters.

Every original component is retained, including components whose aligned
Pauli is a stabilizer and components with dependent logical signatures.
Their separate readouts are joined into the single requested product.
"""

import numpy as np

from ..gf2 import rank
from ..gadget_common import restriction, signature
from ..universal_adapters import repetition_adapter
from ..universal_adapters.graphs import (
    adjacency, budget_check, cycle_basis, graph_matrix, relative_expansion,
    tree_data,
)


def _adapter_options(adapter_size, adapter_shift):
    if adapter_size is not None and (
            isinstance(adapter_size, (bool, np.bool_))
            or not isinstance(adapter_size, (int, np.integer))
            or adapter_size < 1):
        raise ValueError("component adapter_size must be a positive integer")
    if adapter_shift is None:
        return
    if (isinstance(adapter_shift, (bool, np.bool_))
            or not isinstance(adapter_shift, (int, np.integer))
            or adapter_shift < 0):
        raise ValueError("component adapter_shift must be a nonnegative integer")
    if adapter_size is not None and adapter_shift >= adapter_size:
        raise ValueError("component adapter_shift must be smaller than adapter_size")


def paired_component_graph(code, target, basis, *, adapter_size=None,
                           adapter_shift=None, exact_limit=20,
                           max_entries=50_000_000):
    """Return ``(B, M, C, report)`` for a prepared, signed Pauli target.

    ``B`` has one column per sorted physical support qubit. ``M B`` equals
    the full memory-check anticommutation boundary, and ``C`` retains the
    original fundamental cycles and the actual SkipTree connector rows.
    Each link uses only its two adjacent *original* components, preventing
    later links from repeatedly attaching to the first component.

    The default adapter size is the smaller adjacent component size. An
    explicit size/shift must be valid for every link. ``adapter_shift=None``
    picks the cyclic rotation with the smallest largest connector weight,
    breaking ties by shift. The helper never adds
    dummy vertices or silently increases graph size. It does not assert a
    distance bound. ``verify_component_adapter_expansion`` can independently
    check the compositional expansion hypotheses on the returned graph.
    """
    _adapter_options(adapter_size, adapter_shift)
    if (isinstance(exact_limit, (bool, np.bool_))
            or not isinstance(exact_limit, (int, np.integer))
            or not 1 <= exact_limit <= 24):
        raise ValueError("exact_limit must be an integer in [1,24]")
    r = len(code.hx) + len(code.hz)
    budget_check(max_entries, 2 * r * code.n)
    support, raw, _, touching_rows = restriction(code, target)
    w = len(support)
    if w == 0:
        raise ValueError("Cannot build component adapters for the identity")
    if np.any(raw.sum(axis=1) % 2):
        raise ValueError("Every memory boundary must have even weight")
    edges, owners = [], []
    for boundary, owner in zip(raw, touching_rows):
        vertices = np.flatnonzero(boundary).tolist()
        for j in range(0, len(vertices), 2):
            edges.append((vertices[j], vertices[j + 1]))
            owners.append(int(owner))
    e = len(edges)
    budget_check(max_entries, e * w, r * e, w * w)
    original_graph = np.zeros((e, w), dtype=np.uint8)
    original_matching = np.zeros((r, e), dtype=np.uint8)
    for edge, ((u, v), owner) in enumerate(zip(edges, owners)):
        original_graph[edge, [u, v]] = 1
        original_matching[owner, edge] = 1
    full_boundary = np.zeros((r, w), dtype=np.uint8)
    full_boundary[touching_rows] = raw
    if not np.array_equal((original_matching @ original_graph) % 2, full_boundary):
        raise RuntimeError("Paired component matching boundary invariant failed")

    adj = adjacency(original_graph)
    seen, vertex_groups = set(), []
    for root in range(w):
        if root in seen:
            continue
        queue = [root]
        seen.add(root)
        for u in queue:
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        vertex_groups.append(sorted(queue))

    count = len(vertex_groups)
    budget_check(max_entries, 2 * count * code.n, 2 * count * code.k)
    qx = np.zeros((count, code.n), dtype=np.uint8)
    qz = np.zeros_like(qx)
    components = []
    for index, vertices in enumerate(vertex_groups):
        physical = support[vertices]
        qx[index, physical] = target[0][physical]
        qz[index, physical] = target[1][physical]
        edge_rows = np.flatnonzero(original_graph[:, vertices].sum(axis=1) == 2)
        graph = original_graph[edge_rows][:, vertices]
        cycles = cycle_basis(graph, max_entries=max_entries)
        components.append(dict(component_index=index, vertices=vertices,
            physical_support=physical.tolist(), graph=graph, cycle_basis=cycles,
            original_edge_rows=edge_rows.tolist(),
            sign=int(target[2]) if index == 0 else 0))
    if np.any((qx @ code.hz.T) % 2) or np.any((qz @ code.hx.T) % 2):
        raise RuntimeError("A paired-boundary component is not in the memory normalizer")
    gamma = signature(qx, qz, basis)
    target_signature = signature(target[0], target[1], basis)
    if (not np.array_equal(np.bitwise_xor.reduce(qx, axis=0), target[0])
            or not np.array_equal(np.bitwise_xor.reduce(qz, axis=0), target[1])
            or not np.array_equal(np.bitwise_xor.reduce(gamma, axis=0), target_signature)):
        raise RuntimeError("Component product does not reproduce the requested target")
    for index, component in enumerate(components):
        component.update(signature=gamma[index].copy(),
                         kind="logical" if np.any(gamma[index]) else "stabilizer")

    graph = components[0]["graph"].copy()
    cycles = components[0]["cycle_basis"].copy()
    matching = original_matching[:, components[0]["original_edge_rows"]].copy()
    vertex_order = list(vertex_groups[0])
    original_edge_to_final = np.full(e, -1, dtype=np.int64)
    first_rows = components[0]["original_edge_rows"]
    original_edge_to_final[first_rows] = np.arange(len(first_rows))
    components[0]["final_edge_rows"] = list(range(len(first_rows)))
    adapters = []
    previous_offset = 0
    for index in range(1, count):
        previous, current = components[index - 1], components[index]
        size = min(len(previous["vertices"]), len(current["vertices"])) if adapter_size is None else int(adapter_size)
        if size > min(len(previous["vertices"]), len(current["vertices"])):
            raise ValueError(f"component adapter_size={size} exceeds adjacent component sizes at link {index - 1}->{index}")
        if adapter_shift is not None and adapter_shift >= size:
            raise ValueError(f"component adapter_shift must be in [0,{size}) at link {index - 1}->{index}")
        ml, nl = graph.shape
        mr = len(current["graph"])
        left_vertices = [previous_offset + v for v in tree_data(previous["graph"])[3][:size]]
        right_vertices = tree_data(current["graph"])[3][:size]
        chosen_shift = 0 if adapter_shift is None else int(adapter_shift)
        adapter_options = dict(left_cycles=cycles,
            right_cycles=current["cycle_basis"], adapter_size=size,
            left_vertices=left_vertices, right_vertices=right_vertices,
            max_entries=max_entries)
        adapter = repetition_adapter(graph, current["graph"],
            adapter_shift=chosen_shift, **adapter_options)
        if adapter_shift is None:
            # Rotation changes only how the fixed left/right path weights
            # meet in a row. Score the short vectors, not whole candidate
            # gadgets, then rebuild at most once with the selected shift.
            fixed_weights = adapter["T_left"].sum(axis=1) + adapter["H_C"].sum(axis=1)
            right_weights = adapter["T_right"].sum(axis=1)
            chosen_shift = min(range(size), key=lambda shift:
                (int((fixed_weights + np.roll(right_weights, shift)).max(initial=0)), shift))
            if chosen_shift:
                adapter = repetition_adapter(graph, current["graph"],
                    adapter_shift=chosen_shift, **adapter_options)
        new_edge_count = ml + size + mr
        budget_check(max_entries, r * new_edge_count)
        next_matching = np.zeros((r, new_edge_count), dtype=np.uint8)
        next_matching[:, :ml] = matching
        next_matching[:, ml + size:] = original_matching[:, current["original_edge_rows"]]
        final_rows = list(range(ml + size, new_edge_count))
        original_edge_to_final[current["original_edge_rows"]] = final_rows
        current["final_edge_rows"] = final_rows
        adapters.append(dict(left_component=index - 1, right_component=index,
            adapter_size=size, adapter_shift=chosen_shift,
            shift_selection="automatic_min_cycle_weight" if adapter_shift is None else "explicit",
            edge_rows=adapter["edge_layout"]["adapter"],
            cycle_rows=adapter["connector_cycle_rows"],
            T_left=adapter["T_left"], T_right=adapter["T_right"],
            P_left=adapter["P_left"], P_right=adapter["P_right"], H_C=adapter["H_C"],
            left_vertex_order=list(vertex_order), right_vertex_order=list(current["vertices"]),
            left_edge_rows=list(range(ml)), right_edge_rows=final_rows,
            vertices_left=[vertex_order[v] for v in adapter["vertices_left"]],
            vertices_right=[current["vertices"][v] for v in adapter["vertices_right"]],
            connector_max_weight=adapter["connector_max_weight"],
            source=adapter["source"]))
        previous_offset = nl
        vertex_order.extend(current["vertices"])
        graph, matching, cycles = adapter["incidence"], next_matching, adapter["cycle_basis"]

    # Adapter composition groups columns by component. Restore the same sorted
    # physical-support convention as build_graph's default port selector.
    graph = graph[:, np.argsort(vertex_order)]
    if (np.any(original_edge_to_final < 0)
            or not np.array_equal(graph[original_edge_to_final], original_graph)
            or not np.array_equal((matching @ graph) % 2, full_boundary)
            or np.any((cycles @ graph) % 2)
            or rank(graph) != w - 1
            or rank(cycles) != len(graph) - w + 1):
        raise RuntimeError("Component adapter composition invariant failed")
    report = dict(method="paired check boundaries with repetition-code component adapters",
        component_count=count, components=components,
        component_signature_matrix=gamma, initial_logical_rank=rank(gamma),
        target_signature=target_signature, target_sign=int(target[2]),
        component_pauli_x=qx, component_pauli_z=qz,
        support=support, raw_restriction_matrix=raw,
        touching_memory_check_rows=touching_rows,
        original_graph=original_graph, original_matching=original_matching,
        original_edge_to_final=original_edge_to_final,
        adapters=adapters, added_adapter_edges=sum(a["adapter_size"] for a in adapters),
        added_adapter_cycle_rows=sum(len(a["cycle_rows"]) for a in adapters),
        final_cycle_rank=rank(cycles),
        cycle_construction="original fundamental cycles plus retained cyclic SkipTree adapter rows",
        connection_topology="chain of adjacent original components",
        automatic_expansion=False, distance_preservation_guaranteed=False,
        distance_note="Connectivity certifies a single graph product; distance requires a separate certificate")
    return graph, matching, cycles, report


def verify_component_adapter_expansion(graph, ports, report, distance, *,
                                       exact_limit=20, max_entries=50_000_000):
    """Recheck a chain-adapter expansion certificate from actual graph rows.

    The report supplies only a proposed vertex partition and edge layout.
    Internal factor graphs, relative expansion, connector bijections, port
    membership and sizes are recomputed. No supplied Boolean is trusted.
    """
    b = graph_matrix(graph, max_entries=max_entries)
    if not isinstance(report, dict) or not isinstance(report.get("components"), list):
        raise ValueError("Component adapter proof requires the construction report")
    components = report["components"]
    count = len(components)
    if count < 1 or report.get("component_count") != count:
        raise ValueError("Invalid component count in adapter report")
    ports = list(ports)
    if (not ports or len(set(ports)) != len(ports)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or not 0 <= v < b.shape[1] for v in ports)):
        raise ValueError("ports must be distinct valid graph vertices")
    port_set = set(ports)
    owner = np.full(b.shape[1], -1, dtype=np.int64)
    for index, component in enumerate(components):
        vertices = component.get("vertices", [])
        if (not vertices or len(set(vertices)) != len(vertices)
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                       or not 0 <= v < b.shape[1] for v in vertices)):
            raise ValueError("Invalid component vertices in adapter report")
        if np.any(owner[vertices] != -1):
            raise ValueError("Adapter component vertex sets overlap")
        owner[vertices] = index
    if np.any(owner == -1):
        raise ValueError("Adapter components must partition all graph vertices")

    internal = [[] for _ in components]
    cross = [[] for _ in range(count - 1)]
    endpoints = [np.flatnonzero(row).tolist() for row in b]
    for edge, (u, v) in enumerate(endpoints):
        left, right = sorted((int(owner[u]), int(owner[v])))
        if left == right:
            internal[left].append(edge)
        elif right == left + 1:
            cross[left].append(edge)
        else:
            raise ValueError("Adapter graph has edges outside the original-component chain")
    adapters = report.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != count - 1:
        raise ValueError("Component adapter report does not describe every chain link")
    verified_links = []
    for index, (adapter, edge_rows) in enumerate(zip(adapters, cross)):
        if (adapter.get("left_component") != index or adapter.get("right_component") != index + 1
                or adapter.get("edge_rows") != edge_rows
                or adapter.get("adapter_size") != len(edge_rows) or not edge_rows):
            raise ValueError("Reported component adapter disagrees with actual graph edges")
        left_vertices, right_vertices = [], []
        for edge in edge_rows:
            u, v = endpoints[edge]
            if owner[u] != index:
                u, v = v, u
            left_vertices.append(u)
            right_vertices.append(v)
        if len(set(left_vertices)) != len(edge_rows) or len(set(right_vertices)) != len(edge_rows):
            raise ValueError("Component adapter edges must form a bijective port matching")
        on_ports = set(left_vertices + right_vertices) <= port_set
        verified_links.append(dict(left_component=index, right_component=index + 1,
            edge_rows=edge_rows, adapter_size=len(edge_rows), vertices_are_ports=on_ports,
            size_at_least_distance=bool(distance is not None and len(edge_rows) >= distance)))

    factors = []
    for index, component in enumerate(components):
        vertices = component["vertices"]
        local_ports = [j for j, vertex in enumerate(vertices) if vertex in port_set]
        if not local_ports:
            raise ValueError("Each adapter component must contain a measurement port")
        factor = b[internal[index]][:, vertices]
        cert = relative_expansion(factor, local_ports, distance=distance,
                                  exact_limit=exact_limit, max_entries=max_entries)
        factors.append(dict(component_index=index, vertices=vertices,
                            edge_rows=internal[index], relative_expansion=cert))
    established = bool(distance is not None
        and all(item["relative_expansion"]["condition_met"] for item in factors)
        and all(item["vertices_are_ports"] and item["size_at_least_distance"] for item in verified_links))
    return dict(condition_met=established, asserted_memory_distance=distance,
        verified_against_actual_graph=True, component_count=count,
        factor_relative_expansion=[item["relative_expansion"] for item in factors],
        factors=factors, adapters=verified_links,
        source="Universal adapters Lemma 9 (v4 numbering), applied along the component chain",
        note="Static beta_d >= 1 certificate only; no circuit fault-distance claim")
