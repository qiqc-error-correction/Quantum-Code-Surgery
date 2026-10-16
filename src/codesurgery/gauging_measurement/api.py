"""Williamson--Yoder auxiliary-graph measurement, including signed Y/XZ.

Graph construction choices are explicit. This implements the static,
flux-fixed deformed code, not a syndrome-extraction circuit simulator.
"""
import numpy as np
from ..gf2 import binary_matrix, solve, nullspace, rank
from ..symplectic import commutation_matrix
from ..gadget_common import (prepare, prepare_pair, memory_checks, restriction,
                            consistent_signs, validate_readout)
from ..universal_adapters import repetition_adapter
from ..universal_adapters.graphs import (graph_matrix, cycle_basis, relative_expansion,
                                        adjacency, budget_check)
from .models import GraphMeasurementCode
from ..distance_preservation.api import finish_distance


def paired_graph(code, target, *, max_entries=50_000_000):
    """Pair each even check boundary; join components by a chain of extra edges.

    This is the short-matching construction in Universal adapters II.3.
    Sorted pairing and representative choice are deterministic implementation
    choices. No expansion guarantee is asserted for the resulting graph.
    """
    support, raw, _, rows = restriction(code, target)
    w, r = len(support), len(code.hx) + len(code.hz)
    edges, owners = [], []
    for boundary, owner in zip(raw, rows):
        vertices = np.flatnonzero(boundary).tolist()
        for j in range(0, len(vertices), 2):
            edges.append((vertices[j], vertices[j + 1]))
            owners.append(int(owner))
    budget_check(max_entries, (len(edges) + w) * max(w, r))
    graph = np.zeros((len(edges), w), dtype=np.uint8)
    for e, endpoints in enumerate(edges):
        graph[e, list(endpoints)] = 1
    adj = adjacency(graph)
    seen, representatives = set(), []
    for root in range(w):
        if root in seen:
            continue
        component = [root]
        seen.add(root)
        for u in component:
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    component.append(v)
        representatives.append(min(component, key=lambda v: (len(adj[v]), v)))
    joins = list(zip(representatives, representatives[1:]))
    edges.extend(joins)
    graph = np.zeros((len(edges), w), dtype=np.uint8)
    matching = np.zeros((r, len(edges)), dtype=np.uint8)
    for e, endpoints in enumerate(edges):
        graph[e, list(endpoints)] = 1
        if e < len(owners):
            matching[owners[e], e] = 1
    return graph, matching, dict(method="paired check boundaries with component chain",
                                raw_restriction_matrix=raw, component_join_edges=joins,
                                automatic_expansion=False)


def build_graph(code, target, basis, graph, *, ports=None, matching=None,
                cycles=None, scheme="gauging_measurement", extra_metadata=None,
                incidence_kind="graph", exact_limit=20, max_entries=50_000_000,
                _defer_expansion=False):
    # Search candidates must be checked for actual distance before enumerating
    # graph cuts. Deferral grants no theorem certificate and skips no algebra.
    if not isinstance(_defer_expansion, (bool, np.bool_)):
        raise ValueError("_defer_expansion must be a Boolean")
    if _defer_expansion and extra_metadata and any(
            key in extra_metadata for key in ("adapter_distance_proof", "component_adapter_distance_proof",
                                              "postprocessing_distance_proof")):
        raise ValueError("Deferred expansion cannot include a graph distance proof")
    if incidence_kind == "graph":
        b = graph_matrix(graph, max_entries=max_entries)
    elif incidence_kind == "even_tanner":
        budget_check(max_entries, np.asarray(graph).size)
        b = binary_matrix(graph, name="Tanner restriction")
        if b.shape[1] < 1 or np.any(b.sum(axis=1) % 2):
            raise ValueError("Tanner restriction needs support columns and even rows (F @ 1 = 0)")
    else:
        raise ValueError("incidence_kind must be graph or even_tanner")
    e, v = b.shape
    n = code.n
    x, z, sign = target
    support = np.flatnonzero(x | z)
    r = len(code.hx) + len(code.hz)
    upper = r + v + e
    budget_check(max_entries, 2 * upper * (n + e), upper * upper, v * n, e * e)
    ports = list(range(len(support))) if ports is None else list(ports)
    if len(ports) != len(support) or len(set(ports)) != len(ports) or any(isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or not 0 <= p < v for p in ports):
        raise ValueError("ports must give one distinct graph vertex for each sorted support qubit")
    s = np.zeros((v, n), dtype=np.uint8)
    s[ports, support] = 1
    oldx, oldz = memory_checks(code)
    boundary = (((oldx * z + oldz * x) % 2) @ s.T) % 2
    if matching is None:
        m = solve(b.T, boundary.T).T
    else:
        budget_check(max_entries, np.asarray(matching).size)
        m = binary_matrix(matching, n_cols=e, name="matching")
    if m.shape[0] != r or not np.array_equal((m @ b) % 2, boundary):
        raise ValueError("matching must satisfy M B = memory anticommutation boundary")
    ordinary_connected = bool(np.all(b.sum(axis=1) == 2) and rank(b) == v - 1)
    if ordinary_connected:
        cycles = cycle_basis(b, cycles, max_entries=max_entries)
    else:
        if cycles is None:
            cycles = nullspace(b.T)
        else:
            budget_check(max_entries, np.asarray(cycles).size)
            cycles = binary_matrix(cycles, n_cols=e, name="cycle_basis")
        if np.any((cycles @ b) % 2) or rank(cycles) != e - rank(b):
            raise ValueError("cycle_basis must span the complete kernel of F.T")
    total = r + v + len(cycles)
    budget_check(max_entries, 2 * total * (n + e), total * total)
    cx = np.zeros((total, n + e), dtype=np.uint8)
    cz = np.zeros_like(cx)
    cx[:r, :n], cz[:r, :n] = oldx, oldz
    cx[r:r + v, :n], cz[r:r + v, :n] = s * x, s * z
    ancilla = "Z" if np.any(z) and not np.any(x) else "X"
    same, opposite = (cz, cx) if ancilla == "Z" else (cx, cz)
    opposite[:r, n:] = m
    same[r:r + v, n:] = b.T
    opposite[r + v:, n:] = cycles
    if np.any(commutation_matrix(cx, cz)):
        raise ValueError("Graph gadget has noncommuting checks")
    witness = np.zeros(total, dtype=np.uint8)
    witness[r:r + v] = 1
    signs = consistent_signs(cx, cz, witness, sign, list(range(r)) + list(range(r + v, total)))
    validation = validate_readout(code, cx, cz, target, basis)
    if _defer_expansion:
        certificate = dict(parameter_t=code.distance or len(ports), port_vertices=ports,
            exact=False, beta=None, lower_bound=0.0, condition_met=False,
            certificate="deferred: evaluate actual stabilizer distance before graph expansion",
            deferred=True)
    elif ordinary_connected:
        certificate = relative_expansion(b, ports, distance=code.distance,
                                         exact_limit=exact_limit, max_entries=max_entries)
    else:
        # The WSC Tanner-boundary criterion applies to even rows of any weight.
        # Ordinary-graph connectivity and adapter beta_d shortcuts do not.
        from ..explicit_gadgets.expansion import expand_restriction
        _, certificate = expand_restriction(b, mode="none", exact_limit=exact_limit,
                                            max_entries=max_entries)
        certificate = dict(certificate, measure="Tanner parity-boundary Cheeger constant",
                           condition_met=certificate["distance_condition_met"])
    # Re-derive any compositional certificate from these actual graph blocks.
    # A supplied Boolean certificate is never trusted on its own.
    if extra_metadata and "adapter_distance_proof" in extra_metadata:
        if not ordinary_connected:
            raise ValueError("Adapter distance proof requires an ordinary connected graph")
        adapter = extra_metadata.get("adapter")
        if not isinstance(adapter, dict) or not np.array_equal(adapter.get("incidence"), b):
            raise ValueError("Adapter certificate requires the actual adapter construction")
        pl, pr = adapter["P_left"], adapter["P_right"]
        ml, mr = adapter["T_left"].shape[1], adapter["T_right"].shape[1]
        nl, nr, a = pl.shape[0], pr.shape[0], adapter["adapter_size"]
        if (b.shape != (ml + a + mr, nl + nr) or pl.shape[1] != a or pr.shape[1] != a
                or np.any(pl.sum(axis=0) != 1) or np.any(pr.sum(axis=0) != 1)
                or np.any(pl.sum(axis=1) > 1) or np.any(pr.sum(axis=1) > 1)
                or not np.array_equal(b[ml:ml + a, :nl], pl.T)
                or not np.array_equal(b[ml:ml + a, nl:], pr.T)
                or np.any(b[:ml, nl:]) or np.any(b[ml + a:, :nl])):
            raise ValueError("Adapter blocks do not match their distance certificate")
        lp, rp = [p for p in ports if p < nl], [p - nl for p in ports if p >= nl]
        factors = [relative_expansion(b[:ml, :nl], lp, distance=code.distance,
                    exact_limit=exact_limit, max_entries=max_entries),
                   relative_expansion(b[ml + a:, nl:], rp, distance=code.distance,
                    exact_limit=exact_limit, max_entries=max_entries)]
        # An outer adapter may join factors that were themselves constructed
        # from several components. Re-derive those compositional proofs from
        # the actual factor blocks instead of discarding them when full-graph
        # cut enumeration exceeds exact_limit.
        factor_reports = extra_metadata.get("factors", [])
        if len(factor_reports) == 2:
            from .components import verify_component_adapter_expansion
            for index, (factor_graph, factor_ports) in enumerate(
                    ((b[:ml, :nl], lp), (b[ml + a:, nl:], rp))):
                design = factor_reports[index].get("graph_design", {})
                if "component_count" not in design:
                    continue
                proof = verify_component_adapter_expansion(factor_graph, factor_ports, design,
                    code.distance, exact_limit=exact_limit, max_entries=max_entries)
                if proof["condition_met"] and not factors[index]["condition_met"]:
                    if factors[index]["exact"]:
                        raise RuntimeError("Factor component proof conflicts with exact expansion")
                    factors[index].update(condition_met=True, lower_bound=1.0,
                        certificate="Verified component-adapter composition inside outer factor")
                factors[index]["component_adapter_proof"] = proof
        on_ports = set(np.flatnonzero(pl.sum(axis=1))) <= set(lp) and set(np.flatnonzero(pr.sum(axis=1))) <= set(rp)
        established = bool(code.distance is not None and a >= code.distance and on_ports
                           and all(item["condition_met"] for item in factors))
        extra_metadata = dict(extra_metadata)
        extra_metadata["adapter_distance_proof"] = dict(factor_relative_expansion=factors,
            vertices_are_ports=bool(on_ports), condition_met=established,
            source="Universal adapters Lemma 9 (v4 numbering)")
        if established and not certificate["condition_met"]:
            if certificate["exact"]:
                raise RuntimeError("Adapter theorem certificate conflicts with exact expansion")
            certificate.update(condition_met=True, lower_bound=1.0,
                certificate="Universal adapters Lemma 9: both factor beta_d>=1, adapter ports and size>=d")
    if extra_metadata and "component_adapter_distance_proof" in extra_metadata:
        if not ordinary_connected:
            raise ValueError("Component adapter proof requires an ordinary connected graph")
        from .components import verify_component_adapter_expansion
        verified = verify_component_adapter_expansion(b, ports, extra_metadata.get("graph_design"),
            code.distance, exact_limit=exact_limit, max_entries=max_entries)
        extra_metadata = dict(extra_metadata)
        extra_metadata["component_adapter_distance_proof"] = verified
        if verified["condition_met"] and not certificate["condition_met"]:
            if certificate["exact"]:
                raise RuntimeError("Component adapter proof conflicts with exact expansion")
            certificate.update(condition_met=True, lower_bound=1.0,
                certificate="Universal adapters Lemma 9 iterated along the verified component chain")
    if extra_metadata and "postprocessing_distance_proof" in extra_metadata:
        if not ordinary_connected:
            raise ValueError("Postprocessing proof requires an ordinary connected final graph")
        from ..distance_preservation.api import verify_embedded_expansion
        verified = verify_embedded_expansion(b, ports, code.distance,
            extra_metadata["postprocessing_distance_proof"], max_entries=max_entries)
        if certificate["exact"] and not certificate["condition_met"]:
            raise RuntimeError("Embedded expansion proof conflicts with exact final expansion")
        if not certificate["condition_met"]:
            certificate = verified
        extra_metadata = dict(extra_metadata)
        extra_metadata["postprocessing_distance_proof"] = dict(
            extra_metadata["postprocessing_distance_proof"], verified_base_certificate=verified["base_certificate"])
    weights, degrees = (cx | cz).sum(axis=1), (cx | cz).sum(axis=0)
    metadata = dict(scheme=scheme, incidence_kind=incidence_kind,
        construction="connected auxiliary graph with full flux closure" if ordinary_connected else
                     "even Tanner restriction with complete auxiliary kernel checks",
        graph_incidence=b, port_selector=s, ports=ports, support=support,
        matching_matrix=m, cycle_basis=cycles, logical_basis=basis,
        ancilla_vertex_pauli=ancilla,
        relative_expansion=certificate if ordinary_connected else None,
        boundary_expansion=None if ordinary_connected else certificate,
        max_check_weight=int(weights.max(initial=0)), max_qubit_degree=int(degrees.max(initial=0)),
        max_matching_weight=int(m.sum(axis=1).max(initial=0)),
        max_matching_congestion=int(m.sum(axis=0).max(initial=0)),
        max_cycle_weight=int(cycles.sum(axis=1).max(initial=0)),
        max_cycle_congestion=int(cycles.sum(axis=0).max(initial=0)),
        ldpc_family_guaranteed=False,
        automatic_expansion=bool(extra_metadata and extra_metadata.get("expansion", {}).get("augmentation_rows", 0)),
        distance_preservation_guaranteed=bool(certificate["condition_met"] and code.distance is not None),
        distance_lower_bound=code.distance if certificate["condition_met"] else None,
        distance_note=("Static code-distance certificate via beta_d >= 1" if ordinary_connected else
                       "Static code-distance certificate via Tanner h(F) >= 1") + "; memory distance is user asserted",
        measurement_fault_distance=None, circuit_fault_distance=None,
        row_layout=dict(memory=list(range(r)), vertex=list(range(r, r + v)), cycles=list(range(r + v, total))),
        column_layout=dict(memory=list(range(n)), graph_edges=list(range(n, n + e))),
        protocol=dict(edge_initialization="+" if ancilla == "Z" else "0",
                      logical_readout="product of all signed vertex-check outcomes",
                      ungauging_measurement="X on each edge qubit" if ancilla == "Z" else "Z on each edge qubit",
                      status="static code and protocol skeleton; no fault-tolerant circuit compiled"),
        source_paper="https://arxiv.org/abs/2511.15989" if scheme.startswith("webster") else "https://arxiv.org/abs/2410.02213",
        adapter_reference="https://arxiv.org/abs/2410.03628", **validation)
    if extra_metadata:
        if set(extra_metadata) & set(metadata):
            raise ValueError("Extra metadata cannot override validated construction fields")
        metadata.update(extra_metadata)
    is_css = not np.any(np.any(cx, axis=1) & np.any(cz, axis=1))
    result = GraphMeasurementCode(cx, cz, signs, x, z, sign, witness, n, code.k - 1,
        metadata=metadata, hx=cx[np.any(cx, axis=1)].copy() if is_css else None,
        hz=cz[np.any(cz, axis=1)].copy() if is_css else None, _source_code=code)
    result._seal()
    result.validate()
    return result


def gauging_deform(code, logical=None, *, graph=None, ports=None, matching=None,
                   cycle_basis=None, pauli="Z", logical_index=0, logical_basis=None,
                   component_connection="adapter", component_adapter_size=None,
                   component_adapter_shift=None,
                   exact_limit=20, max_entries=50_000_000,
                   require_distance=True, distance_options=None,
                   optimize_logical=False, logical_optimization_options=None):
    """Gauge one signed target; join automatic graph components with adapters.

    The default uses SkipTree repetition adapters and their complete cycle
    checks. ``component_connection='chain'`` explicitly selects the previous
    single-edge chain. Component options apply only to automatic graphs;
    user-supplied graphs are never silently connected or changed. The adapter
    size defaults to the smaller of each pair of adjacent component sizes.
    A None component_adapter_shift chooses a cyclic rotation minimizing the
    heaviest new cycle; an explicit integer fixes the rotation.

    By default reduction runs only at weight>2*code.distance; unknown distance
    skips the automatic rule. Explicit options may override this threshold.
    Any supplied graph/ports/matching/cycle basis must describe the optimized
    support. Use reduce_logical_weight separately to design those blocks.
    """
    source, target, basis = prepare(code, logical, pauli=pauli, logical_index=logical_index,
                                    logical_basis=logical_basis, max_entries=max_entries)
    if component_connection not in ("adapter", "chain"):
        raise ValueError("component_connection must be adapter or chain")
    if component_adapter_size is not None and (isinstance(component_adapter_size, (bool, np.bool_))
            or not isinstance(component_adapter_size, (int, np.integer)) or component_adapter_size < 1):
        raise ValueError("component_adapter_size must be a positive integer")
    if component_adapter_shift is not None and (isinstance(component_adapter_shift, (bool, np.bool_))
            or not isinstance(component_adapter_shift, (int, np.integer)) or component_adapter_shift < 0):
        raise ValueError("component_adapter_shift must be a nonnegative integer or None")
    custom_component_options = component_adapter_size is not None or component_adapter_shift is not None
    if component_connection == "chain" and custom_component_options:
        raise ValueError("component adapter options require component_connection='adapter'")
    if graph is not None and (custom_component_options or component_connection != "adapter"):
        raise ValueError("component connection options require an automatic graph")
    metadata = {}
    if not isinstance(optimize_logical, (bool, np.bool_)):
        raise ValueError("optimize_logical must be a Boolean")
    if logical_optimization_options is not None and not isinstance(logical_optimization_options, dict):
        raise ValueError("logical_optimization_options must be a dictionary")
    if logical_optimization_options is not None and not optimize_logical:
        raise ValueError("logical_optimization_options requires optimize_logical=True")
    if optimize_logical:
        from ..logical_optimization import optimize_prepared_logical
        target, metadata["logical_optimization"] = optimize_prepared_logical(
            source, target, basis, **(logical_optimization_options or {}))
    if graph is None:
        if ports is not None or matching is not None:
            raise ValueError("ports/matching require an explicit graph")
        if component_connection == "adapter":
            from .components import paired_component_graph
            graph, matching, generated_cycles, report = paired_component_graph(
                source, target, basis, adapter_size=component_adapter_size,
                adapter_shift=component_adapter_shift, exact_limit=exact_limit,
                max_entries=max_entries)
            if cycle_basis is None:
                cycle_basis = generated_cycles
            metadata["component_adapter_distance_proof"] = {}
        else:
            graph, matching, report = paired_graph(source, target, max_entries=max_entries)
        metadata["graph_design"] = report
    result = build_graph(source, target, basis, graph, ports=ports, matching=matching,
                         cycles=cycle_basis, extra_metadata=metadata,
                         exact_limit=exact_limit, max_entries=max_entries)
    return finish_distance(result, require_distance=require_distance, distance_options=distance_options,
                           exact_limit=exact_limit, max_entries=max_entries)


def join_graph_gadgets(code, targets, target, basis, specs, *, scheme, adapter_size=None,
                       adapter_vertices=None, adapter_shift=0, extra_metadata=None,
                       exact_limit=20, max_entries=50_000_000):
    left, right = specs
    if adapter_vertices is not None and len(adapter_vertices) != 2:
        raise ValueError("adapter_vertices must contain two vertex subsets")
    connector = repetition_adapter(left["graph"], right["graph"],
        left_cycles=left.get("cycles"), right_cycles=right.get("cycles"), adapter_size=adapter_size,
        adapter_shift=adapter_shift,
        left_vertices=None if adapter_vertices is None else adapter_vertices[0],
        right_vertices=None if adapter_vertices is None else adapter_vertices[1], max_entries=max_entries)
    nl = left["graph"].shape[1]
    ml, mr = len(left["graph"]), len(right["graph"])
    size = connector["adapter_size"]
    m = np.zeros((len(code.hx) + len(code.hz), ml + size + mr), dtype=np.uint8)
    m[:, :ml], m[:, ml + size:] = left["matching"], right["matching"]
    port_map = {}
    for t, spec, offset in zip(targets, specs, (0, nl)):
        for q, port in zip(np.flatnonzero(t[0] | t[1]), spec["ports"]):
            port_map[int(q)] = int(port) + offset
    ports = [port_map[int(q)] for q in np.flatnonzero(target[0] | target[1])]
    metadata = dict(extra_metadata or {})
    metadata["adapter"] = connector
    metadata["adapter_distance_proof"] = {}  # Recomputed from actual blocks by build_graph.
    return build_graph(code, target, basis, connector["incidence"], ports=ports, matching=m,
        cycles=connector["cycle_basis"], scheme=scheme, extra_metadata=metadata,
        exact_limit=exact_limit, max_entries=max_entries)


def gauging_pair_deform(code, left=None, right=None, *, graphs=None, ports=None,
                        matchings=None, cycle_bases=None, paulis=("Z", "Z"), logical_indices=(0, 1),
                        logical_basis=None, adapter_size=None, adapter_vertices=None,
                        adapter_shift=0,
                        component_connection="adapter", component_adapter_size=None,
                        component_adapter_shift=None,
                        exact_limit=20, max_entries=50_000_000,
                        require_distance=True, distance_options=None):
    source, l, r, target, basis = prepare_pair(code, left, right, paulis=paulis,
        logical_indices=logical_indices, logical_basis=logical_basis, max_entries=max_entries)
    for name, value in (("graphs", graphs), ("ports", ports), ("matchings", matchings), ("cycle_bases", cycle_bases)):
        if value is not None and len(value) != 2:
            raise ValueError(f"{name} must contain exactly two entries")
    specs, reports = [], []
    for i, factor in enumerate((l, r)):
        # Construct and validate each graph gadget before attaching the adapter.
        factor_result = gauging_deform(source, dict(x=factor[0], z=factor[1], sign=factor[2]),
            logical_basis=basis, graph=None if graphs is None else graphs[i],
            ports=None if ports is None else ports[i], matching=None if matchings is None else matchings[i],
            cycle_basis=None if cycle_bases is None else cycle_bases[i], exact_limit=exact_limit, max_entries=max_entries,
            component_connection=component_connection, component_adapter_size=component_adapter_size,
            component_adapter_shift=component_adapter_shift,
            require_distance=False)
        md = factor_result.metadata
        specs.append(dict(graph=md["graph_incidence"], matching=md["matching_matrix"],
                          ports=md["ports"], cycles=md["cycle_basis"]))
        reports.append(dict(target_x=factor[0], target_z=factor[1], target_sign=factor[2],
                            relative_expansion=md["relative_expansion"],
                            graph_design=md.get("graph_design", {"method": "user graph"})))
    result = join_graph_gadgets(source, (l, r), target, basis, specs,
        scheme="gauging_measurement_with_universal_adapter", adapter_size=adapter_size,
        adapter_vertices=adapter_vertices, adapter_shift=adapter_shift, extra_metadata=dict(factors=reports),
        exact_limit=exact_limit, max_entries=max_entries)
    return finish_distance(result, require_distance=require_distance, distance_options=distance_options,
                           exact_limit=exact_limit, max_entries=max_entries)
