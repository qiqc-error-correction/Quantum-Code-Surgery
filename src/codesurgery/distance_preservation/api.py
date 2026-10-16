"""Computed static distance and sparse-check postprocessing.

Existing small constructions are retained when their certificates and selected
weight bounds already suffice. The finite-size layer-coloring construction
has checkable bounds; it does not claim Freedman--Hastings optimal overhead.
"""
import copy
import numpy as np

from ..gf2 import rank
from ..universal_adapters.graphs import graph_matrix, budget_check
from .expansion import certify_expansion, enhance_expansion, _integer


def finish_distance(result, *, require_distance=True, distance_options=None,
                    exact_limit=20, max_entries=50_000_000):
    if not isinstance(require_distance, (bool, np.bool_)):
        raise ValueError("require_distance must be a Boolean")
    if distance_options is not None and not isinstance(distance_options, dict):
        raise ValueError("distance_options must be a dictionary")
    if not require_distance:
        return result  # Keep configured limits available for a later opt-in run.
    options = dict(distance_options or {})
    options.setdefault("exact_limit", exact_limit)
    options.setdefault("max_entries", max_entries)
    return ensure_distance(result, **options)


def verify_embedded_expansion(b, ports, distance, proof, *, max_entries=50_000_000):
    """Verify the actual preserved base subgraph, then recompute its certificate.

    Every cut of the final graph restricts to a cut of this base on the same
    ports. Base cut edges are retained, so beta_d(final,ports)>=beta_d(base,ports).
    No claim about global expansion of the enlarged graph is required.
    """
    if not isinstance(proof, dict) or "base_graph" not in proof:
        raise ValueError("Postprocessing distance proof needs its actual base graph")
    base = graph_matrix(proof["base_graph"], max_entries=max_entries)
    e, v = base.shape
    base_ports = list(proof.get("base_ports", []))
    if (distance is None or proof.get("distance") != distance or list(ports) != base_ports
            or b.shape[0] < e or b.shape[1] < v
            or not np.array_equal(b[:e, :v], base) or np.any(b[:e, v:])):
        raise ValueError("Postprocessing certificate does not match the actual embedded base and ports")
    cert = certify_expansion(base, ports=base_ports, distance=distance,
        exact_limit=proof.get("exact_limit", 20),
        certificate_limit=proof.get("certificate_limit", 128), max_entries=max_entries)
    if not cert["condition_met"]:
        raise ValueError("The embedded base graph does not have a verified distance certificate")
    return dict(condition_met=True, lower_bound=1.0, exact=False, beta=None,
        parameter_t=int(distance), port_vertices=list(ports),
        certificate="verified embedded base graph on unchanged ports; cuts cannot shrink",
        base_certificate=cert, verified_from_actual_incidence=True)


def _metrics(result):
    md = result.metadata
    b, m, c = md["graph_incidence"], md["matching_matrix"], md["cycle_basis"]
    return dict(max_check_weight=int((result.check_x | result.check_z).sum(axis=1).max(initial=0)),
                max_qubit_degree=int((result.check_x | result.check_z).sum(axis=0).max(initial=0)),
                max_graph_degree=int(b.sum(axis=0).max(initial=0)),
                max_matching_weight=int(m.sum(axis=1).max(initial=0)),
                max_matching_congestion=int(m.sum(axis=0).max(initial=0)),
                max_cycle_weight=int(c.sum(axis=1).max(initial=0)),
                max_cycle_congestion=int(c.sum(axis=0).max(initial=0)))


def _distance_state(report, target):
    if report.get("distance_is_infinite") or report.get("status") == "no_logicals":
        return "adequate"
    upper = report.get("upper_bound")
    if upper is not None and upper < target:
        return "insufficient"
    if report["proven_lower_bound"] >= target:
        return "adequate"
    return "unknown"


def _checked_distance(result, source_report, target, *, search_options):
    from .distance import verify_distance
    result.validate()
    report = verify_distance(result, target, **search_options)
    state = _distance_state(report, target)
    # An existing rigorous graph certificate can establish the lower bound
    # when the selected distance backend exhausts its separate budget.
    certificate = result.metadata.get("relative_expansion") or result.metadata.get("boundary_expansion") or {}
    theorem = bool(certificate.get("condition_met"))
    if state == "insufficient" and theorem:
        raise ValueError("Computed distance contradicts the graph distance certificate; check source distance and construction")
    verified = state == "adequate" or theorem
    output = copy.deepcopy(result)
    output._source_code.distance = target
    output.metadata.update(source_distance_calculation=source_report,
        distance_calculation=report, distance_target=target,
        distance_preservation_guaranteed=verified,
        distance_lower_bound=target if verified else None,
        distance_verification_method="actual stabilizer distance search" if state == "adequate" else
            "graph lower-bound certificate; distance threshold search incomplete" if theorem else
            "actual distance is insufficient" if state == "insufficient" else "distance remains undetermined",
        distance_note="Static code distance; actual search and graph lower-bound certificate are reported separately")
    output._digest = None
    output._seal()
    output.validate()
    return output, state


def _annotated(result, report, limits):
    result = copy.deepcopy(result)
    actual = _metrics(result)
    failures = {key: dict(actual=actual[key], limit=value)
                for key, value in limits.items() if actual[key] > value}
    if failures:
        row_weights = (result.check_x | result.check_z).sum(axis=1)
        heavy_rows = np.flatnonzero(row_weights > limits["max_check_weight"]).tolist()
        raise ValueError(f"Cannot meet requested sparsity limits after sparsification: {failures}; "
                         f"overweight check rows={heavy_rows}. The memory-based threshold was not increased")
    if not result.metadata["distance_preservation_guaranteed"]:
        if _distance_state(result.metadata["distance_calculation"], result.metadata["distance_target"]) == "insufficient":
            raise ValueError("Computed distance remains below target after expansion and sparsification; no certified result returned")
        raise ValueError("Actual distance remains undetermined within the search budget, and no distance certificate is available; increase the selected distance search budget or choose another distance_method")
    report = dict(report, required=True, static_distance_only=True,
                  distance_lower_bound=result.metadata["distance_lower_bound"],
                  final_distance_calculation=result.metadata["distance_calculation"],
                  final_n=result.n, sparsity_certificate=dict(condition_met=True,
                  actual=actual, limits=limits, verified_against_full_check_matrix=True,
                  family_note="Uniform bounds imply LDPC for an LDPC input family; no asymptotic family is inferred from one instance"))
    result.metadata["distance_postprocessing"] = report
    result.metadata["sparsity_conditions_verified"] = True
    result._digest = None
    result._seal()
    result.validate()
    return result


def ensure_distance(result, *, max_base_degree=12, max_cycle_weight=None,
                    max_cycle_congestion=None, max_layers=128,
                    optimize_cycle_basis=True, cycle_basis_max_roots=64,
                    cycle_basis_max_candidates=10_000,
                    max_matching_weight=None, max_matching_congestion=1,
                    max_check_weight=None, max_qubit_degree=None,
                    distance_max_candidates=2_000_000, distance_max_weight=None,
                    distance_method="auto", distance_ris_trials=64, distance_seed=0,
                    distance_time_limit=60.0, distance_node_limit=None,
                    seed=0, attempts=12, exact_limit=20,
                    certificate_limit=128, max_entries=50_000_000,
                    expansion_strategy="degree_incremental",
                    unknown_distance_policy="certify"):
    """Certify distance, repair or seek a certificate, and enforce fixed weights.

    The default threshold uses [HX | M_X] for a pure Z target, its dual
    [HZ | M_Z] for pure X, and all dressed symplectic memory rows for Y/XZ.
    An unknown search result is never interpreted as an actual distance loss.
    By default, when no lower-bound certificate is available, bounded graph
    repair may continue to seek one. unknown_distance_policy='error' instead
    stops immediately and preserves the earlier strict-search behavior.
    Default finite-resource searches either prove enough, use a valid graph
    lower-bound certificate, or raise without returning an uncertified code.
    A declared source distance is a trusted target, checked for counterexamples;
    an undeclared source distance must be computed exactly. Final codes need
    only a threshold certificate, so their exact distance may remain unknown.
    Distance backend options have a separate seed and budgets from graph repair.
    The default repair raises the candidate degree cap one at a time, retries
    seeded random degree completions, and evaluates the fully sparsified code
    before graph expansion. Expansion-based acceptance requires global h>1;
    an adequate actual distance can stop earlier. All weight caps stay fixed.
    Before graph layering, default bounded cycle-basis optimization tries to
    satisfy cycle weight/congestion limits without changing graph or matching.
    Set optimize_cycle_basis=False to keep the previous layering behavior.
    """
    from ..gauging_measurement.api import build_graph, paired_graph
    from ..gauging_measurement.models import GraphMeasurementCode
    from .distance import code_distance, verify_distance
    from .sparsify import sparsify_cycles
    from .cycles import optimization_options, optimize_cycles
    if not isinstance(result, GraphMeasurementCode):
        raise TypeError("ensure_distance requires an explicit/gauging graph measurement result")
    if expansion_strategy not in ("degree_incremental", "hamiltonian"):
        raise ValueError("expansion_strategy must be degree_incremental or hamiltonian")
    if unknown_distance_policy not in ("certify", "error"):
        raise ValueError("unknown_distance_policy must be certify or error")
    max_base_degree = _integer(max_base_degree, "max_base_degree")
    max_layers = _integer(max_layers, "max_layers")
    optimize_cycle_basis, cycle_basis_max_roots, cycle_basis_max_candidates = optimization_options(
        optimize_cycle_basis, cycle_basis_max_roots, cycle_basis_max_candidates)
    max_matching_congestion = _integer(max_matching_congestion, "max_matching_congestion", minimum=0)
    exact_limit = _integer(exact_limit, "exact_limit", maximum=24)
    certificate_limit = _integer(certificate_limit, "certificate_limit")
    seed = _integer(seed, "seed", minimum=0)
    attempts = _integer(attempts, "attempts")
    distance_max_candidates = _integer(distance_max_candidates, "distance_max_candidates", minimum=0)
    if distance_max_weight is not None:
        distance_max_weight = _integer(distance_max_weight, "distance_max_weight", minimum=0)
    if max_cycle_weight is not None:
        max_cycle_weight = _integer(max_cycle_weight, "max_cycle_weight")
    budget_check(max_entries, result.check_x.size, result.check_z.size,
                 result.metadata["graph_incidence"].size, result.metadata["cycle_basis"].size,
                 result.metadata["matching_matrix"].size)
    result.validate()
    source = result._source_code
    search_options = dict(method=distance_method, max_weight=distance_max_weight,
        max_candidates=distance_max_candidates, ris_trials=distance_ris_trials,
        seed=distance_seed, time_limit=distance_time_limit, node_limit=distance_node_limit)
    target_distance = source.distance
    source_report = (code_distance(source, **search_options) if target_distance is None
                     else verify_distance(source, target_distance, **search_options))
    if target_distance is None:
        target_distance = source_report["distance"]
        if target_distance is None:
            raise ValueError("Source distance is undetermined within the search budget; increase the selected distance search budget, choose an exact backend, or supply a trusted source distance")
    elif (source_report.get("upper_bound") is not None
          and source_report["upper_bound"] < target_distance):
        raise ValueError(f"Declared source distance {target_distance} exceeds computed distance {source_report['upper_bound']}")
    initial, initial_state = _checked_distance(result, source_report, target_distance,
        search_options=search_options)
    source = initial._source_code
    md = initial.metadata
    memory_weights = np.concatenate([source.hx.sum(axis=1), source.hz.sum(axis=1)])
    dressed_weights = memory_weights + md["matching_matrix"].sum(axis=1)
    if not np.any(initial.measurement_x):
        reference_rows = list(range(len(source.hx)))
        reference = "maximum row weight of horizontal [HX | M_X] for a pure Z target"
    elif not np.any(initial.measurement_z):
        reference_rows = list(range(len(source.hx), len(memory_weights)))
        reference = "maximum row weight of horizontal [HZ | M_Z] for a pure X target (dual convention)"
    else:
        reference_rows = list(range(len(memory_weights)))
        reference = "maximum dressed symplectic memory row weight wt(CX OR CZ)+wt(M) for a mixed Y/XZ target"
    threshold = int(dressed_weights[reference_rows].max(initial=0))
    row_cap = max(1, threshold) if max_check_weight is None else _integer(max_check_weight, "max_check_weight")
    # A code without memory checks has a zero reference threshold. It cannot
    # support a nonempty measured check below zero; require an explicit cap.
    if threshold == 0 and max_check_weight is None:
        raise ValueError("The memory block [H | M] has row threshold 0; supply max_check_weight for a check-free source")
    cycle_cap = max(4, max_base_degree) if max_cycle_congestion is None else _integer(max_cycle_congestion, "max_cycle_congestion")
    cycle_weight_cap = min(row_cap, max_cycle_weight) if max_cycle_weight is not None else row_cap
    mmw = int(md["matching_matrix"].sum(axis=1).max(initial=0)) if max_matching_weight is None else _integer(max_matching_weight, "max_matching_weight", minimum=0)
    q0 = int((source.hx.sum(axis=0) + source.hz.sum(axis=0)).max(initial=0))
    col_cap = max(q0 + 1, 2 + max_matching_congestion + cycle_cap) if max_qubit_degree is None else _integer(max_qubit_degree, "max_qubit_degree")
    limits = dict(max_check_weight=row_cap, max_qubit_degree=col_cap,
                  max_graph_degree=max_base_degree + 4, max_matching_weight=mmw,
                  max_matching_congestion=max_matching_congestion,
                  max_cycle_weight=cycle_weight_cap, max_cycle_congestion=cycle_cap)
    report = dict(initial_n=result.n, initial_distance_calculation=initial.metadata["distance_calculation"],
                  source_distance_calculation=source_report, target_distance=target_distance,
                  distance_search_options=search_options,
                  expansion_strategy=expansion_strategy,
                  unknown_distance_policy=unknown_distance_policy,
                  initial_distance_state=initial_state,
                  sparsity_threshold=threshold, effective_row_threshold=row_cap,
                  sparsity_threshold_source=reference, sparsity_reference_rows=reference_rows,
                  dressed_memory_row_weights=dressed_weights.astype(int).tolist(),
                  expansion_added_edges=0, expansion_trigger="none", rounds=[])
    before = _metrics(initial)
    if (initial.metadata["distance_preservation_guaranteed"]
            and all(before[k] <= value for k, value in limits.items())):
        return _annotated(initial, dict(report, changed=False,
            action="computed distance and sparsity limits already pass",
            layers=1, cellulation_added_edges=0), limits)
    if (initial_state == "unknown" and not initial.metadata["distance_preservation_guaranteed"]
            and unknown_distance_policy == "error"):
        raise ValueError("Actual distance is undetermined within the search budget; expansion is not triggered by an unknown distance. Increase the selected distance search budget or choose another distance_method")

    current = initial
    target = result.measurement_x, result.measurement_z, result.measurement_sign
    origin = dict(scheme=md["scheme"], graph_incidence=md["graph_incidence"],
        matching_matrix=md["matching_matrix"], cycle_basis=md["cycle_basis"], ports=md["ports"],
        adapter=md.get("adapter"), precheck=md.get("precheck"), graph_design=md.get("graph_design"),
        component_adapter_distance_proof=md.get("component_adapter_distance_proof"),
        factors=md.get("factors"))
    converted = False
    representative_metadata = ({"logical_optimization": md["logical_optimization"]}
                               if "logical_optimization" in md else {})

    def prepare_sparse(graph, matching, supplied_cycles):
        if cycle_weight_cap < 4:
            from ..universal_adapters.graphs import cycle_basis
            actual_cycles = cycle_basis(graph, supplied_cycles, max_entries=max_entries)
            actual_cycles, basis_report = optimize_cycles(graph, actual_cycles,
                max_cycle_weight=cycle_weight_cap, max_cycle_congestion=cycle_cap,
                enabled=optimize_cycle_basis, max_roots=cycle_basis_max_roots,
                max_candidates=cycle_basis_max_candidates, max_entries=max_entries)
            if (int(actual_cycles.sum(axis=1).max(initial=0)) > cycle_weight_cap
                    or int(actual_cycles.sum(axis=0).max(initial=0)) > cycle_cap):
                raise ValueError("Sparsification needs layer-square weight 4, exceeding the fixed memory-based sparsity threshold")
            return graph, matching, actual_cycles, dict(applied=False, layers=1, added_chord_count=0,
                cycle_basis_optimization=basis_report)
        return sparsify_cycles(graph, matching=matching, cycles=supplied_cycles,
            max_cycle_weight=cycle_weight_cap, max_cycle_congestion=cycle_cap,
            max_layers=max_layers, max_entries=max_entries,
            optimize_cycle_basis=optimize_cycle_basis, cycle_basis_max_roots=cycle_basis_max_roots,
            cycle_basis_max_candidates=cycle_basis_max_candidates)

    # A later graph change can invalidate a direct search result, so every
    # rebuilt code is searched again. One additional repair pass is allowed.
    for round_index in range(2):
        cmd = current.metadata
        b, m, ports = cmd["graph_incidence"].copy(), cmd["matching_matrix"].copy(), list(cmd["ports"])
        cycles = cmd["cycle_basis"].copy()
        if not (np.all(b.sum(axis=1) == 2) and rank(b) == b.shape[1] - 1):
            b, m, conversion = paired_graph(source, target, max_entries=max_entries)
            ports = list(range(int(np.count_nonzero(target[0] | target[1]))))
            cycles = None
            converted = True
            candidate = build_graph(source, target, md["logical_basis"], b, ports=ports,
                matching=m, scheme=md["scheme"], extra_metadata=representative_metadata,
                exact_limit=exact_limit, max_entries=max_entries)
            current, _ = _checked_distance(candidate, source_report, target_distance,
                search_options=search_options)
            report["conversion"] = conversion
        if (int(m.sum(axis=1).max(initial=0)) > mmw
                or int(m.sum(axis=0).max(initial=0)) > max_matching_congestion):
            raise ValueError("Input matching exceeds requested sparsity weight/congestion limits; supply a sparser matching or adjust explicit limits")
        state = _distance_state(current.metadata["distance_calculation"], target_distance)
        expansion = dict(added_edge_count=0, reason="distance already certified; expansion not needed")
        needs_certificate = (state == "unknown"
            and not current.metadata["distance_preservation_guaranteed"]
            and unknown_distance_policy == "certify")
        if state == "insufficient" or needs_certificate:
            # Leave one unit of weight for the physical target at a port.
            degree_cap = min(max_base_degree, row_cap - 1)
            if degree_cap < 1:
                raise ValueError("The memory-based sparsity threshold leaves no degree budget for expansion")
            evaluated = {}

            def evaluate_candidate(graph, context):
                # Evaluate the code we would return, including full flux closure
                # and cycle sparsification. No graph certificate is computed here.
                candidate_matching = np.pad(m, ((0, 0), (0, len(graph) - len(b))))
                try:
                    fb, fm, fc, sparse_report = prepare_sparse(graph, candidate_matching, None)
                except ValueError as exc:
                    # A finite sparsification/memory budget failure is local to
                    # this candidate, so another seed may still fit the limits.
                    evaluated.clear()
                    return dict(eligible=False, distance_state="unknown",
                                reason="sparsification rejected candidate: " + str(exc))
                try:
                    candidate = build_graph(source, target, md["logical_basis"], fb, ports=ports,
                        matching=fm, cycles=fc, scheme=md["scheme"],
                        extra_metadata=dict(initial_construction=origin, **representative_metadata),
                        exact_limit=exact_limit, max_entries=max_entries, _defer_expansion=True)
                except ValueError as exc:
                    if "exceeds max_entries" not in str(exc):
                        raise  # Algebra, sign and readout failures are never ignored.
                    evaluated.clear()
                    return dict(eligible=False, distance_state="unknown",
                                reason="candidate matrix budget exceeded: " + str(exc))
                metrics = _metrics(candidate)
                failures = {key: dict(actual=metrics[key], limit=value)
                            for key, value in limits.items() if metrics[key] > value}
                if failures:
                    evaluated.clear()
                    return dict(eligible=False, distance_state="unknown",
                        reason="full deformed-code sparsity limits exceeded",
                        sparsity_failures=failures, final_metrics=metrics)
                checked, candidate_state = _checked_distance(candidate, source_report, target_distance,
                    search_options=search_options)
                evaluated.clear()
                evaluated.update(graph=graph.copy(), result=checked, sparsification=sparse_report)
                return dict(eligible=True, distance_state=candidate_state,
                    distance_report=checked.metadata["distance_calculation"],
                    final_n=checked.n, final_metrics=metrics)

            expanded, expansion = enhance_expansion(b, ports=ports, distance=target_distance,
                max_degree=degree_cap, seed=seed, attempts=attempts,
                exact_limit=exact_limit, certificate_limit=certificate_limit, max_entries=max_entries,
                allow_partial_overlay=True, strategy=expansion_strategy,
                candidate_evaluator=evaluate_candidate if expansion_strategy == "degree_incremental" else None)
            report["expansion_added_edges"] += expansion["added_edge_count"]
            report["expansion_trigger"] = (
                "distance certificate unavailable within search budget" if needs_certificate
                else "computed distance below target")
            if expansion.get("acceptance") == "distance":
                if not evaluated or not np.array_equal(evaluated["graph"], expanded):
                    raise RuntimeError("Expansion acceptance does not match the evaluated code")
                current = evaluated["result"]
                sparse = evaluated["sparsification"]
                report["rounds"].append(dict(expansion=expansion, sparsification=sparse,
                    distance_calculation=current.metadata["distance_calculation"]))
                report.update(expansion=expansion, sparsification=sparse,
                              tanner_to_paired_graph=converted)
                break
            if len(expanded) != len(b):
                m = np.pad(m, ((0, 0), (0, len(expanded) - len(b))))
                cycles = None
            b = expanded
        elif state == "unknown" and not current.metadata["distance_preservation_guaranteed"]:
            raise ValueError("Actual distance remains undetermined after graph conversion; increase distance search budget")
        final_b, final_m, final_cycles, sparse = prepare_sparse(b, m, cycles)
        extra = dict(initial_construction=origin, **representative_metadata)
        if (np.array_equal(final_b, cmd['graph_incidence'])
                and np.array_equal(final_m, cmd['matching_matrix'])
                and list(ports) == list(cmd['ports'])):
            # A basis-only change preserves the actual adapter graph blocks.
            # Keep their proof inputs; build_graph re-derives every certificate.
            # Dropping them would lose valid compositional distance bounds on
            # graphs too large for a standalone expansion certificate.
            for key in ('graph_design', 'component_adapter_distance_proof',
                        'adapter', 'adapter_distance_proof', 'factors', 'precheck',
                        'postprocessing_distance_proof'):
                if key in cmd:
                    extra[key] = copy.deepcopy(cmd[key])
            if sparse.get('cycle_basis_optimization', {}).get('changed') and any(
                    key in extra for key in ('adapter', 'graph_design')):
                extra['preserved_cycle_metadata'] = dict(
                    note='Cycle rows in adapter/graph_design describe the original connector construction, not the current check generators',
                    active_cycle_basis='metadata.cycle_basis',
                    active_cycle_check_rows='metadata.row_layout.cycles')
        base_certificate = certify_expansion(b, ports=ports, distance=target_distance,
            exact_limit=exact_limit, certificate_limit=certificate_limit, max_entries=max_entries)
        if base_certificate["condition_met"]:
            extra["postprocessing_distance_proof"] = dict(base_graph=b, base_ports=ports,
                distance=target_distance, exact_limit=exact_limit, certificate_limit=certificate_limit)
        rebuilt = build_graph(source, target, md["logical_basis"], final_b, ports=ports,
            matching=final_m, cycles=final_cycles, scheme=md["scheme"], extra_metadata=extra,
            exact_limit=exact_limit, max_entries=max_entries)
        current, state = _checked_distance(rebuilt, source_report, target_distance,
            search_options=search_options)
        report["rounds"].append(dict(expansion=expansion, sparsification=sparse,
                                     distance_calculation=current.metadata["distance_calculation"]))
        report.update(expansion=expansion, sparsification=sparse,
                      tanner_to_paired_graph=converted)
        if state != "insufficient":
            break
    report.update(changed=not np.array_equal(current.check_x, result.check_x)
                          or not np.array_equal(current.check_z, result.check_z),
                  action="distance certification and memory-threshold sparsification",
                  overhead_note="Deterministic finite-size cycle coloring; no polylogarithmic-overhead claim")
    return _annotated(current, report, limits)
