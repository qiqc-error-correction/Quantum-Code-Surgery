"""WSC Tanner restriction gadgets, optionally connected by a universal adapter.

First construct the paper gadget, then by default compute distances and
enforce the dressed-memory row-weight threshold. Explicit expansion options
are legacy, opt-in heuristics, applied only AFTER strict input checks.
"""
import numpy as np
from ..gadget_common import (prepare, prepare_pair, inspect_restriction,
                            GaugeMeasurementRequired)
from ..gauging_measurement.api import build_graph, join_graph_gadgets
from .expansion import expand_restriction
from ..distance_preservation.api import finish_distance


def gadget_deform(code, logical=None, *, pauli="Z", logical_index=0,
                  logical_basis=None, expansion="none", extra_rows=None,
                  cycle_basis=None, exact_limit=20, max_entries=50_000_000,
                  require_distance=True, distance_options=None,
                  optimize_logical=False, logical_optimization_options=None):
    """One observable. Allow arbitrary even F rows; reject extra logical readout.

    Only the Pauli-aligned support precheck may use the weight<2d shortcut.
    Actual final readout, phases, Tanner structure and commutation are checked
    unconditionally. Default postprocessing computes source and output
    distances; use require_distance=False to return the base construction.
    optimize_logical=True first searches for a lighter representative in its
    stabilizer coset only when weight>2*code.distance by default. Unknown
    distance skips this rule; explicit options can override the threshold.
    Supplied graph blocks must refer to the optimized representative;
    for manually designed blocks, reduce_logical_weight first is preferable.
    """
    source, target, basis = prepare(code, logical, pauli=pauli, logical_index=logical_index,
                                    logical_basis=logical_basis, max_entries=max_entries)
    if not isinstance(optimize_logical, (bool, np.bool_)):
        raise ValueError("optimize_logical must be a Boolean")
    if logical_optimization_options is not None and not isinstance(logical_optimization_options, dict):
        raise ValueError("logical_optimization_options must be a dictionary")
    if logical_optimization_options is not None and not optimize_logical:
        raise ValueError("logical_optimization_options requires optimize_logical=True")
    optimization = None
    if optimize_logical:
        from ..logical_optimization import optimize_prepared_logical
        target, optimization = optimize_prepared_logical(source, target, basis,
                                                        **(logical_optimization_options or {}))
    info = inspect_restriction(source, target, basis, max_entries=max_entries)
    raw = info["restriction_matrix"]
    graph, report = expand_restriction(raw, mode=expansion, extra_rows=extra_rows,
                                       exact_limit=exact_limit, max_entries=max_entries)
    matching = np.pad(info["pasting_matrix"], ((0, 0), (0, len(graph) - len(raw))))
    metadata = dict(raw_restriction_matrix=raw, restriction_matrix=graph,
                    pasting_matrix=matching, touching_memory_check_rows=info["touching_memory_check_rows"],
                    precheck=info["precheck"], expansion=report,
                    cheeger_constant=report["h"], cheeger_lower_bound=report["h_lower_bound"],
                    expansion_certified=report["distance_condition_met"],
                    added_edge_qubits=len(graph) - len(raw),
                    expansion_provenance="none" if expansion == "none" else
                    "user-selected implementation heuristic; not a WSC edge-selection algorithm")
    if optimization is not None:
        metadata["logical_optimization"] = optimization
    result = build_graph(source, target, basis, graph, matching=matching, cycles=cycle_basis,
        scheme="webster_smith_cohen_explicit_gadget", incidence_kind="even_tanner", extra_metadata=metadata,
        exact_limit=exact_limit, max_entries=max_entries)
    return finish_distance(result, require_distance=require_distance, distance_options=distance_options,
                           exact_limit=exact_limit, max_entries=max_entries)


def gadget_pair_deform(code, left=None, right=None, *, paulis=("Z", "Z"),
                       logical_indices=(0, 1), logical_basis=None, adapter_size=None,
                       adapter_vertices=None, adapter_shift=0, cycle_bases=None,
                       exact_limit=20, max_entries=50_000_000,
                       require_distance=True, distance_options=None):
    """Measure only left*right using two strict F graphs and SkipTree adapter.

    The factors must have disjoint physical supports. Logical Y is also
    available as a single target through gadget_deform(pauli='Y').
    adapter_shift chooses a cyclic origin for the right SkipTree labels;
    it can reduce connector check weights without adding adapter qubits.
    """
    source, l, r, target, basis = prepare_pair(code, left, right, paulis=paulis,
        logical_indices=logical_indices, logical_basis=logical_basis, max_entries=max_entries)
    if cycle_bases is not None and len(cycle_bases) != 2:
        raise ValueError("cycle_bases must contain two entries")
    if adapter_vertices is not None and len(adapter_vertices) != 2:
        raise ValueError("adapter_vertices must contain two entries")
    infos = [inspect_restriction(source, factor, basis, require_graph=True, max_entries=max_entries) for factor in (l, r)]
    specs = [dict(graph=info["restriction_matrix"], matching=info["pasting_matrix"],
                  ports=list(range(len(info["support"]))),
                  cycles=None if cycle_bases is None else cycle_bases[i]) for i, info in enumerate(infos)]
    result = join_graph_gadgets(source, (l, r), target, basis, specs,
        scheme="webster_smith_cohen_with_universal_adapter", adapter_size=adapter_size,
        adapter_vertices=adapter_vertices, adapter_shift=adapter_shift, extra_metadata=dict(factors=infos),
        exact_limit=exact_limit, max_entries=max_entries)
    return finish_distance(result, require_distance=require_distance, distance_options=distance_options,
                           exact_limit=exact_limit, max_entries=max_entries)
