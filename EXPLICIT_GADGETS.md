# Explicit WSC gadgets, Universal adapters, and gauging measurement

The current implementation includes single-observable WSC Tanner gadgets,
SkipTree repetition adapters for disjoint products, and a separate
auxiliary-graph gauging framework. See [the full Chinese algorithm and
pseudocode](EXPLICIT_GADGETS_ZH.md).

References: [WSC 2025 II.1](https://arxiv.org/abs/2511.15989),
[Universal adapters, 2024](https://arxiv.org/abs/2410.03628), and
[Williamson--Yoder gauging, 2024](https://arxiv.org/abs/2410.02213).

```python
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform

single = gadget_deform(code, logical)
pair = gadget_pair_deform(code, left, right)
fallback = gauging_deform(code, logical)
fallback_pair = gauging_pair_deform(code, left, right)
```

All four Python APIs now default to `require_distance=True`: compute static distances, expand only when an actual logical witness proves a distance deficit, and enforce the initial horizontal dressed-memory row-weight threshold. Use `require_distance=False` for the basic construction, which adds no expansion edges. The older
`expansion='auto'/'star'/'complete'` options are opt-in implementation
heuristics, not edge-selection algorithms specified by WSC. They cannot
bypass the new strict raw-restriction checks.

Single-observable gadgets allow arbitrary even-weight restriction rows. The
kernel's Pauli-aligned logical signature must span precisely the target;
this support precheck alone may use weight<2d when a trusted distance bound
is supplied. Supported stabilizer kernel directions and disconnected Tanner
components are allowed. Auxiliary checks span the full ker(F.T), whose rank
is number_of_rows(F) - rank(F).

Only the two-factor SkipTree adapter route requires each nonzero F_i row to
have weight two and its ordinary graph to be connected. Failures raise
`GaugeMeasurementRequired` with a reason and details. Actual output
commutation, signed dependencies, witness, and full logical readout are
always validated.

“Pair” means measuring ONE observable left*right, with one +/-1 outcome;
it does not measure the two factors separately. The single-observable API
can also accept a logical product supplied as one complete Pauli string.

For pure X, let S select support vertices, T select touching Z checks, and N
span ker(F.T). The base code is

```text
HX_new = [[HX, 0], [S, F.T]]
HZ_new = [[HZ, T], [0, N]]
```

For a pair, SkipTree gives T_i F_i = H_C P_i.T on equal-sized connected
induced subgraphs. The assembled graph and cycle checks are

```text
B = [[F_l, 0], [P_l.T, P_r.T], [0, F_r]]
N = [[N_l, 0, 0], [T_l, H_C, T_r], [0, 0, N_r]]
M = [M_l, 0, M_r]
```

The connector cycles retain their sparse construction (weight at most eight,
new congestion at most two), rather than being replaced by arbitrary kernel
rows. Column order is memory / left edges / adapter edges / right edges.
The final graph measures only the product. Overlapping physical factors are
rejected by this adapter interface; supply the full Hermitian target to
`gauging_deform` instead. Logical Y is directly supported with `pauli='Y'`.

All targets use phase-aware symplectic data. Optional
`logical_basis={'jx': JX, 'jz': JZ}` must be a paired pure CSS basis; otherwise
the Zhang--Li standard form is used. Logical-coordinate targets accept
`{'logical_x': alpha, 'logical_z': beta}` with Hermitian convention
`i^(alpha.dot(beta)) Xbar(alpha) Zbar(beta)`. Omitting the target selects Z0;
omitting both pair factors selects Z0 and Z1, whose representatives still
must pass the physical-disjointness check.

Gauging accepts an explicit connected graph, ports, matching/dressing, and
cycle basis. The default graph pairs every even memory-check boundary and
joins connected components. That deterministic design guarantees algebraic
readout, not optimal overhead or distance. Both single and pair routes return
flux-fixed static deformed codes; protocol metadata is a skeleton, not a
compiled fault-tolerant circuit.

Ordinary connected graphs use relative expansion beta_d on the ports. Small
graphs are enumerated exactly. A pair may instead use Universal adapters
Lemma 9 if both branches are certified and the adapter joins at least d port
vertices. General single-observable Tanner matrices instead use parity-boundary
Cheeger h(F)>=1, without graph-connectivity shortcuts. Missing graph certificates do not prove a distance loss. Automatic mode additionally computes the actual source and output distances. A supplied source distance is treated as a trusted bound and rejected if a logical witness contradicts it. LDPC family guarantees additionally require uniformly bounded graph degree, matching length/congestion, and cycle length/congestion; arbitrary fundamental cycle bases may be dense. See [the distance and sparsity processor](DISTANCE_PRESERVATION_ZH.md).

The runnable repository example is [BB90](examples/bb90_compare.py), which
compares the original and weight-reduced representatives and their overhead.

CLI: `python -m codesurgery.explicit_gadgets input.json -o output.json`.
Provide your own `input.json` with the input matrices and target.
Set `scheme` to `explicit` (default) or `gauging`; use `logical` for one target
or `logicals: [left, right]` for their product. A module entry is also available
as `python -m codesurgery.gauging_measurement`.

## Automatic static distance and sparse checks

All four Python builders default to `require_distance=True`; pass `False` to retain the basic construction. `distance_options` controls finite search and construction budgets. `code_distance(code, max_weight=None, max_candidates=2_000_000)` and `stabilizer_distance(CX, CZ, ...)` are public in `codesurgery.distance_preservation`. They search the symplectic centralizer outside the stabilizer row space, minimizing the physical support `wt(x OR z)`. Reports distinguish exact distance, rigorous bounds, budget-limited unknowns, and codes without remaining logical qubits.

If the source distance is absent, automatic mode requires an exact computed source distance to select the target. A supplied distance remains a trusted lower bound; it is never mislabeled as a computed answer. A valid graph theorem may certify the target even when exact output search is incomplete, without populating the exact-distance field.

The default full row-weight threshold for pure Z targets is the maximum row weight of the **horizontal** initial block `[HX | M_X]`; pure X uses the dual block `[HZ | M_Z]`. For Y/mixed XZ it uses all memory rows as `tau = max_i(wt(CX_i OR CZ_i) + wt(M_i))`. The other CSS block cannot silently inflate a pure target’s threshold, but still must pass final full-matrix checks. The threshold is fixed before postprocessing. Cycle checks default to this same cap; column participation has a separate cap. `max_check_weight` is an explicit override. The code never silently raises the threshold to accommodate a heavier result.

An actual distance deficit triggers bounded-degree expansion; a graph expansion value below one alone does not. Overweight or congested cycle checks trigger layer decongestion and cellulation. Rebuilt results are checked for full row/column limits, commutation, signs, precise logical readout, and actual static distance. Unknown distance without a valid theorem, infeasible limits, and exhausted construction budgets raise clear errors. The finite-size coloring construction does not claim optimal polylogarithmic overhead or circuit fault distance.

Interactive CLI calls still ask when no choice was supplied. Use `--require-distance` / `--no-require-distance` (or JSON `require_distance`) to choose explicitly. Noninteractive calls default to the basic route and say so on stderr. Missing source distance is now computed when automatic processing is enabled; `--distance d` can still provide a trusted bound.

See [the distance and sparsity documentation](DISTANCE_PRESERVATION_ZH.md) for the API and construction examples. Their recorded small-code results illustrate that an expansion bound below one does not itself imply a distance deficit, while full-matrix sparsity limits can require cellulation independently of distance repair.
