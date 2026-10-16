# Static distance, graph expansion, and sparse checks

This English guide retains its existing filename for compatibility.
The four explicit/gauging builders default to `require_distance=True`.
Their shared processor searches or certifies a static distance threshold,
repairs or seeks a graph certificate when necessary, and checks fixed sparsity
limits. `require_distance=False` skips this entire processor and returns the
base construction.

The graph-design background is
[Williamson–Yoder's gauging construction](https://arxiv.org/abs/2410.02213),
the relative-expansion and adapter results are in
[Universal adapters](https://arxiv.org/abs/2410.03628), and the Tanner
parity-boundary condition appears in
[Cross–He–Rall–Yoder](https://arxiv.org/abs/2407.18393) and
[Webster–Smith–Cohen](https://arxiv.org/abs/2511.15989).
The finite-size heuristics and resource caps below are implementation
choices, not the papers' complete asymptotic constructions.

## API

```python
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.distance_preservation import code_distance, ensure_distance

basic = gauging_deform(code, logical, require_distance=False)
result = gauging_deform(
    code, logical,
    distance_options={
        "distance_method": "gurobi",
        "distance_time_limit": 60.0,
        "max_base_degree": 12,
        "max_layers": 128,
    },
)

# Apply the same processor to an existing base result.
processed = ensure_distance(basic, distance_max_candidates=2_000_000)
print(result.metadata["source_distance_calculation"])
print(result.metadata["distance_calculation"])
print(result.metadata["distance_postprocessing"]["sparsity_threshold"])
```

The Python API never waits for interactive input. `distance_options` only
controls automatic processing when it is enabled. Targets can include Y,
mixed XZ, and an overall minus sign. An omitted logical basis is obtained by
binary Gaussian elimination; a supplied paired CSS basis is validated.

The default distance backend is `"auto"`, preferring Gurobi when available.
The 60-second default is **per solver invocation**, not a time limit for the
entire surgery pipeline. Source checks, candidate graphs, rebuilding, and
final checks may make multiple calls. A solver timeout returns the available
bounds for that call; it does not then run an unrestricted cluster search.
The strict construction pipeline still needs a sufficient final threshold
bound or a valid graph certificate.

## Which distance is computed?

For commuting signed checks with binary symplectic stabilizer row space
`S`, the underlying stabilizer-code distance is

$$
d=\min_{(x,z)\in\mathcal S^{\perp_s}\setminus\mathcal S}
\operatorname{wt}(x\lor z).
$$

The Boolean OR counts a physical Y once. Checks are validated before
searching, and stabilizers are excluded by GF(2) row-space membership. The
minimum nonidentity stabilizer weight is not the code distance.

```python
from codesurgery.distance_preservation import code_distance, stabilizer_distance

report = code_distance(code, max_weight=None, max_candidates=2_000_000)
report = stabilizer_distance(CX, CZ, max_weight=None, max_candidates=2_000_000)

# With an enumerating backend, exhausting weights 1 and 2 proves d >= 3.
report = code_distance(result, method="enumeration", max_weight=2)
```

Enumeration and cluster backends proceed by weight. CSS distance can be
found by considering pure X and pure Z logical candidates; general Pauli
checks require the symplectic search. A partially searched weight layer
cannot establish exclusion of that entire layer. Heuristic searches supply
upper bounds through checked nontrivial logical witnesses, never lower
bounds merely because no witness was found.

Gurobi solves an integer optimization model and may supply a lower bound
before finding an exact optimum. Its bound is accepted according to the
solver's numerical tolerances; it is not an independently verified
exact-arithmetic proof. Returned logical witnesses are checked algebraically.
Graph certificates based on exact cut enumeration or integer arithmetic are
reported through a separate path.

| Report field | Meaning |
| --- | --- |
| `status="exact"` | The distance is determined; `distance` contains it. |
| `status="bounded"` | A reported interval has not closed; it can include an upper-bound-only heuristic result, and `distance=None`. |
| `status="unknown"` | No nontrivial lower bound has been established within the selected budget; any available upper bound is still reported. |
| `status="no_logicals"` | No logical qubit remains; `distance=None` and `distance_is_infinite=True`. |
| `proven_lower_bound` / `upper_bound` | Accepted lower bound and upper bound from a checked logical witness. |
| `distance_interval` / `distance_statement` | Integer interval and readable statement, such as `8 < d <= 10`. |
| `witness` | A logical representative's X/Z vectors, Pauli string, and weight. |
| `completed_weight` / `checked_candidates` | Enumeration progress; these are backend-specific. |

For example, `8 < d <= 10` does not establish `d >= 10`. An exact-distance
field is not filled from a graph theorem or an input declaration. For an
object with gauge qubits, this API computes the distance of its stabilizer
code; it is not automatically the protected subsystem's dressed distance.

### Selecting the target distance

1. If `code.distance` is absent, the source distance must be determined
   exactly before it can become the target. Otherwise processing raises an
   error suggesting a larger budget or a trusted input lower bound.
2. If `code.distance=d` is supplied, it is a trusted source lower bound.
   The processor searches for a contradiction; a logical witness of weight
   below `d` rejects the declaration. An incomplete search does not independently
   prove the input bound.
3. The output is checked against the target using its complete stabilizer
   matrices. A sufficient lower bound passes. A checked logical witness below
   the target establishes a deficit.
4. If search remains unresolved, an applicable graph theorem may supply the
   lower bound while the exact-distance field stays `None`. Without such a
   certificate, the default `unknown_distance_policy="certify"` continues
   bounded graph attempts to seek one. The recorded cause is a missing
   certificate, not a proved distance loss. `"error"` stops instead.

A successful automatic result therefore reaches the requested static
threshold under the stated input-bound and backend assumptions. It does not
by itself establish a circuit fault distance.

## Fixed row-weight threshold

For pure Z, the reference block is the initial **horizontal** dressed-memory
block `[HX | M_X]`:

$$
\tau=\max_i\left[\operatorname{wt}((H_X)_i)
+\operatorname{wt}((M_X)_i)\right].
$$

The two weights belong to the same row. With HX row weights `(4,8)` and
matching row weights `(3,0)`, the threshold is `max(7,8)=8`, not `8+3=11`.
Pure X uses the dual block `[HZ | M_Z]`. For mixed Y/XZ targets, all memory
rows contribute:

$$
\tau=\max_i\left[\operatorname{wt}((C_X)_i\lor(C_Z)_i)
+\operatorname{wt}(M_i)\right].
$$

For CSS input the memory rows are ordered as all HX rows followed by all HZ
rows. `sparsity_reference_rows` records those used to obtain the threshold.
The other CSS block cannot inflate a pure target's reference threshold, but
its rows must still satisfy the final full-matrix cap. If an unmodifiable
memory row exceeds that cap, construction raises an error.

The threshold is fixed **before expansion and sparsification**. By default,
every final check must satisfy

$$
\max_i\operatorname{wt}((\widetilde C_X)_i\lor(\widetilde C_Z)_i)
\le\tau.
$$

Cycle rows use the same cap. An explicit `max_cycle_weight` can tighten it;
`max_check_weight` overrides the full-row cap, while the original reference
value remains in the audit. No later graph change silently increases it.
A source with no memory checks has reference threshold zero and requires
an explicit feasible `max_check_weight`.

Column participation has a separate cap. Let `q0` be the maximum input
memory-qubit degree, `m_c` the matching-column cap, and `c_N` the cycle-column
cap. The default full-column cap is

$$
q_{\rm cap}=\max\{q_0+1,\;2+m_c+c_N\}.
$$

All limits are checked on the actual full Pauli support
`check_x | check_z`, as well as on `B`, `M`, and `N`. Cellulation alone need
not fix an overweight vertex check. Failure to meet a fixed limit raises
an error.

## Expansion: repair a deficit or seek a certificate

If direct distance verification already reaches the target, a graph value
below one alone does not trigger expansion. Otherwise a proved deficit, or
an unresolved bound without a certificate under the default policy, can
start the expansion search. Existing edges are retained; matching columns
on new edges are zero.

The default `expansion_strategy="degree_incremental"` tries increasing degree
caps, seeded random completions, and repairs guided by deficient cuts. Each
candidate is evaluated with the full rebuilt, sparsified code and its weight
limits. A sufficient direct distance bound can accept it immediately. When
acceptance relies on the expansion search's graph certificate, this strategy
requires strict global `h > 1`. This stronger stopping rule is an implementation
choice; the underlying static-distance sufficient condition permits equality
at one. The optional `"hamiltonian"` strategy retains an earlier construction.

The base degree is bounded by both `max_base_degree` and `row_cap - 1`,
leaving room for the physical target at a port. Attempts and matrix sizes
are bounded. Random-regular-graph intuition motivates the search but does
not certify an individual candidate. The code verifies its accepted distance
bound or graph certificate.

If a general even-row Tanner matrix needs graph processing, the processor
converts it to the separate gauging paired graph and checks the new code
again. This does not bypass the explicit route's original readout precheck
or the pair interface's graph requirements.

## Sparse cycle checks

### Optimize the cycle basis first

When the existing complete cycle generators already satisfy their row and
column caps, they are retained. Otherwise the default first tries alternative
bases of the same cycle space: cycles from multiple BFS trees, shorter
independent candidates, and edge-congestion-aware ordering. A replacement is
accepted only if it spans the full same kernel and meets both limits.
This changes no graph edge, port, matching, or auxiliary-qubit count.

The default search tries at most 64 roots and retains at most 10,000 distinct
candidates, also subject to `max_entries`. It is a bounded heuristic, not a
minimum-cycle-basis or minimum-congestion guarantee. Set
`optimize_cycle_basis=False` to skip it. This stage runs before layering on
each graph sent for sparsification; expansion may already have changed that
graph.

### Decongestion by layers

If basis optimization cannot meet both caps, a fundamental cycle basis is
colored so cycles sharing vertices use different layers. The construction
takes the Cartesian product of the base graph with a path, places each
basis cycle in one layer, and adds interlayer square checks. The number of
layers is limited by `max_layers` and the matrix budget.

This finite-size coloring is not the optimal Freedman–Hastings decongestion
algorithm and carries no claim of the paper's polylogarithmic layer bound.

### Cellulation of long cycles

Long simple cycles are split into triangles by alternating ear removal and
new chord edges. Each old vertex acquires at most two chords. A parallel
edge is retained as a separate auxiliary qubit when needed. Short cycles
are kept. Layering requires weight-four squares, so that route needs a cycle
cap of at least four. A smaller cap can work only if the original or optimized
basis already meets it without layering.

For base degree `Delta`, the implemented layered/chord construction has

$$
\Delta_{\rm final}\le\Delta+4,\qquad
\max\operatorname{rowwt}(N)\le\text{cycle cap},\qquad
\max\operatorname{colwt}(N)\le\max(3,\Delta).
$$

These intermediate graph bounds do not replace full-matrix checks. Every
rebuilt result receives a new distance check. A new deficit after rebuilding
allows one further bounded repair pass; failure then raises an error.

## Graph certificates and preserved ports

For an ordinary graph `B` and its physical port vertices `U`, use

$$
\beta_d(B,U)=
\min_{A:\min(d,|A\cap U|,|U\setminus A|)>0}
\frac{|\partial_B A|}{\min(d,|A\cap U|,|U\setminus A|)}.
$$

The sufficient condition is `beta_d >= 1`, with the other graph-construction
hypotheses satisfied. It need not be strictly greater than one. The final
graph with dummy vertices or layers need not have global `h >= 1`.
For connected adapted branches, the verified composition theorem can supply
a bound when each branch is certified and at least `d` port pairs are joined.

Small graph cuts are enumerated exactly. A separate sufficient certificate
for larger graphs uses integer Bareiss elimination to test positive
definiteness of a principal `(n-1) × (n-1)` submatrix of

$$
A=nL-2nI+2J,
$$

where `L` is the Laplacian and `J` is the all-ones matrix. Passing establishes
`lambda_2(L) > 2`, hence global edge expansion greater than one and the
required relative bound. Failure of this sufficient test is inconclusive.
Floating-point eigenvalues and sampled cuts are not used as proof of the
lower bound.

Layering retains the base graph and all ports in layer zero. Every final
cut includes the boundary of its restriction to that base graph, while the
relative-expansion denominator uses unchanged ports. A certified base bound
can therefore survive the extra layers. The implementation verifies the
embedded base, ports, and certificate against the actual matrices. A logical
witness contradicting the theorem causes an error rather than being ignored.

## Options

Pass these in `distance_options`, or directly to `ensure_distance`:

| Option | Default | Purpose |
| --- | --- | --- |
| `distance_method` | `"auto"` | Distance backend; prefer Gurobi when available. |
| `distance_time_limit` | `60.0` | Seconds per supported solver call. |
| `distance_max_candidates` | `2_000_000` | Enumeration candidate budget per search. |
| `distance_max_weight` | `None` | Optional search-weight cutoff. |
| `max_check_weight` | `None` | Initial dressed-memory threshold unless overridden. |
| `max_cycle_weight` | `None` | Full-row cap, or a tighter supplied cap. |
| `max_qubit_degree` | `None` | Full-column cap from the formula above. |
| `max_base_degree` | `12` | Expansion-stage degree cap, also constrained by row weight. |
| `max_cycle_congestion` | `None` | Uses `max(4, max_base_degree)`. |
| `optimize_cycle_basis` | `True` | Try an equivalent sparse basis before layering. |
| `cycle_basis_max_roots` | `64` | Maximum BFS roots. |
| `cycle_basis_max_candidates` | `10_000` | Maximum stored cycle candidates. |
| `unknown_distance_policy` | `"certify"` | Seek a missing certificate; `"error"` stops instead. |
| `max_layers` | `128` | Layer limit. |
| `max_matching_weight` | `None` | Initial maximum matching row weight. |
| `max_matching_congestion` | `1` | Matching column cap. |
| `seed` / `attempts` | `0` / `12` | Graph search seed and retry budget. |
| `distance_seed` / `distance_ris_trials` | `0` / `64` | Separate distance-witness search controls. |
| `exact_limit` | `20` | Vertex limit for exact graph-cut enumeration. |
| `certificate_limit` | `128` | Vertex limit for the integer spectral test. |
| `max_entries` | `50_000_000` | Dense matrix and related graph resource budget. |

A finite example meeting these caps is not by itself an LDPC-family proof.
Such a claim needs an LDPC input family and bounds uniform across that family.

## Command line

```bash
python -m codesurgery.gauging_measurement input.json --require-distance -o certified.json
python -m codesurgery.gauging_measurement input.json --no-require-distance -o basic.json
```

Supply the input checks and target in JSON. Interactive calls without a
choice ask whether to enable automatic processing; `y` enables it and `n`
or Enter selects the base route. A noninteractive call without a choice uses
the base route and prints a notice. A missing source distance is computed
when processing is enabled, rather than requested interactively.

The flags override JSON `require_distance`. `--distance d` provides a trusted
source bound but does not itself turn on processing. A JSON fragment is

```json
{
  "require_distance": true,
  "distance_options": {
    "distance_max_candidates": 2000000,
    "max_base_degree": 12
  }
}
```

Also include `hx`, `hz`, and `logical`, or two-factor `logicals`. The
[BB90 example](examples/bb90_compare.py) shows complete package usage.

## Processing pseudocode and audit

```text
BUILD(code, target_or_two_factors, require_distance=True, options=None):
    result = construct the explicit/gauging/adapter base code
    validate commutation, phases, readout, and logical dimension
    if not require_distance: return result

    search the source distance within its selected budget
    if no source distance was declared:
        require an exact finite source distance and use it as target d
    else:
        use the trusted declared lower bound d
        reject a checked logical witness below d

    fix tau from the initial horizontal dressed-memory reference block
    fix row, column, matching, cycle, and resource caps
    check the full output distance against d
    if distance and all sparsity limits pass: return the unchanged result
    if distance is unresolved with no graph certificate:
        "error": raise
        "certify": retain the unresolved report and seek a certificate

    convert general Tanner incidence to a gauging graph only if needed
    if a deficit is proved or a required certificate is still missing:
        attempt bounded degree-incremental expansion
        evaluate candidate full codes and record the actual trigger
    if cycle weight or congestion exceeds the cap:
        optimize an equivalent complete cycle basis first
        if that fails, use decongestion layers and cellulation

    rebuild signed checks and check the full distance again
    allow one further bounded repair pass for a new deficit
    validate NB=0, complete cycle space, signs, and exact requested readout
    check every full-matrix row and column against the fixed caps
    raise on insufficient certification, excess weight, or exhausted budget
    return the code with search reports and separate graph/sparsity audits
```

`metadata["distance_postprocessing"]` records the initial and final distance
reports, `sparsity_threshold`, effective row cap, `expansion_trigger`, added
edge count, layers, cellulation, and final full-matrix metrics.
`sparsification.cycle_basis_optimization` records budgets, candidate count,
truncation, basis changes, metrics, and span verification.

A basis-only change has `sparsification.applied=False` and `layers=1`, while
its nested `cycle_basis_optimization.changed=True`. Old cycle-row indices
under `adapter` or `graph_design` describe the original connector. The
current checks are identified by `metadata.cycle_basis` and
`metadata.row_layout.cycles`. Use the returned audit for the current build;
historical example counts are not a substitute for its matrices or certificate.
