# Explicit gadgets, Universal adapters, and gauging measurement

These modules construct deformed stabilizer check matrices for a requested
logical Pauli measurement. For the full matrices and pseudocode, see the
[construction guide](EXPLICIT_GADGETS_ZH.md); for automatic distance and
sparsity processing, see the [distance guide](DISTANCE_PRESERVATION_ZH.md).
Both guides are in English; the existing filenames are retained for links.

The single-observable explicit construction follows Section II.1 of
Webster, Smith, and Cohen's
[*Explicit construction of low-overhead gadgets for gates on quantum LDPC codes*](https://arxiv.org/abs/2511.15989).
Its Tanner-graph expansion background is the framework of Cross, He, Rall,
and Yoder's
[*Improved QLDPC Surgery: Logical Measurements and Bridging Codes*](https://arxiv.org/abs/2407.18393).
The two-factor route uses SkipTree from
[*Universal adapters between quantum LDPC codes*](https://arxiv.org/abs/2410.03628),
while a separate graph-construction route implements
[*Low-overhead fault-tolerant quantum computation by gauging logical operators*](https://arxiv.org/abs/2410.02213).
The package does not implement WSC's complete fixed architecture with seed
operators, automorphism routing, and all overlapping-gadget merge rules.

```python
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform

single = gadget_deform(code, logical)
pair = gadget_pair_deform(code, left, right)
alternative = gauging_deform(code, logical)
alternative_pair = gauging_pair_deform(code, left, right)
```

All four APIs default to `require_distance=True`. They search or certify a
static distance threshold and enforce fixed row/column limits. Expansion may
be attempted after a logical witness establishes a deficit, or when the
default `unknown_distance_policy="certify"` seeks a missing lower-bound
certificate. An unresolved search is reported as unresolved, never as a
proved distance loss. `require_distance=False` skips this entire combined
distance/expansion/sparsification processor and returns the base construction.

The single-target builders accept `optimize_logical=True` to reduce the target
within its signed stabilizer coset before construction; their default is
`False`. For a native pair call, reduce the factors beforehand and recheck
physical disjointness. The comparison interface can construct both versions
of a target and compare their actual ancilla overhead. The repository
[BB90 example](examples/bb90_compare.py) demonstrates that interface.

## Single observables and joint products

A single target can contain X, Y, and Z, or be a product of logical operators.
The explicit route accepts arbitrary even-weight nonzero rows of the
restricted matrix `F`. It rejects unintended independent logical readout.
Supported stabilizer directions in `ker(F)` are allowed. Auxiliary checks
span all of `ker(F.T)`; their rank is `F.shape[0] - rank(F)`.

Only the two-factor adapter route requires each nonzero `F_i` row to have
weight two and each ordinary graph to be connected. The implemented pair
interfaces also require disjoint physical factor supports. Failures raise
`GaugeMeasurementRequired` with details. For a target that does not fit this
route, supply its complete Hermitian Pauli representative to
`gauging_deform`.

**A pair measures one observable `left * right`, with one ±1 outcome.** It
does not return separate outcomes for the two factors. The final readout
space, commutation, signs, and remaining logical dimension are checked.

For a pure X target, the base CSS matrices are

```text
HX_new = [[HX, 0], [S, F.T]]
HZ_new = [[HZ, T], [0, N]]
```

Here `S` selects the target support, `T` maps restricted rows to their original
Z checks, and `N` spans `ker(F.T)`. General Pauli targets use the full signed
symplectic representation described in the construction guide.

An optional `logical_basis={"jx": JX, "jz": JZ}` supplies a paired CSS basis.
Otherwise the basis is obtained by binary Gaussian elimination. Targets may
be physical Pauli strings or logical coefficients
`{"logical_x": alpha, "logical_z": beta}`. Logical Y uses the Hermitian
convention `i^(alpha.dot(beta)) Xbar(alpha) Zbar(beta)`; physical signs are
tracked separately.

The automatic gauging graph pairs each check's even anticommutation boundary.
Disconnected components are joined with repetition-code adapters by default
(`component_connection="adapter"`); `"chain"` selects single-edge component
links instead. These connections ensure one product readout, with distance
checked separately.

## What the certificates mean

Graph expansion supplies a sufficient static-distance condition. A failed
or unavailable graph certificate does not prove that the code distance has
decreased. The output may instead pass a direct distance-threshold search.
Exact distances, search bounds, and graph-theorem bounds are recorded
separately. Gurobi bounds depend on the solver's numerical tolerances;
exhaustive and integer graph certificates have different verification paths.

The default row cap is fixed from the initial horizontal dressed-memory
block: `[HX | M_X]` for pure Z, `[HZ | M_Z]` for pure X, and all dressed
symplectic memory rows for mixed targets. It is not increased after adding
edges. Column participation has a separate cap. Cycle-basis optimization is
attempted before layering; longer or congested cycle checks may require
decongestion and cellulation. These are finite-size implementation choices,
not a claim to reproduce the papers' optimal asymptotic overhead.

Successful automatic processing certifies the requested static threshold
and the configured full-matrix weight limits, conditional on a trusted
source-distance bound when one is supplied. This does not establish a
circuit fault distance or an LDPC family from a single finite instance.
Protocol metadata describes the measurement stages; it is not a compiled
fault-tolerant syndrome circuit.

## Command line

```bash
python -m codesurgery.explicit_gadgets input.json -o output.json
python -m codesurgery.gauging_measurement input.json --require-distance -o output.json
```

The JSON contains `hx`, `hz`, and either `logical` or `logicals: [left, right]`.
The latter means their product. Use `scheme="explicit"` or `"gauging"` with
the explicit-gadgets entry point. `--require-distance` and
`--no-require-distance` select automatic processing explicitly. Without a
choice, an interactive terminal asks; a noninteractive call uses the basic
route and prints a notice. See the detailed guides for options and limits.
