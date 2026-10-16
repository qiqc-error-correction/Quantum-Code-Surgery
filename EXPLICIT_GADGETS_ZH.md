# Explicit gadgets and gauging measurement: matrices and pseudocode

This English guide retains its existing filename for compatibility.
It describes `codesurgery.explicit_gadgets`, `universal_adapters`, and
`gauging_measurement`. CKBB and devised sticking have separate implementations.

The explicit single-observable gadget follows
[Webster–Smith–Cohen, Section II.1](https://arxiv.org/html/2511.15989v1#S2.SS1).
The Tanner boundary-expansion framework is developed in
[Cross–He–Rall–Yoder](https://arxiv.org/abs/2407.18393).
The graph adapter follows [Swaroop–Jochym-O'Connor–Yoder](https://arxiv.org/abs/2410.03628),
with Lemmas 9 and 10 numbered as in version 4. The alternative auxiliary-graph
construction follows [Williamson–Yoder](https://arxiv.org/abs/2410.02213).
The implemented route is not WSC's full universal architecture based on
fixed seed gadgets, automorphism routing, and overlapping-gadget merges.

## API and targets

```python
from codesurgery import CSSCode
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform

code = CSSCode(HX, HZ, distance=d)
single = gadget_deform(code, logical=L)
joint = gadget_pair_deform(code, left=L1, right=L2)
alternative = gauging_deform(code, logical=L)
```

`L`, `L1`, and `L2` are physical Pauli representatives of length `code.n`.
The pair interface measures **one product** `L1 * L2`, with projectors
`(I ± L1 L2)/2`, rather than measuring its factors separately. A single target
may itself be a logical product, such as X on one encoded qubit and Z on
another.

All four builders default to `require_distance=True`. This invokes combined
distance certification, expansion when needed, and sparsity processing.
`require_distance=False` returns the algebraic base construction without that
processor. The legacy `expansion="auto"/"star"/"complete"` options are explicit
implementation heuristics, not WSC's specified edge-selection algorithm;
they cannot bypass input readout checks.

An optional `logical_basis={"jx": JX, "jz": JZ}` supplies paired pure CSS
logical generators. The code checks their normalizer conditions and
`JX @ JZ.T = I` over GF(2). If omitted, binary Gaussian elimination provides
a basis in the original physical column order. Targets can be signed IXYZ
strings, physical `{x, z, sign}` dictionaries, binary vectors with `pauli`,
or coefficient dictionaries such as

```python
import numpy as np

# X on logical qubit 0 and Z on logical qubit 1; requires code.k >= 2.
alpha = np.zeros(code.k, dtype=np.uint8)
beta = np.zeros(code.k, dtype=np.uint8)
alpha[0] = beta[1] = 1
logical = {"logical_x": alpha, "logical_z": beta}
```

Logical coefficients use the Hermitian convention
`i^(alpha.dot(beta)) Xbar(alpha) Zbar(beta)`. In particular, logical Y is
`i Xbar_j Zbar_j`; its physical string may carry a minus sign. For example,
`i XXX ZZZ = -YYY`. `check_signs` and `measurement_sign` preserve these signs.
Python logical indices start at zero. With no target, the single route
selects Z0; the pair selects Z0 and Z1. Their representatives must still pass
the pair route's physical-disjointness check.

The single-target builders default to `optimize_logical=False`. Enable it to try
stabilizer-based weight reduction before construction, or use
`compare_deformations` to build both representatives and compare overhead.
Supplied graph blocks must match the representative actually used.
For the native pair interfaces, reduce factors separately and recheck their
physical disjointness before constructing the adapter.

## Extracting and checking F

Write all memory checks as `C = (CX | CZ)`, with X rows followed by Z rows
for a CSS input. A target `(x | z)` has support `V` of size `w`, and `S` is
the `w × n` selector for that support. All matrix arithmetic below is over
GF(2). Define

```text
F_full[j,u] = CX[j,V[u]] z[V[u]] + CZ[j,V[u]] x[V[u]]
F = all nonzero rows of F_full
```

No row-basis reduction is applied to `F`: repeated restriction rows correspond
to different auxiliary qubits. For pure X, `F` is the nonzero restriction of
`HZ`; for pure Z it is the nonzero restriction of `HX`.

The single-observable explicit route checks:

1. The target is a nontrivial logical Pauli. Its commutation with memory
   checks makes every nonzero `F` row even; the row need not have weight two.
2. If a trusted source-distance lower bound `d` is available and `w < 2d`,
   the Pauli-aligned kernel precheck can be skipped.
3. Otherwise, let the rows of `K` span `ker(F)`. Form
   `QX = (K S) diag(x)` and `QZ = (K S) diag(z)`. Their logical-signature space
   must be exactly the one-dimensional target space. Extra independent
   logical information raises `GaugeMeasurementRequired`.
4. Use a full row basis `N` of `ker(F.T)`, of rank
   `F.shape[0] - rank(F)`. Kernel directions that represent source stabilizers
   are allowed. Ordinary-graph connectivity is not required for this route.

The `w < 2d` shortcut only concerns suboperators aligned with the target's
local Pauli. If such a suboperator represented a logical class independent
of the target, it and its complementary suboperator would be nontrivial
logicals on disjoint supports, implying `w >= 2d`. Supported stabilizers are
not excluded by this argument. Equality `w = 2d` does not qualify.

The shortcut never skips final commutation, phases, full readout validation,
or the adapter's graph conditions. It is not a distance-preservation proof.

## Base matrices for general Pauli targets

For pure X, let `T` map the rows of `F` to the original Z checks. Then

```text
HX_new = [ HX   0  ]       HZ_new = [ HZ  T ]
         [ S   F.T ]                [ 0   N ]
```

The general signed symplectic construction uses `B` with auxiliary-qubit
rows and vertex columns, a vertex-to-memory port selector `S`, and a memory
check dressing matrix `M`. Ordinary graphs have weight-two rows of `B`;
the single explicit route also permits more general even rows.

```text
M B = (CX diag(z) + CZ diag(x)) S.T
N B = 0,    rank(N) = number_of_rows(B) - rank(B)

check_x = [ CX         0  ]       check_z = [ CZ         M ]
          [ S diag(x) B.T ]                 [ S diag(z) 0 ]
          [ 0          0  ]                 [ 0          N ]
```

Every memory support position is assigned to one port; dummy vertices may
have no memory qubit. Multiplying all vertex checks produces the target.
Signs are solved separately. For a pure Z target, the implementation applies
the auxiliary Hadamard dual convention so that the final code is CSS.
For Y or mixed XZ targets, the two halves of a row in
`check_x/check_z/check_signs` describe one Pauli check.

## Two-factor products: SkipTree repetition adapter

Each factor first passes its own logical-readout check. The implemented
adapter additionally requires disjoint physical factor supports and connected
ordinary graphs `F_l`, `F_r`, whose nonzero rows all have weight two. These
are restrictions of this interface; the cited adapter paper treats broader
compatible-overlap cases. A logical Y should be supplied as its complete
Hermitian target, for example with `pauli="Y"`, rather than as the bare
product of anticommuting X and Z factors.

There is one copy of the memory checks. A memory check can be dressed on
both auxiliary graphs. Select `A` vertices on each side, each inducing a
connected subgraph. The default is `A = min(|V_l|, |V_r|)`, selected through
a BFS prefix; `adapter_size` and `adapter_vertices` can override this choice.

SkipTree supplies `T_l`, `T_r`, `P_l`, `P_r`, and a cyclic repetition check
matrix `H_C` satisfying

```text
T_l F_l = H_C P_l.T,      T_r F_r = H_C P_r.T.
```

Each `T` has row weight at most three and column weight at most two.
`P_i` has shape `|V_i| × A` and selects the labeled vertices. Connect matching
labels with `A` new edges, each holding one auxiliary qubit. The assembled
blocks are

```text
B = [ F_l    0   ]       N = [ N_l   0    0   ]
    [ P_l.T  P_r.T ]        [ T_l  H_C  T_r ]
    [ 0      F_r ]          [ 0     0    N_r ]

M_joint = [ M_l   0_A   M_r ]
```

Physical column order is `(memory, left_edges, adapter_edges, right_edges)`.
The added connector cycle checks have weight at most eight and add at most
two uses of an old edge. There are `A - 1` independent connector cycles;
the displayed cyclic generators may be redundant. For `A = 1`, the zero
connector row is omitted. Sparse generators are retained rather than
replaced by an arbitrary reduced kernel basis.

The final full check group must read out only `L_l L_r`, not either factor
separately. An inadequate adapter size can still reduce distance; SkipTree's
sparsity bound alone is not a distance guarantee.

## Construction pseudocode

```text
EXPLICIT_BASE(code, one_target_or_two_factors, optional_basis):
    basis = supplied paired CSS basis, or compute one by Gaussian elimination
    parse signed targets and check nontrivial normalizer membership

    for each factor L_i:
        extract all nonzero restriction rows F_i
        if d is unknown or wt(L_i) >= 2d:
            K_i = a row basis of ker(F_i)
            Q_i = K_i S_i with the local Paulis of L_i
            require logical_signature_space(Q_i) == span(signature(L_i))
            otherwise raise GaugeMeasurementRequired

    if there is one target:
        B = F_1
        M = selection of original checks associated with F_1 rows
        N = a full row basis of ker(B.T)
        target = L_1
    else:
        require disjoint physical factor supports
        require each F_i to have weight-two rows and connected graph
        otherwise raise GaugeMeasurementRequired
        select equal-sized connected induced vertex sets
        run SkipTree and assemble B, M_joint, N as above
        target = signed Hermitian product L_1 L_2

    assemble memory, vertex, and cycle symplectic checks
    solve signs, excluding -I dependencies and matching the target product
    validate commutation, signed readout, and remaining logical dimension k-1
    report the applicable boundary/relative expansion certificate
    return the sealed base deformed code

VALIDATE_ACTUAL_READOUT(CX_new, CZ_new, target, basis):
    A = [CX_new[:,aux] | CZ_new[:,aux]]
    U = a row basis of ker(A.T)
    RX = U CX_new[:,memory]
    RZ = U CZ_new[:,memory]
    Gamma = row_basis([RX JZ.T | RZ JX.T])
    require rowspace(Gamma) == span(signature(target))
    otherwise raise an unintended-logical-readout error
    separately validate phases and the signed target witness
```

The public builders apply the distance processor after this base stage
unless `require_distance=False`. Validation also runs when the result is
validated or saved. Changing matrices or certificates invalidates the stored
snapshot.

## The separate gauging route

```python
result = gauging_deform(code, logical=L)
result = gauging_deform(
    code, logical=L, graph=B, ports=ports, matching=M, cycle_basis=N,
)
pair = gauging_pair_deform(code, L1, L2, graphs=(B1, B2))
```

Without a supplied graph, each check's even anticommuting boundary is paired
in sorted order, adding one edge for each pair. `M` records which edges dress
each memory check. Disconnected components are joined in a chain of
**repetition-code adapters** by default (`component_connection="adapter"`).
Each link joins two adjacent original components, with default size equal to
the smaller component. `component_adapter_size` overrides the size;
`component_adapter_shift=None` selects a cyclic rotation minimizing the
heaviest connector check. Components representing stabilizers or dependent
logical signatures are retained so that their product remains the target.

`component_connection="chain"` instead selects one edge between successive
component representatives. This simpler option is not the default adapter
construction. Both choices ensure connectedness and correct readout; neither
alone establishes expansion, distance, or optimal overhead. Component options
apply only to automatically constructed graphs; a supplied graph is validated
as supplied.

```text
GAUGING_BASE(code, target, optional_graph):
    parse target and paired basis
    if graph was not supplied:
        pair each memory check's even local-anticommutation boundary
        join adjacent components with repetition-code adapters by default
        use single component-link edges only if component_connection="chain"
        record the dressing M
    else:
        require a connected graph with weight-two incidence rows
        require an injective support-to-vertex port assignment
        solve or verify M B = the port anticommutation boundary
    construct or validate complete cycle generators N
    assemble the signed symplectic checks
    validate commutation, full readout, signs, and logical dimension
    return the base deformed code and measurement-stage metadata

GAUGING_PAIR_BASE(code, L1, L2):
    construct the two auxiliary graphs and their dressings
    require disjoint physical supports
    join them with the same SkipTree adapter
    validate the final single-product readout
```

The result is a static deformed stabilizer code with a complete set of cycle
checks. It is not a general noncommuting gauge-generator measurement engine.
The metadata describes initialization and measurement stages, without a
compiled noisy syndrome circuit.

## Distance and LDPC conditions

For an ordinary graph with ports `U`, the relative expansion used is

```text
beta_d(B,U) = min wt(Bv) / min(d, |v intersect U|, |U|-|v intersect U|),
```

over subsets with a nonzero denominator. A lower bound of one suffices under
the graph construction's other hypotheses. Small graphs are checked by exact
cut enumeration. For an adapter, Lemma 9 can combine two certified branch
bounds when the selected vertices are ports and `A >= d`.

For a general single-target Tanner matrix, the relevant parity-boundary
constant is `h(F) = min wt(Fv)/wt(v)` for `0 < wt(v) <= w/2`. Here boundary
means odd incidence. `h(F) >= 1` is the cited sufficient condition; ordinary
graph connectivity arguments cannot replace it.

Failure of these sufficient conditions does not prove distance loss. The
automatic processor searches the actual full stabilizer code as well. A
supplied source distance is a trusted lower bound, rejected if contradicted
by a logical witness. With no supplied distance, an exact finite source
distance is required to choose the target. The output need only have a
valid threshold lower bound; its exact distance may remain unknown.

By default, unresolved output distance without a graph certificate can
trigger bounded expansion attempts (`unknown_distance_policy="certify"`).
This is recorded as a missing certificate, separately from an actual deficit.
Use `unknown_distance_policy="error"` to stop in that case. Overweight or
congested cycle checks first prompt cycle-basis optimization, then layering
and cellulation if needed. Final rebuilt matrices are searched and checked
again. Full row and column caps remain fixed throughout.

Finite row/column bounds alone do not prove an asymptotic LDPC family, and
static distance does not establish circuit fault distance. See the
[distance and sparsity guide](DISTANCE_PRESERVATION_ZH.md) for the exact
thresholds, search reports, and finite-resource behavior.

## A structural example with a weight-four row

```text
HX = [[1,1,1,1,0], [1,1,0,0,0], [1,0,1,0,0]]
HZ = an empty (0,5) matrix
L  = ZZZZI
F  = [[1,1,1,1], [1,1,0,0], [1,0,1,0]]
```

`ker(F) = span(1111)`, so this target passes the single-route support check
without removing the weight-four row. The base construction has eight
physical qubits and one remaining logical qubit. Its `h(F) = 1/2` does not
provide the sufficient expansion certificate. In contrast, using this factor
in the two-factor adapter route fails the weight-two condition. This
difference concerns the construction route, not whether a Pauli target is
written as a product.

## Command line

```bash
python -m codesurgery.explicit_gadgets input.json --require-distance -o deformed.json
python -m codesurgery.gauging_measurement input.json --no-require-distance -o basic.json
```

Supply `hx`, `hz`, and `logical`, or `logicals: [L1, L2]` for one product.
The explicit-gadgets entry point accepts `scheme="explicit"` or `"gauging"`.
An interactive terminal asks whether to enable automatic processing if no
choice is provided. A noninteractive call defaults to the basic route and
prints a notice. `--distance d` supplies a trusted bound but does not itself
enable processing. The [BB90 comparison example](examples/bb90_compare.py)
shows package usage with user-selected measurements.
