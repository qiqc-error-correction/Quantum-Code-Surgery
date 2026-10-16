# Brute-force branching: separating logical supports

Brute-force branching and branch stickers were introduced by Guo Zhang and Ying Li in [Time-Efficient Logical Operations on Quantum Low-Density Parity Check Codes](https://arxiv.org/abs/2408.01339), Sections III–V and Appendices C, D, and F. This module constructs short branch stickers and combines them into a binary tree. For targets with locally compatible Pauli axes, it also implements the extension described by Cowtan et al. in [Parallel Logical Measurements via Quantum Code Surgery](https://arxiv.org/abs/2503.05003), Section 2.2, especially Lemma 2.10.

By default, it branches only the targets needed to satisfy the configured weight limits. Other targets remain on memory. Set `branching_mode="always"` to use the full balanced tree. This selection policy and its finite-size cost search are package optimizations.

**Branching preserves logical information; it does not measure the targets.** The returned code retains all $k$ logical qubits. Measurement requires a subsequent gauging or other measurement construction. A separate preview of possible measurement matrices is returned to check weight growth.

All binary arithmetic is over $\mathbb F_2$. Checks are rows and qubits are columns. Pauli signs are stored separately. Newly added checks use a fixed $+1$ reference sector; an executed protocol would need measurement outcomes and Pauli-frame updates.

## Inputs and logical basis

```python
from codesurgery.parallel_branching import branching_deform

result = branching_deform(
    code,
    targets=None,
    logical_paulis=None,
    pauli="Z",
    logical_indices=None,
    logical_basis=None,
    representatives="auto",
    optimize_logical=True,
    logical_optimization_options=None,
    leaf_pauli="auto",
    force_copy=False,
    branching_mode="adaptive",
    max_check_increase=2,
    max_qubit_increase=2,
    max_entries=50_000_000,
)
```

`code` is a `CSSCode`. Choose one target input mode:

| Input | Meaning | Example |
| --- | --- | --- |
| `targets` | Signed physical Pauli strings of length $n$ | `["ZZII", "ZIZI"]` |
| `logical_paulis` | Pauli strings of length $k$, in the selected logical basis | `["YI", "IX"]` |
| `pauli` and `logical_indices` | The same Pauli type on selected logical qubits, with zero-based indices | `pauli="Z", logical_indices=[0, 2]` |

If targets are omitted, all logical $Z$ generators in the selected basis are used. If the basis is omitted, the package obtains a paired CSS logical basis by Gaussian elimination. A custom basis can be supplied as `logical_basis={"jx": ..., "jz": ...}`. It must contain $k$ pure $X$/$Z$ pairs satisfying

$$
J_XH_Z^T=0,\qquad J_ZH_X^T=0,\qquad J_XJ_Z^T=I_k.
$$

Basis computation selects logical representatives; it does not replace original check rows. Redundant input checks are retained.

## Local compatibility and signs

Simultaneous branching requires **qubit-wise commutation (QWC)**: at every physical qubit, any two targets either use the same nonidentity Pauli or at least one uses $I$. Global commutation alone is insufficient.

The initial representative policy is:

- `representatives="preserve"`: keep physical representatives and signs, and reject them if they are not QWC.
- `representatives="auto"`: keep already compatible representatives; otherwise reconstruct representatives in the same logical classes and selected basis, then check QWC again.
- `representatives="canonical"`: always reconstruct representatives in the selected basis, then check QWC.

Support reduction is controlled separately by `optimize_logical`. To retain physical representatives exactly, use both `representatives="preserve"` and `optimize_logical=False`.

Reconstruction does not change the user's basis and cannot make every globally commuting target set locally compatible. If it fails, supply suitable stabilizer-equivalent representatives or a different verified basis.

For a physical target $(x\mid z)$, its logical coefficients are

$$
a=xJ_Z^T,\qquad b=zJ_X^T.
$$

The reconstructed binary representative is $x_c=aJ_X,\ z_c=bJ_Z$. A stabilizer witness $w$ solves

$$
wH_{\mathrm{symp}}=(x+x_c\mid z+z_c),
\qquad
H_{\mathrm{symp}}=
\begin{pmatrix}H_X&0\\0&H_Z\end{pmatrix}.
$$

Full Pauli multiplication then determines the sign, ensuring

$$
P_{\mathrm{prepared}}
=P_{\mathrm{original}}\prod_{j:w_j=1}S_j.
$$

Source-check order is all $H_X$ rows followed by all $H_Z$ rows. Binary row-space membership alone does not determine the sign.

For logical strings, use $P(x,z)=i^{x\cdot z}X(x)Z(z)$, with exponents evaluated as integers modulo four. The phase of logical $i^{a\cdot b}\bar X(a)\bar Z(b)$ relative to physical $P(aJ_X,bJ_Z)$ is

$$
i^{a\cdot b-(aJ_X)\cdot(bJ_Z)}.
$$

A valid paired CSS basis makes this phase real.

Every target must be a nontrivial normalizer, and targets must commute. Repeated or linearly dependent targets are allowed; their independent logical rank is recorded. A target's support may contain other logical information because branching does not read out logical observables.

## Reduce support before branching

The default sequence is: prepare compatible representatives, try stabilizer-based weight reduction, check compatibility again, estimate measurement attachment costs, and choose the required branches.

Reduction preserves signed logical classes and records source-check witnesses. Pure $X$ targets use $X$ stabilizers, and pure $Z$ targets use $Z$ stabilizers. Mixed targets use Pauli weight `count(x OR z)`, so physical $Y$ contributes one.

`optimize_logical=True` uses `method="heuristic", threshold="2d", max_trials=128, seed=0` by default. Search is attempted only when target weight is **strictly greater than** $2d$. If input distance is unknown, this default threshold skips search. Set `threshold=None` to always try, or supply a nonnegative integer threshold.

```python
# Default: seeded support reduction when weight > 2d.
result = branching_deform(code, logical_paulis=["ZI", "IZ"])

# Try reduction regardless of initial weight.
result = branching_deform(
    code,
    logical_paulis=["ZI", "IZ"],
    logical_optimization_options={
        "method": "heuristic", "threshold": None,
        "max_trials": 128, "seed": 0,
    },
)

# Preserve supplied physical representatives exactly.
result = branching_deform(
    code, targets=["ZZII", "ZIZI"],
    representatives="preserve", optimize_logical=False,
)
```

Separate reductions can destroy QWC or increase incident-check count. Candidates are accepted only when compatible and when they do not increase the selected plan's ancilla count. The implementation first considers the complete candidate set, then individual replacements. Rejected candidates leave verified representatives unchanged.

A sticker on support size $s$, with $c$ incident checks, adds $s+c$ qubits. Lower target weight alone does not guarantee lower sticker cost. The heuristic proves neither globally minimal weight nor globally minimal branching overhead.

`logical_optimization_options` also accepts the existing reducer's `method="auto"` or `"milp"`, `time_limit`, and related options. Supplying this dictionary with `optimize_logical=False` raises an error.

`result.metadata["support_optimization"]` records options, initial/candidate/accepted weights, cost estimates, witnesses, and decisions. An incompatible candidate set has `candidate_ancillas=None`. If the initial representatives admit no feasible plan, `before_ancillas=None`; this is not a proof that all equivalent representatives are infeasible.

## One CSS $Z$ branch

Let the current code have $n$ qubits and checks $H_X\in\mathbb F_2^{m_X\times n}$, $H_Z\in\mathbb F_2^{m_Z\times n}$. Let $U$ be the union of current target supports, with $|U|=s$. Define:

- $S\in\mathbb F_2^{s\times n}$: selection of $U$.
- $\chi$: all $X$-check rows incident on $U$, including redundant rows; $c=|\chi|$.
- $T\in\mathbb F_2^{m_X\times c}$: embedding of those check coordinates.
- $F=T^TH_XS^T\in\mathbb F_2^{c\times s}$, so $TF=H_XS^T$.

Add qubit groups $a,b$ of sizes $c,s$. In column order $(\mathrm{old},a,b)$, with checks arranged by CSS type,

$$
\boxed{
H_X'=
\begin{pmatrix}
H_X&T&0\\
0&I_c&F
\end{pmatrix},
\qquad
H_Z'=
\begin{pmatrix}
H_Z&0&0\\
S&F^T&I_s
\end{pmatrix}.
}
$$

The relation $TF=H_XS^T$ proves commutation. The last $X$-check block closes the branch without measuring a logical operator; it must not be replaced by a measurement sticker's termination.

This sticker does not require $F$ to have weight-two rows, full rank, connectivity, or an expansion bound. When $c=0$, it reduces to copied qubits joined by $ZZ$ checks.

For a target $z$ supported in $U$, write $r=zS^T$. Since $rF^T=0$, multiplication by selected new $Z$ checks gives

$$
(z,0,0)+(rS,rF^T,r)=(0,0,r).
$$

This is the target representative on the open boundary $b$.

## Transport the complete logical basis

A general lift after a $Z$ branch is

$$
\boxed{
J_X'=(J_X,0,J_XS^T),\qquad
J_Z'=(J_Z,0,0).
}
$$

Padding old $J_X$ with zeros would generally violate commutation with new $Z$ checks. The code transports every generator, including unselected ones.

If a particular $X(x)$ satisfies $aF=xS^T$, then $(x,a,0)$ is equivalent to $(x,0,xS^T)$. Solvability requires commutation with all relevant supported $Z$ logical classes, not just requested targets. The implementation uses the general lift.

An $X$ branch exchanges $X$ and $Z$. Set `leaf_pauli="X"` for $X$ leaves. With `"auto"`, each sticker uses $X$ leaves for a pure $X$ group and $Z$ leaves otherwise.

## Locally compatible Pauli targets, including $Y$

This extension follows Cowtan et al., Lemma 2.10. QWC assigns a consistent nonidentity local axis $P_j=(p_{X,j},p_{Z,j})$ to every qubit in the union. Define

$$
P_X=\operatorname{diag}(p_X)S,\qquad
P_Z=\operatorname{diag}(p_Z)S.
$$

For current symplectic checks $[A\mid B]$, local anticommutation is

$$
R=AP_Z^T+BP_X^T.
$$

Let $F$ retain all nonzero rows of $R$, and let $T$ embed them, so $TF=R$. With $Z$ leaves and check order old, junction, terminal,

$$
\boxed{
H'=
\left[
\begin{array}{ccc|ccc}
A&T&0&B&0&0\\
P_X&0&0&P_Z&F^T&I_s\\
0&I_c&F&0&0&0
\end{array}
\right].
}
$$

A junction check uses $P_j$ on memory. For $P_j=Y$, the output is generally not CSS and requires the full symplectic matrix.

An arbitrary Pauli lifts as

$$
u=xP_Z^T+zP_X^T,\qquad
(x\mid z)\longmapsto(x,0,u\mid z,0,0).
$$

New factors act on new qubits, so the literal Pauli sign is unchanged. Selected QWC targets have $u=0$; multiplication by junction checks transfers them to leaves.

This equals changing local axes to $Z$, constructing the branch, and changing memory axes back. The code directly fills these matrices; physical Clifford gates are not required for that offline calculation. Choosing $X$ leaves exchanges $X/Z$ only on new qubits.

## Branch only when weight budgets require it

Budgets apply separately to **each original check** and **each original physical qubit**, not a global maximum. For symplectic support $C_0=C_{0X}\lor C_{0Z}$, let

$$
w_i^{(0)}=\sum_q(C_0)_{iq},\qquad
\Delta_q^{(0)}=\sum_i(C_0)_{iq}.
$$

The final measurement preview must satisfy

$$
\boxed{
w_i^{\mathrm{preview}}
\le w_i^{(0)}+\texttt{max\_check\_increase},\qquad
\Delta_q^{\mathrm{preview}}
\le\Delta_q^{(0)}+\texttt{max\_qubit\_increase}.
}
$$

Both permitted increases default to 2; equality is allowed. $Y$ is counted once in check support.

The preview builds a basic gauging graph by pairing each check's local anticommutation positions, connecting components, and adding cycle checks. Growth includes branch connections and future measurement attachments. Expansion edges, decongestion, and cellulation are outside this preview. Changing a later graph requires checking the complete matrices again.

For local anticommutation row $F_i$, direct attachment adds $\operatorname{wt}(F_i)/2$ auxiliary factors to that original check and one new check participation at every target-support qubit. Contributions from different targets add. A single target can therefore exceed the default budget when a local row has weight six. Global commutation ensures even weight, not a bound of four.

- `"adaptive"` starts with all targets attached directly. If needed, it searches subsets to branch, using two balanced roots or one shared root followed by a balanced subtree.
- Up to 12 targets, it enumerates subsets within this candidate family. Larger sets use a bounded deterministic greedy search. Reported optimality concerns this family, not all surgery constructions or representatives.
- Targets on memory may overlap while remaining QWC. Targets on different terminal branches have disjoint supports.
- `"always"` uses the full balanced tree for multiple targets. It reports growth but does not reject the tree for exceeding adaptive budgets.
- `force_copy=True` forces one sticker for a single target, including in `"always"` mode. Otherwise a single target is copied only if adaptive budgeting requires it. This option adds no constraint for multiple targets.
- Infeasibility after the configured support-reduction attempt raises an error in adaptive mode.

New checks and qubits have no input baseline; their actual weights are reported separately. Meeting original-item budgets does not establish a uniform bound for the entire deformed code.

`result.routing` records the plan, budgets, targets, roots, ancilla count, search method, and projected original-item growth. `result.measurement_preview` contains actual preview matrices and a consistent reference outcome sector. `result.check_x/z` remains the branching code without measurement checks.

Preview `target_outcome_signs` respect dependencies; they do not predict experimental outcomes. `memory_row_growth` and `memory_column_growth` are relative to the branching code. Total increments relative to input are in `routing.projected_original_row_growth` and `routing.projected_original_column_growth`. Summary `projected_measurement_n` includes measurement ancillas; `n` counts only branching qubits.

## Tree structure and pseudocode

For $t\ge2$, the full balanced tree has $2(t-1)$ stickers and maximum path length $\lceil\log_2t\rceil$. A shared root in adaptive mode adds one common copy before splitting its group. Depth depends on target count, not on repeating $d$ layers.

```text
BRANCHING_DEFORM(code, targets, options):
    prepare signed nontrivial logical representatives
    use the supplied basis or compute a paired CSS logical basis
    check commutation and local Pauli compatibility

    if optimize_logical:
        try stabilizer-equivalent reductions above the threshold
        verify signed equivalence witnesses
        accept compatible candidates that do not increase routing cost

    if branching_mode == "always":
        plan the full balanced tree
        for one target, copy only when force_copy is true
    else:
        begin with direct measurement attachments
        estimate original-check and original-qubit growth
        if budgets are exceeded:
            search subsets and balanced/shared-root layouts
            choose a feasible plan with the fewest branching ancillas
        apply the single-target force_copy rule
        raise an error if no feasible plan is found

    for every sticker in the plan:
        form the group's support union and compatible local Pauli axes
        form R, retain its nonzero rows F, and their embedding T
        append the sticker check matrices
        lift the full logical basis and all targets
        transfer this group's representatives to the open boundary

    construct actual measurement preview matrices
    verify agreement with predicted original-item growth
    verify check commutation, signs, unchanged k, and logical pairings
    verify signed equivalence of inputs and final representatives
    eliminate auxiliaries to check that branching reads no logical information
    return branching code, representatives, witnesses, routing, and preview
```

The preview does not certify measurement-graph expansion, distance, or fault tolerance.

## Example and saved results

```python
from codesurgery.examples import four_qubit_code
from codesurgery.parallel_branching import branching_deform

code = four_qubit_code()  # HX = HZ = [[1, 1, 1, 1]], [[4, 2, 2]]
result = branching_deform(
    code, targets=["ZZII", "ZIZI"],
    representatives="preserve", optimize_logical=False,
)
result.validate()
result.save("branched.json")
result.save("branched.npz")

# Compatible mixed targets, routed to separate leaves.
mixed = branching_deform(
    code, logical_paulis=["YI", "IX"],
    leaf_pauli="Z", branching_mode="always",
)
mixed.validate()
```

The first example meets both default budgets directly, so adaptive mode retains four physical and two logical qubits without branching. With `"always"`, each branch adds $1+2=3$ qubits, giving ten physical qubits. Neither performs the measurement. Preview measurement ancillas do not contribute to `result.n`.

`check_x`, `check_z`, and `check_signs` describe all checks. `leaf_x`, `leaf_z`, and `leaf_signs` describe final targets on memory or new leaves. Mixed $Y$ inputs require the symplectic description.

Source data distinguishes preparation stages:

| Field in `result.source` | Meaning |
| --- | --- |
| `target_x/z/signs` | Physical representatives of the original request |
| `pre_optimization_x/z/signs` | After initial reconstruction, before reduction |
| `prepared_x/z/signs` | Representatives used for branching |
| `initial_canonicalization_witnesses` | Original request to initial representative |
| `support_reduction_witnesses` | Initial to accepted reduced representative |
| `canonicalization_witnesses` | Combined original-to-prepared witness; field name retained for compatibility |

Witnesses refer to original check rows and are verified by full Pauli multiplication. `result.equivalence_witnesses` also relates transported original requests to final representatives.

## Command-line interface

After installation:

```bash
python -m codesurgery.parallel_branching input.json -o branched.json
```

Minimal input:

```json
{
  "name": "four-qubit branching example",
  "hx": [[1, 1, 1, 1]],
  "hz": [[1, 1, 1, 1]],
  "distance": 2,
  "targets": ["ZZII", "ZIZI"],
  "representatives": "preserve",
  "optimize_logical": false,
  "branching_mode": "adaptive",
  "max_check_increase": 2,
  "max_qubit_increase": 2
}
```

Replace `targets` with `logical_paulis`, or with `pauli` and `logical_indices`. `logical_basis` uses the API's `jx/jz` format. If both check matrices are empty, include `"n": ...` to set physical width.

```bash
python -m codesurgery.parallel_branching input.json -o branched.npz \
  --force-copy --leaf-pauli Z --representatives auto

python -m codesurgery.parallel_branching input.json -o branched.json \
  --branching-mode adaptive --max-check-increase 2 --max-qubit-increase 2

python -m codesurgery.parallel_branching input.json -o full-tree.json \
  --branching-mode always
```

Limits must be nonnegative integers. Invalid inputs and infeasible plans raise errors before saving. Saving to an existing output path replaces that file, so use a new filename to retain an earlier result.

CLI options override corresponding JSON options. Reduction defaults to enabled; `--no-optimize-logical` disables it, and `--optimize-logical` overrides JSON `false`. Detailed settings go in JSON:

```json
{
  "optimize_logical": true,
  "logical_optimization_options": {
    "method": "heuristic", "threshold": null,
    "max_trials": 128, "seed": 0
  }
}
```

Merge this fragment with the code and targets. Disabling reduction while supplying its options raises an error. CLI output identifies `stage="branching_only"` and `logical_measurement_performed=false`.

## Static distance preservation

The distance claim concerns the **branching code**, not its measurement preview.

Each sticker adds $c+s$ qubits and the same stabilizer rank, so $k'=k$. For $Z$ leaves in the general Pauli construction, the binary retraction is

$$
\rho(x_o,x_a,x_b\mid z_o,z_a,z_b)
=(x_o+z_bP_X\mid z_o+z_bP_Z).
$$

It maps new normalizers to old normalizers and new stabilizers to old stabilizers, and inverts the logical lift on the quotient. Each $b_j$ factor maps to at most one original qubit, giving

$$
\operatorname{wt}_{\mathrm{Pauli}}(\rho(P))
\le\operatorname{wt}_{\mathrm{Pauli}}(P),
\qquad
d_{\mathrm{branched}}\ge d_{\mathrm{source}}.
$$

This is an algebraic argument, not a physical Clifford circuit. Zhang–Li's Theorem 2 gives the CSS branch-sticker bound with $|S|=1$ for the selector used here. Cowtan et al., Lemma 2.7, states the branching distance consequence; Lemma 2.10 extends it through local basis changes.

No distance solver runs by default. Request a separate calculation explicitly:

```python
from codesurgery.distance_preservation import code_distance

report = code_distance(result, method="gurobi", time_limit=60.0)
result.metadata["distance_calculation"] = report
result.save("branched-with-distance.json")
```

Gurobi requires `gurobipy` and an available license. The 60-second limit is a solver budget; input processing and other steps can add wall-clock time. A timeout may leave only upper and lower bounds.

```bash
python -m codesurgery.parallel_branching input.json -o branched.json --compute-distance
```

Without `distance_options`, the CLI uses `method="auto"` and a 60-second solver budget. To select Gurobi, merge this fragment into input:

```json
{
  "compute_distance": true,
  "distance_options": {"method": "gurobi", "time_limit": 60.0}
}
```

An input `distance` is user-supplied. An inherited numerical lower bound relies on that value being correct; matrix construction does not independently prove the input distance.

The fault-distance statement in Cowtan et al., Lemma 2.9 and Appendix D, also requires specified measurement rounds and time-boundary conditions. This module does not construct or verify a syndrome-extraction circuit, so its static certificate is not a circuit-level fault-distance certificate.
