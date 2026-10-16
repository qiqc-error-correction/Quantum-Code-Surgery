# High-rate surgery with a shared ancilla

`codesurgery.high_rate_surgery` follows Guo Zheng, Liang Jiang, and Qian Xu's [*High-Rate Surgery: towards constant-overhead logical operations*](https://arxiv.org/abs/2510.08523). Its main constructor uses the paper's randomized, shared-ancilla approach, with bounded searches suitable for finite inputs. A separate algebraic entry point accepts user-supplied ancilla matrices and attachment maps.

This module constructs and validates static deformed-code matrices. It does not establish a universal overhead guarantee for every input and search budget, and it is not an implementation of surgery from linear PCPPs.

## Inputs and measurement semantics

The input is a `CSSCode(hx, hz, distance=None, name=...)`. A user-supplied `logical_basis` may specify the paired logical generators. Otherwise, the basis is computed by binary Gaussian elimination.

~~~python
from codesurgery.examples import four_qubit_code
from codesurgery.high_rate_surgery import high_rate_deform

code = four_qubit_code()

# Two separate logical outcomes.
parallel = high_rate_deform(code, logical_paulis=["ZI", "IZ"])

# One product outcome; the individual factors must not be exposed.
joint = high_rate_deform(code, logical_paulis=["ZZ"])

parallel.validate()
parallel.save("parallel.json")
joint.save("joint.npz")
~~~

Physical strings and logical strings have different lengths:

| Input | Meaning |
| --- | --- |
| `targets=["ZZII"]` | One physical Pauli representative of length `n` |
| `targets=[{"x": [...], "z": [...], "sign": 0}]` | Explicit physical binary halves; `sign=1` means minus |
| `logical_paulis=["ZZ"]` | One logical Pauli product of length `k` |
| `logical_indices=[0, 2], pauli="Z"` | Separate measurements of Z generators 0 and 2 |
| No target selection | Separate measurements of every Z basis generator |

Choose only one of `targets`, `logical_paulis`, and `logical_indices`. A list means several observables; a product measurement is one complete string or one complete pair of binary halves. In particular, Y must be supplied as one target, rather than split into separate X and Z measurements.

The selected physical representatives must be **qubit-wise commuting (QWC)**: on each physical qubit, any two targets act with the same Pauli, or at least one acts with I. This includes same-type pure X or pure Z sets. A single mixed Pauli target, including Y or an XZ product, automatically satisfies QWC. Global commutation alone is insufficient for this implementation.

The mixed construction uses local Pauli axes and binary symplectic matrices. This is an offline matrix calculation; it does not prescribe physical basis-change gates. When additional gauge qubits remain, the module does not have a general mixed-Pauli protected-subsystem distance backend. Gauge-free mixed outputs can instead be checked using the full stabilizer-code distance.

## Output matrices and exact readout validation

For pure X measurements, let $S$ select the memory support. Let $F$ have one row per ancilla qubit and one column per attachment check, let $M$ extend the original Z checks to the ancilla, and let $G$ contain the added Z checks:

$$
\widetilde H_X=
\begin{pmatrix}H_X&0\\S&F^T\end{pmatrix},
\qquad
\widetilde H_Z=
\begin{pmatrix}H_Z&M\\0&G\end{pmatrix}.
$$

Commutation requires

$$
MF=H_ZS^T,\qquad GF=0.
$$

After adding layers, these matrices and the attachment map expand accordingly. Exchange X and Z for pure Z measurement. General Pauli outputs use `check_x,check_z,check_signs`; these are halves and signs of the same Pauli-check rows. Row weights and column degrees count each Y once:

$$
w_i=\sum_q(X_{iq}\lor Z_{iq}),\qquad
\deg(q)=\sum_i(X_{iq}\lor Z_{iq}).
$$

The shared support may contain logical information beyond the targets. Initialization adds separating constraints as needed and checks the actual measured logical space, rather than assuming that independently restricted target supports can simply be concatenated.

Every returned construction is checked for:

1. Pairwise check commutation and phase consistency, including absence of $-I$ dependencies.
2. All memory-only check products obtained by eliminating auxiliary support.
3. Equality between their logical-signature space and the requested target-signature space.
4. Signed measurement witnesses for the individual targets.

Thus `["XX"]` must not additionally read `XI` or `IX`; `["XI", "IX"]` explicitly requests both outcomes. If $w_{ij}$ is `measurement_witnesses[i,j]` and $c_i$ is `readout_sign_corrections[i]`, then

$$
\text{target outcome}_i
=(-1)^{c_i}\prod_{j:w_{ij}=1}
\text{signed-check outcome}_j.
$$

This is an outcome rule, not a generated sample from a device or simulator. Repeated targets, including sign-reversed copies, retain their required outcome dependencies.

## Defaults, weight limits, and distance

~~~python
result = high_rate_deform(
    code,
    logical_paulis=["ZI", "IZ"],
    optimize_support=True,
    ensure_distance=True,
    target_distance=None,
    distance_method="auto",
    distance_time_limit=60,
    max_check_weight=None,
    max_qubit_degree=None,
    seed=0,
    max_steps=100,
    max_layers=20,
    strict=False,
    trials=64,
    max_entries=50_000_000,
)
~~~

By default, the constructor first tries to reduce support within each signed stabilizer coset, while retaining a compatible target set. This native support search has no `2d` trigger. Disable it with `optimize_support=False` to preserve supplied physical representatives.

Unless explicitly configured, the two independent limits are

$$
w_{\max}^{\mathrm{limit}}
=\max_i w_i(\text{original checks})+2,\qquad
\Delta_{\max}^{\mathrm{limit}}
=\max_q\deg_q(\text{original checks})+2.
$$

They apply to the **complete output**, including new checks and ancilla qubits. This global-maxima policy differs from branching's separate growth allowance for each original row and column. A returned high-rate construction must obey both limits, even with `strict=False`.

`ensure_distance=True` enables distance checking and attempted repair; success is reported separately. The target distance defaults to the supplied `code.distance`. This input distance is an assumption, not a distance calculation performed by the code constructor.

`distance_time_limit=60` is a **shared budget** for distance checks and growth after initialization. Solver calls share the remaining time. Support optimization, initialization, and final validation are outside this budget, and ongoing matrix operations are not forcibly interrupted. It is not a strict one-minute limit for the full API call.

The available backends are:

- `"auto"`: prefer Gurobi when usable, with the implemented fallback when it is unavailable.
- `"gurobi"`: require that backend; missing installation or license is reported as incomplete certification.
- `"enum"`: search Pauli errors by increasing weight, using the applicable distance problem.
- `"heuristic"`: search for short logical representatives. Failure to find one is not a distance lower bound.

For pure CSS outputs, the backend searches protected dressed distance, excluding auxiliary gauge logicals. For gauge-free QWC mixed outputs, it searches all Pauli errors with weight $\operatorname{wt}(x\lor z)$, including Y counted once. Mixed outputs with additional gauge qubits are reported as unsupported for protected-distance certification.

Distance reports distinguish an upper-bound witness from a lower bound:

- A validated nontrivial logical of weight $u$ proves $d\le u$.
- Exhaustive exclusion through weight $l$, or an accepted solver lower bound, is needed to establish $d>l$.
- A timeout or unsuccessful heuristic search alone proves neither.
- MILP bounds rely on the solver's numerical tolerances and the backend's validation rules.

The gauge-free mixed report uses scope `protected_mixed_stabilizer_gauge_free`. Growth proposals may be scored using the auxiliary X component of a validated short error, but the complete Pauli distance must be rechecked on the resulting matrices.

With `strict=False`, an algebraically valid, weight-feasible candidate may be returned without a completed distance certificate. Inspect `result.metadata["status"]` and its distance report. With `strict=True`, failure to establish the requested protected distance raises `DistanceCertificationError`; its `.result` retains the valid candidate.

CLI strict-mode failure exits with code 2. If an explicit `-o`/`--output` path was supplied, the validated candidate is still saved there, and the summary records `strict_failed=true` and `candidate_saved=true`. Without an explicit path, no candidate file is created.

`protected_distance_certified` concerns the surviving memory logical subsystem. `measurement_fault_distance_certified` remains false: the module does not certify the fault distance of a complete measurement record or gate schedule. Measuring all memory logicals, so that `protected_k=0`, does not certify reliable measurement outcomes.

| Status | Meaning |
| --- | --- |
| `constructed_distance_disabled` | Algebra and weight checks passed; distance checking was disabled |
| `protected_distance_certified` | The protected-distance lower bound reached the requested target |
| `constructed_mixed_distance_unverified` | A mixed construction has extra gauge qubits and no supported protected-distance backend |
| `constructed_all_logicals_measured_readout_distance_unverified` | No memory logical remains; measurement-record fault distance is still unverified |
| `constructed_no_distance_target` | No trusted target distance was supplied; available bounds are returned |
| `constructed_algebraic_distance_unverified` | A supplied algebraic ancilla is valid, but the requested distance is not certified |
| `distance_inconclusive` / `distance_time_limit` | Available bounds are insufficient or the shared budget expired |
| `step_limit` / `no_admissible_growth` | The growth limit was reached or no acceptable move was found |

### Growth and its limits

Candidates are scored by how many known short-error witnesses they detect, then the accepted construction is revalidated and its distance checked again. No separate $h>1$ expansion certificate is inferred from random connections.

Initialization can fail if its bounded search finds no separating constraints that meet the weight limits. This is a search failure, not a proof that no construction exists.

When more layers are allowed, the implementation reserves one connection position on the last layer. Following the kernel condition of Appendix E Eq. (E7), in-layer growth is used only for a layer with the required completed kernel. Copies from an adjacent layer may first complete an unfinished layer, even when that copy does not directly detect a current short-error witness.

This is a finite heuristic construction. It does not implement a complete worst-case HGP fallback or guarantee distance preservation for every input, degree limit, and time budget. The distance certificate belongs to the actual final matrices, not an earlier growth stage.

## Pseudocode

~~~text
HIGH_RATE(code, targets, options):
    basis = supplied basis or paired generators from Gaussian elimination
    parse signed targets; require nontrivial logical operators
    check global commutation
    optionally reduce supports within the original signed logical classes
    require compatible physical Pauli axes (QWC)

    build a shared restriction and attachment maps
    add separating constraints to exclude unwanted logical readout
    construct compatible auxiliary checks
    require exact target logical readout and complete-matrix weight limits

    if distance checking is enabled and a backend supports the output:
        repeat within distance/growth, step, and layer budgets:
            compute available distance bounds and short-error witnesses
            stop if the requested lower bound is established
            stop if no repair witness or no time remains
            propose target-preserving connections
            score them against validated short-error witnesses
            reject moves violating complete-matrix weight limits
            add an admissible connection, copy, or layer
            rebuild and revalidate the full output
            discard bounds that belonged to earlier matrices
        refine bounds on the final matrices if time remains

    fail if no algebraically valid, weight-feasible construction is found
    in strict mode, fail if the requested distance is not certified
    return matrices, witnesses, distance bounds, and growth history
~~~

## Algebraic HGP ancillas

`hgp_ancilla(HB, HD)` returns binary blocks $(B,C)$:

$$
B=\begin{pmatrix}I_s\otimes H_D\\H_B\otimes I_u\end{pmatrix},
\qquad
C=\begin{pmatrix}H_B\otimes I_t&I_r\otimes H_D\end{pmatrix},
$$

where `HB.shape=(r,s)` and `HD.shape=(t,u)`. Therefore $CB=0$. The tool constructs these blocks only; arbitrary HGP factors do not determine valid attachment maps for an arbitrary memory code or establish the paper's distance hypotheses.

With blocks and maps already available, use

~~~python
high_rate_from_ancilla(
    code, boundary, cycles, gamma0, gamma1,
    targets=None,
    pauli="X",  # Pure X or pure Z only; this entry point defaults to X.
    logical_paulis=None,
    logical_indices=None,
    logical_basis=None,
    ensure_distance=True,
    target_distance=None,
    distance_method="auto",
    distance_time_limit=60,
    max_check_weight=None,
    max_qubit_degree=None,
    strict=False,
    seed=0,
)
~~~

For `boundary.shape=(a,s)`, require `gamma1.shape=(n,s)` and
`gamma0.shape=(m_opposite,a)`. For pure X, `m_opposite=len(code.hz)`; for pure Z, it is `len(code.hx)`. The supplied `gamma0` contains only the opposite-type original check rows, unlike the internal randomized state, which stores maps for all original check rows.

For pure X, the chain-map conditions are

$$
\Gamma_0B=H_Z\Gamma_1,\qquad CB=0.
$$

The entry point validates these identities, full Pauli commutation and signs, exact target readout, and complete-matrix weights, and optionally calculates distance. It does not search for attachment maps or run the randomized repair loop.

The following example passes an existing randomized construction's blocks and maps to the algebraic entry point:

~~~python
from codesurgery.examples import four_qubit_code
from codesurgery.high_rate_surgery import high_rate_deform, high_rate_from_ancilla

code = four_qubit_code()
shared = high_rate_deform(
    code, logical_paulis=["XI"],
    optimize_support=False, ensure_distance=False,
)

rebuilt = high_rate_from_ancilla(
    code,
    shared.state.boundary,
    shared.state.cycles,
    shared.state.gamma0[len(code.hx):],  # Original Z-check rows only.
    shared.state.attachment_x.T,        # Shape: n by s.
    logical_paulis=["XI"], pauli="X",
    ensure_distance=False,
)
rebuilt.validate()
rebuilt.save("algebraic-deformed.json")
~~~

If the blocks instead come from `hgp_ancilla`, the caller must still supply valid `gamma0,gamma1` that realize exactly the target space. The identity $CB=0$ alone does not establish a correct surgery.

## Command-line interface

~~~sh
# Two separate outcomes on a small code, without distance repair.
python -m codesurgery.high_rate_surgery --example four \
    --logical-pauli ZI --logical-pauli IZ \
    --no-ensure-distance -o four-two-z.json

# One product outcome.
python -m codesurgery.high_rate_surgery --example four \
    --logical-pauli ZZ --no-ensure-distance -o four-zz.npz

# BB90: distance checking and growth share the 60-second budget.
python -m codesurgery.high_rate_surgery --example bb90 \
    --logical-pauli ZIIIIIII --logical-pauli IZIIIIII \
    --ensure-distance --distance-method gurobi \
    --distance-time-limit 60 --seed 0 --trials 32 -o bb90-two-z.npz
~~~

When neither the command line nor JSON specifies `ensure_distance`, an interactive terminal asks whether distance preservation is required, defaulting to yes. Noninteractive use enables it by default and reports that choice on stderr. Explicit `--ensure-distance` or `--no-ensure-distance` avoids the prompt.

Example JSON:

~~~json
{
  "example": "four",
  "logical_paulis": ["ZI", "IZ"],
  "optimize_support": true,
  "ensure_distance": false,
  "max_check_weight": 6,
  "max_qubit_degree": 4,
  "seed": 0
}
~~~

~~~sh
python -m codesurgery.high_rate_surgery input.json -o deformed.json
~~~

Replace `example` with `hx,hz` (or `H_X,H_Z`) and optional `distance,name` for a custom code. If both check sets are empty, supply positive `n`. Do not combine an example name with explicit matrices.

An explicit CLI target selection replaces the JSON target selection. JSON export is suitable for inspecting matrices, witnesses, and metadata; NPZ export is suitable for NumPy. The CLI writes its JSON summary to stdout and interactive prompts or diagnostics to stderr.
