# Devised sticking

`codesurgery.devised_sticking` implements the finely devised glue code and measurement sticker from Guo Zhang and Ying Li's [*Time-Efficient Logical Operations on Quantum Low-Density Parity Check Codes*](https://arxiv.org/abs/2408.01339). Appendix G supplies the glue construction; Appendices C and D supply the sticker and its attachment to the memory. This is an independent constructor.

The same paper introduces **brute-force branching**, which uses branch stickers to separate overlapping logical targets before measurement. Devised sticking instead uses one measurement sticker designed to retain exactly the requested logical space.

## Inputs and use

~~~python
from codesurgery import CSSCode
from codesurgery.devised_sticking import devised_deform, build_devised_glue

code = CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]], distance=2)

# With no target, measure one Z generator from the computed paired basis.
default = devised_deform(code)

# Simultaneously measure two independent logicals; supports may overlap.
joint = devised_deform(code, ["ZZII", "ZIZI"])

# Pure X uses the dual construction; signed targets are supported.
x_result = devised_deform(code, "-XXII", pauli="X")

# Equivalent binary input: one physical representative per row.
joint = devised_deform(code, [[1, 1, 0, 0], [1, 0, 1, 0]])
joint.save("devised.json")
joint.save("devised.npz")
print(joint.glue.h_g, joint.glue.s, joint.glue.t)
print(joint.hx, joint.hz, joint.summary())

# The lower-level glue constructor accepts unsigned binary rows.
glue = build_devised_glue(code, [[1, 1, 0, 0], [1, 0, 1, 0]])
~~~

The public signature is

~~~text
devised_deform(
    code, logicals=None, *, pauli="Z", logical_index=0,
    d_R=None, ldpc=True, max_entries=50_000_000,
    optimize_logical=False, logical_optimization_options=None,
)
~~~

- `logicals` accepts one physical Pauli string, a binary vector, a list of strings, or a binary row matrix. All targets must be pure X or all pure Z, matching `pauli`. Physical coordinates follow the input check-matrix columns.
- Omitted targets select only generator `logical_index` of the requested type. Indices start at zero. Paired logical generators are computed by binary Gaussian elimination.
- Each target must be a nontrivial normalizer, and their logical classes must be independent modulo the original stabilizers. Identity, stabilizers, non-normalizers, empty lists, and logically dependent targets are rejected.
- Supplied physical representatives are preserved by default. `optimize_logical=True` first attempts stabilizer-based weight reduction in each target's own signed logical class. The default trigger is strictly `weight > 2 * code.distance`; unknown distance skips reduction. Set `logical_optimization_options={"threshold": None}` to attempt it at any weight. Smaller individual supports need not reduce the total sticker overhead.
- `d_R` is the repetition-code length: $d_R$ dual layers and $d_R-1$ primal layers. Its default is the supplied memory distance, or `code.n` if that distance is unknown.
- `ldpc=True` applies bit and check duplication. `False` exposes the dressed glue before this sparsification.
- A single sticker here supports same-type CSS targets. Direct Y, XZ, or mixed X/Z target sets are rejected. The paper's general Pauli protocol in [Appendix I](https://arxiv.org/html/2408.01339v3#A9) uses logical ancillas and several measurement stages; it is not equivalent to one pure X/Z sticker.

The independent command-line interface is

~~~sh
python -m codesurgery.devised_sticking input.json -o deformed.npz
python -m codesurgery.devised_sticking input.json --optimize-logical -o reduced.npz
~~~

JSON accepts `hx,hz,distance,pauli,logical/logicals,logical_index,d_R,ldpc,max_entries,optimize_logical,logical_optimization_options`. Do not supply both `logical` and `logicals`. For empty check sets, provide `n` to retain the matrix width, for example `{"n":2,"hx":[],"hz":[],"logical":"ZZ","d_R":1}`. The top-level `python -m codesurgery` command remains the CKBB interface.

A complete BB90 comparison example is [examples/bb90_compare.py](examples/bb90_compare.py).

## 1. Restricted support and naked glue

Describe pure Z measurement first; exchange X and Z for pure X. Let $J_A\in\mathbb F_2^{q\times n}$ contain the requested physical representatives, $V$ their support union, and $I$ the original X-check rows meeting $V$. Define

$$
H_N=H_X[I,V].
$$

The selector $S_N$ embeds coordinates in $V$ into the memory, and $T_N$ embeds the selected check rows into the original X-check row set. Therefore

$$
H_XS_N^T=T_NH_N.
$$

Let $K_N$ be a row basis of $\ker H_N$. The support-local normalizers and their logical signatures are

$$
Q_N=K_NS_N,\qquad M_N=Q_NJ_X^T.
$$

There are $k_N=\operatorname{rank}M_N$ independent logical classes on this support. The excess $k_N-q$ must be removed. Support-local stabilizers have zero signature and must not be counted as extra logical information.

## 2. Dressing removes only unwanted logical directions

The constructor follows [Appendix G, Algorithm 4](https://arxiv.org/html/2408.01339v3#A7.SS2).

1. Keep a basis of all support-local stabilizer directions:
   $
   G_0=\operatorname{rowbasis}(\ker M_N^T)K_N.
   $
2. Let $G_1$ be an independent target basis restricted to $V$. Internal invertible changes of target basis are allowed; the returned witnesses still refer to the original targets and signs.
3. Complete $G_0,G_1$ to a basis of $\ker H_N$ with $G_2$, removing target components from these complementary rows.
4. Choose paired logical bases $(J_{Z,A},J_{Z,C})$ and $(J_{X,A},J_{X,C})$, where A denotes the measured space and C its logical complement. Then
   $
   G_2S_N=UJ_{Z,C}+VH_Z.
   $

The matrix $U$ is obtained directly by pairing with the complementary X generators:

$$
\boxed{U=(G_2S_N)J_{X,C}^T}.
$$

It has $k_N-q$ rows and $k-q$ columns and full row rank. Solve $UU^r=I_{k_N-q}$ over $\mathbb F_2$, and set

$$
D=(U^r)^TJ_{X,C}S_N^T.
$$

The implementation checks the defining identities

$$
DG_0^T=0,\qquad DG_1^T=0,\qquad DG_2^T=I.
$$

Finally,

$$
H_D=\begin{bmatrix}H_N\\D\end{bmatrix},
\qquad T_D=[T_N\mid0].
$$

Thus $(\ker H_D)S_N$ retains the targets and support-local stabilizers while removing the other logical classes. Fine devising concerns the **logical quotient**: it does not require the entire mapped kernel to contain only the target vectors.

Logical basis changes must preserve pairing. If $J_Z'=RJ_Z$, the conjugate change is $J_X'=R^{-T}J_X$; an untransposed inverse would not in general give $J_X'J_Z'^T=I$.

## 3. Sparsifying the dressed glue

The matrix $D$ can be dense. Appendix G's bit and check duplication convert $H_D$ to the final glue matrix $H_G$:

~~~text
BitDuplication(u, {a,b}):
    add a bit u' and an equality check u + u' = 0
    move the incidences of checks a,b from u to u'

CheckDuplication(a, {u,v}):
    add a bit w and a check b
    move the incidences of u,v from check a to b
    connect both a and b to w, so b imposes w = u + v
~~~

Only dressing incidences move. Naked check rows remain unchanged. New glue-bit rows in $S$ and new glue-check columns in $T$ are zero, preserving

$$
H_XS^T=TH_G,\qquad
(\ker H_G)S=(\ker H_D)S_N.
$$

Write $w_{\max}(H)$ for the maximum of **both row and column weights** of $H$. With `ldpc=True`, the implementation checks

$$
w_{\max}(H_G)\le\max\{w_{\max}(H_X)+1,3\}.
$$

This is a bound on the glue matrix; the final deformed code also contains sticker connections and memory checks.

The implementation row-reduces target coefficients to an identity-pivot form and similarly chooses $U=[I\mid Q]$, up to a column permutation. When the supplied representatives lie in the computed pure logical-generator span, it also checks the optimized dressing bound

$$
\operatorname{nnz}(D)\le(k_N-q)(q+1)
$$

and the resulting glue-size bounds. Arbitrary stabilizer-modified representatives still receive a valid finely devised glue and a checked degree bound; the optimized nonzero-count bound is not assumed for them. Their actual dimensions are reported.

The check-splitting loop applies to dressing checks $\mathcal C_D$, as counted in Appendix G Eq. (109). Applying it only to naked checks would leave dense dressing rows untouched. Compatibility, unchanged naked rows, the mapped kernel, and the final weight bound are all verified.

## 4. Measurement sticker and deformed code

For $H_G\in\mathbb F_2^{r_G\times n_G}$, let
$\lambda=\lambda_{d_R}\in\mathbb F_2^{(d_R-1)\times d_R}$ be the repetition-code check matrix. The sticker matrices are

$$
H_X^M=[I_{d_R-1}\otimes H_G\mid\lambda\otimes I_{r_G}],
\qquad
H_Z^M=[\lambda^T\otimes I_{n_G}\mid I_{d_R}\otimes H_G^T].
$$

The physical order is **memory, all primal layers, all dual layers**, with

$$
n_{\mathrm{anc}}=(d_R-1)n_G+d_Rr_G.
$$

For Z measurement, extend memory X checks by $T$ on the first dual layer and attach $S$ to the first sticker Z-check block. Original memory Z checks have zero auxiliary support. For X measurement, exchange the roles. The returned matrices $\widetilde H_X,\widetilde H_Z$ are checked for commutation.

The remaining logical generators `protected_x,protected_z` are constructed from the paper's logical-operator formulas and verified. Unmeasured Z logicals remain on the memory, while their conjugate X logicals extend to the sticker using a solution of

$$
J_{X,C}S^T=\gamma H_G.
$$

There are $k-q$ protected logical qubits. Additional stabilizer-code logical degrees of freedom are tracked as auxiliary gauges.

## 5. Checking the actual measured space

The glue conditions and target witnesses do not replace final output validation. For the output Pauli-check halves $C_X,C_Z$, eliminate auxiliary support:

$$
A_{\mathrm{aux}}=[C_{X,\mathrm{aux}}\mid C_{Z,\mathrm{aux}}],
\qquad L=\operatorname{rowbasis}(\ker A_{\mathrm{aux}}^T).
$$

The resulting memory-only logical signatures are

$$
\Gamma=[LC_{X,\mathrm{memory}}J_Z^T\mid
        LC_{Z,\mathrm{memory}}J_X^T].
$$

Require its row space to equal the requested target-signature space, of rank $q$. Extra readout, missing targets, or an incorrect space of the same dimension raises `DevisedReadoutError`.

Each row of `measurement_witnesses` selects a check product equal to its signed physical target. Original memory-check signs remain positive, all check dependencies are consistent, and no dependency produces $-I$. Saving revalidates the result before writing; validation failure leaves an existing result file untouched.

## Pseudocode

~~~text
DEVISED_DEFORM(memory, targets=None, pauli=Z, d_R=None):
    choose one paired logical generator if targets are omitted
    require same-type, nontrivial targets independent modulo stabilizers
    optionally reduce each signed representative within its stabilizer coset

    V = union of target supports
    H_N, S_N, T_N = restricted memory checks and selectors
    K = kernel(H_N); Q = K S_N; M = Q J_conjugate^T
    G0 = kernel(M^T) K
    G1 = independent target basis restricted to V
    G2 = complementary kernel basis with target components removed
    construct paired target/complement logical bases
    U = (G2 S_N) J_conjugate,C^T
    choose a full-row-rank pivot basis for U and solve U Uright = I
    D = Uright^T J_conjugate,C S_N^T
    verify D G0^T=0, D G1^T=0, D G2^T=I
    H_D = [H_N; D]
    H_G, S, T = duplication(H_D, S_N, [T_N,0]) if ldpc else dressed glue
    verify compatibility, mapped kernel, and applicable degree bounds

    assemble the measurement sticker and attach it to the memory
    eliminate auxiliary support from the actual output checks
    require measured logical space == target logical space
    solve signed target witnesses and remaining protected logical pairs
    verify commutation, signs, dependencies, witnesses, and protected pairs
    return matrices and construction metadata
~~~

## Outputs and distance

The result exposes `hx,hz`; `check_x,check_z,check_signs`; `measurement_x,measurement_z,measurement_signs,measurement_witnesses`; `protected_x,protected_z,protected_k`; and `glue.h_g,s,t`. Measurement rows retain input target order. JSON and NPZ exports include matrices and construction metadata.

`glue.metadata` records the naked, dressed, and sparsified stages; $G_0,G_1,G_2,U,D$; duplication operations; and before/after $K,Q=KS,M$. `metadata["measurement_validation"]` records the output-based auxiliary elimination and logical signatures.

The reported lower bound is the paper's **protected-subsystem** distance bound

$$
d_{\mathrm{protected}}\ge\min\{d/|S|,d_R\}.
$$

Here $|S|$ is the induced Hamming norm for row vectors, and the construction has $|S|=1$. With a valid supplied memory distance $d$, the reported bound is therefore $\min\{d,d_R\}$. If the input distance is unknown, no numerical lower bound is reported. This constructor does not compute the minimum distance. Auxiliary gauges must not be mistaken for the protected memory logicals when interpreting this bound.

Matrices use dense NumPy storage with explicit size limits. The module does not generate syndrome circuits or decoders.
