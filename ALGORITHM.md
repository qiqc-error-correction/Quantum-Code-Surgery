# CKBB matrix construction and pseudocode

This module constructs a deformed code for one nontrivial logical Pauli observable of a CSS stabilizer code. It follows the alternating-layer and connection constructions in [Cohen, Kim, Bartlett and Brown, *Low-overhead fault-tolerant quantum computing using long-range connectivity*](https://arxiv.org/abs/2110.10794), Figs. 2–5. The formulas below first describe pure X measurements; exchange X and Z for pure Z.

## 1. Choosing a logical basis

When a logical basis is needed, binary Gaussian elimination supplies paired generators. After removing redundant rows for this calculation and applying a common column permutation, write

$$
H_X^S=[A\mid I_{r_X}\mid B],\qquad
H_Z^S=[C\mid D\mid I_{r_Z}].
$$

Then one choice is

$$
J_X^S=[I_k\mid0\mid C^T],\qquad
J_Z^S=[I_k\mid A^T\mid0],\qquad
k=n-r_X-r_Z.
$$

The implementation maps these generators back to the original physical column order. They satisfy

$$
H_ZJ_X^T=H_XJ_Z^T=0,\qquad J_XJ_Z^T=I_k.
$$

This is a standard stabilizer linear-algebra construction; see [Gottesman's thesis, §4.1](https://arxiv.org/abs/quant-ph/9705052) and [Cleve–Gottesman, §III](https://arxiv.org/abs/quant-ph/9607030). The same CSS block convention is used in [Zhang–Li, Appendix A](https://arxiv.org/html/2408.01339v3#A1). The original memory-check rows are retained in the surgery matrices.

With no explicit target, `logical_index` selects one generator of the requested type, starting at zero. A logical Y generator is $i\bar X_j\bar Z_j$, with its physical sign calculated explicitly. Gaussian elimination does not minimize physical support.

## 2. Restricting to the target support

For a pure X target $X(l)$, let $V=\operatorname{supp}(l)$, $w=|V|$, and let $S\in\mathbb F_2^{w\times n}$ select the coordinates in $V$. Retain the original Z-check rows that meet $V$, indexed by $I$, and define

$$
F=H_Z[I,V],\qquad B[I[t],t]=1.
$$

Thus

$$
H_ZS^T=BF,\qquad F\mathbf1=0.
$$

If the rows of $K$ form a basis of $\ker F$, the corresponding memory operators and logical signatures are

$$
Q=KS,\qquad QH_Z^T=0,\qquad
M_X=QJ_Z^T,\qquad a_X=lJ_Z^T.
$$

For pure Z, exchange X and Z. In particular,

$$
\operatorname{rank}M_X
=\operatorname{rank}\begin{pmatrix}H_X\\Q\end{pmatrix}
-\operatorname{rank}H_X.
$$

The requested logical space is $\operatorname{span}(a_X)$. Comparing only the XOR of candidate rows is insufficient: all independent signatures must be considered.

A kernel of dimension greater than one need not imply an extra logical measurement. For example, with `HX=HZ=[110]` and target `111`, the kernel is spanned by `110,001`; one direction is a stabilizer. This can pass a logical-readout check while failing the stronger irreducible-support hypothesis used for the original distance certificate.

A single pure generator from the basis in §1 has a one-dimensional restricted kernel. This fact does not extend automatically to arbitrary products or stabilizer-modified physical representatives. The final code is therefore checked independently.

## 3. Alternating-layer matrices

Let $F$ have shape $c\times w$, and use $r$ dual layers. Order the columns as

$$
(\mathrm{memory};b_1,\ldots,b_r;a_2,\ldots,a_r),
\qquad N=n+rc+(r-1)w,
$$

where each $b_j$ has $c$ qubits and each $a_j$ has $w$ qubits. The first dual-check block attaches through $S$. Internal layers connect through $F,F^T$ and identity matrices.

For $r=2$, with columns $(\mathrm{memory},b_1,b_2,a_2)$,

$$
H_X'=\begin{pmatrix}
H_X&0&0&0\\
S&F^T&0&I_w\\
0&0&F^T&I_w
\end{pmatrix},\qquad
H_Z'=\begin{pmatrix}
H_Z&B&0&0\\
0&I_c&I_c&F
\end{pmatrix}.
$$

The memory interface commutes because $H_ZS^T=BF$; internal contributions cancel over $\mathbb F_2$. Multiplying every new X dual check cancels the primal support, while $F\mathbf1=0$ cancels the dual support, leaving $X(l)$. This supplies a measurement witness.

The unmodified support construction retains $F$; it does not add connections automatically to remove unwanted logical directions. Such directions cause final validation to fail.

## 4. Fig. 3: connecting disjoint factors of the same type

`logical_factors=[l1,...,lt]` requests a product of at least two pure X factors or at least two pure Z factors. Their physical supports must be disjoint. Build separate $F_i,B_i,S_i$ blocks, including separate ancilla copies when an original check meets several factors:

$$
F_0=\operatorname{diag}(F_1,\ldots,F_t),\qquad
B_0=[B_1\ \cdots\ B_t],\qquad
S=\begin{pmatrix}S_1\\\vdots\\S_t\end{pmatrix}.
$$

Then $H_{\mathrm{opposite}}S^T=B_0F_0$. Connection rows join support coordinates in different factors:

$$
F=\begin{pmatrix}F_0\\E_{\mathrm{connection}}\end{pmatrix},
\qquad B=[B_0\mid0].
$$

Each added row has two ones and introduces a connection qubit in every dual layer. These rows describe ancilla connections; they are not direct two-qubit measurements on the memory. The implementation joins consecutive branches for more than two factors. The original memory checks occur only once.

~~~text
FIG3_BRANCHES(code, factors, target=None):
    parse at least two pure factors of the same Pauli type
    require disjoint physical supports and nontrivial logical classes
    product = signed Pauli product of factors
    if target is supplied: require target == product as a signed physical Pauli
    construct Fi, Bi, Si independently
    F0 = blockdiag(Fi); B0 = concatenate(Bi); S = stack(Si)
    add Fig. 3 connections and build alternating-layer matrices
    verify commutation, signs, target witness, and actual logical readout
    raise UnintendedLogicalMeasurementError if extra readout remains
~~~

For two independent `[[4,2,2]]` blocks, the unconnected target `ZZIIZZII` reads two logical classes and is rejected. Supplying the factors `ZZIIIIII` and `IIIIZZII` connects their measurements into one product. A factor that already contains unwanted logical information can still fail after connection.

## 5. Eliminating the auxiliary support

This final check applies to pure, connected-factor, and mixed constructions. Let the output checks have binary halves $C_X,C_Z$, and define

$$
A_{\mathrm{aux}}=[C_{X,\mathrm{aux}}\mid C_{Z,\mathrm{aux}}],
\qquad L=\operatorname{rowbasis}(\ker A_{\mathrm{aux}}^T).
$$

Rows of $L$ select all check products whose auxiliary support vanishes. Their memory components and logical signatures are

$$
R_X=LC_{X,\mathrm{memory}},\quad R_Z=LC_{Z,\mathrm{memory}},\qquad
\Gamma=[R_XJ_Z^T\mid R_ZJ_X^T].
$$

For the requested target $P(x,z)$, set
$\alpha=[xJ_Z^T\mid zJ_X^T]$. Success requires

$$
\alpha\ne0,\qquad
\operatorname{rank}\Gamma
=\operatorname{rank}\begin{pmatrix}\Gamma\\\alpha\end{pmatrix}=1.
$$

Validation also checks commutation with the original stabilizers, pairwise output-check commutation, signed target witnesses, and positive identity for every check dependency. A witness proves that the target can be read; the full signature-space check excludes additional logical measurements.

Failure raises `UnintendedLogicalMeasurementError(ValueError)`, recording the measured rank, measured signature basis, and target signature.

## 6. Main pseudocode

~~~text
CKBB_DEFORM(code, logical=None, logical_factors=None, pauli=Z, layers=None):
    validate binary CSS checks and parameters
    JX, JZ = paired logical generators from Gaussian elimination
    r = layers, or code.distance when given, otherwise code.n

    if logical_factors is supplied:
        require bridge=True
        parse disjoint pure factors of the same type
        derive or verify their signed product
        output = FIG3_BRANCHES(code, factors, target)
    else:
        choose one basis generator if logical is omitted
        require a nontrivial logical target
        if target is pure X or Z:
            construct original support restriction F, B, S
            output = alternating-layer construction(F, B, S, r)
        else:
            output = CKBB_MIXED(code, x, z, bridge)

    construct a signed target witness
    eliminate auxiliary support and verify exact logical readout
    construct applicable CSS cycle gauges after successful validation
    return matrices, witnesses, gauges, and metadata
~~~

## 7. Mixed Pauli targets: Figs. 4 and 5

~~~text
CKBB_MIXED(code, x, z, bridge):
    build pure branches for X(x) and Z(z)
    retain one copy of the memory checks with their branch extensions
    overlap = support(x) intersect support(z)
    if overlap is nonempty:
        fuse the appropriate dual checks at every overlap and layer (Fig. 4)
        add the relative ZX checks on internal layers
    else:
        require bridge=True
        extend branch ends and join their boundary dual checks (Fig. 5)
    solve consistent check signs using Hermitian Pauli arithmetic
    verify the signed witness, absence of -I, and exact logical readout
~~~

The disjoint-support requirement of §4 concerns factors of the same type; it does not exclude overlap between the X and Z parts of a mixed target. The convention is

$$
P(x,z)=i^{\sum_jx_jz_j}X(x)Z(z),
$$

so a site with $x_j=z_j=1$ is a literal Y. A mixed output is generally a non-CSS stabilizer code: `check_x` and `check_z` are halves of the same Pauli rows, not independent CSS check sets. In particular, the pure-layer identity connections in §3 cannot simply be reused unchanged for the fused mixed construction.

## 8. Gauges, distance, and scope

For a successful pure CSS construction, $\ker F^T$ supplies opposite-Pauli cycle candidates. Independent candidates modulo the output opposite-check space are paired as gauge operators, rather than added as measured checks. The counts satisfy

$$
k_{\mathrm{protected}}=k-1,\qquad
g=N-\operatorname{rank}H_X'-\operatorname{rank}H_Z'-(k-1),\qquad
G_XG_Z^T=I_g.
$$

These gauge operators commute with the measured checks. Their weight is not the protected memory distance.

`layers=r` counts dual layers; there are $2r-1$ alternating auxiliary Tanner layers. The original $2d-1$-layer choice corresponds to $r\ge d$. A supplied `code.distance` is an input assumption. The implementation reports a distance certificate only when its layer and support conditions apply; the logical-readout check alone is not a distance proof. In particular, connected factors are checked against the appropriate support conditions separately.

The input is a CSS stabilizer code. General non-CSS or subsystem memory inputs, syndrome circuits, decoders, and circuit-level distance simulation are outside this module. `ckbb_batch` builds independent outputs and does not certify their simultaneous execution.
