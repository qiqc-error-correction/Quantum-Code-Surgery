# CKBB 矩阵构造和伪代码

输入为 CSS stabilizer code `(H_X,H_Z)` 和非平凡逻辑 Pauli。先说明纯 X；纯 Z 交换 X/Z。raw 单支撑构造保留原 restriction；同型 Fig. 3 乘积要求显式给出不相交分支。所有构造最终检查实际逻辑读出。

## 1. 默认逻辑基：Zhang–Li standard form

仅生成基时消去冗余行；CKBB 保留原始校验行。

```text
STANDARD_LOGICALS(HX, HZ):
    检查 HX HZ^T = 0
    X ← HX 的独立 RREF 行基；PX ← 其主元列
    Z ← HZ 的独立行基
    PZ ← Z 限制到 PX 之外后选出的主元列
    V ← 剩余列；p ← [V, PX, PZ]
    归一化 X[:,PX]=I，Z[:,PZ]=I
    在列顺序 p 中写 X=[A,I,B]，Z=[C,D,I]
    JX_standard ← [I_k,0,C^T]
    JZ_standard ← [I_k,A^T,0]
    JX[:,p] ← JX_standard；JZ[:,p] ← JZ_standard
    返回 JX,JZ,p 和 standard-form 元数据
```

`k=n−rank(HX)−rank(HZ)`，并且

\[
H_ZJ_X^T=H_XJ_Z^T=0,\qquad J_XJ_Z^T=I_k.
\]

默认取指定类型的第 0 行。Y 默认生成元由配对 X/Z 的 `i Xbar_j Zbar_j` 生成，并换成有符号物理 Pauli 字符串。standard form 不优化重量；逻辑乘积或手动代表元的支撑若造成额外读出，会被拒绝。

单个纯基生成元有更强性质：`JZ[j]` 的支撑仅在 `V_j` 和 `PX`，所以该支撑内任意 Z normalizer 配对 `JX` 后只能得到 `0` 或 `e_j`。若 signature 为零，它属于 `HZ` 行空间；但它在 `PZ` 为零，而 `HZ[:,PZ]=I`，故它只能是零向量。因此该支撑的 kernel 维数恰为 1。X 方向对称。这证明默认单个纯基生成元安全，同时解释为何任意 product 或手动稳定子修饰后不能直接继承这个结论。

## 2. 支撑 restriction 和逻辑 signature

目标为 `X(l)`，`V=supp(l)`，`w=|V|`。选择矩阵 `S` 为 `w×n`，`S[t,V[t]]=1`。令 `I` 为与支撑相交的原始 Z-check 行：

\[
F=H_Z[I,V],\quad B[I[t],t]=1,\quad H_ZS^T=BF,\quad F\mathbf1=0.
\]

`K` 是 kernel 行基，映回完整物理坐标：

\[
Q=KS,\quad QH_Z^T=0,\quad M_X=QJ_Z^T,\quad a_X=lJ_Z^T.
\]

纯 Z 使用 `QHX^T=0`、`MZ=QJX^T`、`aZ=lJX^T`。配对逻辑基给出逻辑系数，并满足

\[
\operatorname{rank}M_X
=\operatorname{rank}\begin{pmatrix}H_X\\Q\end{pmatrix}
-\operatorname{rank}H_X.
\]

Z 型右侧换 `HZ`。安全逻辑行空间是 `span(a)`，不能只将候选行 XOR 后比较。首个基目标的候选行可为零或 `e0`；任意 product 的 `a` 可有多个 1。

`dim ker F>1` 不必导致额外逻辑，因为候选可为稳定子。例如 `HX=HZ=[110]`、目标 `111`，kernel 为 `span{110,001}`，signature 为 `0,1`。这不满足原文更强的 irreducible-support 条件，不能直接据此认证距离。

raw 构造不改变 `F` 来消除类别。最终输出若实际读出额外逻辑，就抛 `UnintendedLogicalMeasurementError`。

## 3. 原始交替层矩阵

设 `F` 为 `c×w`，使用 `r` 个 dual 层。列按

\[
(\mathrm{data};b_1,\ldots,b_r;a_2,\ldots,a_r),\quad
N=n+rc+(r-1)w
\]

排列，`b_j` 长 `c`，`a_j` 长 `w`。X-check 保留 `HX`；每层添加 `w` 行，在 `b_j` 放 `F^T`，相邻 primal 块放单位阵，第一层在 data 放 `S`。Z-check 将 `HZ` 延伸 `B` 到 `b1`；每个内部层添加 `c` 行，在 primal 块放 `F`，相邻 dual 块放单位阵。

`r=2`，列为 `(data,b1,b2,a2)`：

\[
H_X'=\begin{pmatrix}
H_X&0&0&0\\S&F^T&0&I_w\\0&0&F^T&I_w
\end{pmatrix},\quad
H_Z'=\begin{pmatrix}
H_Z&B&0&0\\0&I_c&I_c&F
\end{pmatrix}.
\]

接口由 `HZ S^T=BF` 保证对易，内部由 `F+F=0` 抵消。全部 dual checks 相乘时，primal 支撑出现两次，dual 支撑由 `F1=0` 消去，只剩 `X(l)`。这给出 measurement witness。

## 4. Fig. 3：显式不相交同型分支

`logical_factors=[l1,...,lt]` 要求 `t≥2`，所有因子同为纯 X 或纯 Z，物理支撑不相交。各因子分别计算 `Fi,Bi,Si`，共享原校验在各分支有自己的辅助副本：

\[
F_0=\operatorname{diag}(F_1,\ldots,F_t),\quad
B_0=[B_1\;\cdots\;B_t],\quad
S=\begin{pmatrix}S_1\\\vdots\\S_t\end{pmatrix}.
\]

`H_opposite S^T=B0 F0`。跨分支连接的 incidence 行只连接两个支撑坐标；`B` 相应补零列：

\[
F=\begin{pmatrix}F_0\\E_{\mathrm{connection}}\end{pmatrix},\quad B=[B_0\;0].
\]

每个 dual 层增加连接比特，进入两个对应 dual checks；内部 opposite connection check 作用在两端 primal 副本及相邻层连接比特。这里的 incidence 行不表示在存储码上直接测量两体算符。多于两个分支时依次连接相邻分支。原始数据校验只保留一份，但可延伸到多个辅助副本。

```text
FIG3_BRANCHES(code, factors, target=None):
    解析至少两个同型纯 X/Z 因子；拒绝支撑重叠
    检查各因子是非平凡存储码 normalizer
    product ← 因子的有符号 Pauli 乘积
    target 未给出则取 product；否则要求完整有符号物理算符一致
    各分支独立生成 Fi,Bi,Si
    F0 ← blockdiag(Fi)；B0 ← concat(Bi)；S ← stack(Si)
    按 Fig.3 连接分支，生成 F,B
    用 F,B,S 创建交替层矩阵
    验证对易、符号、witness 和最终逻辑读出
    如连接后仍额外读出逻辑，抛 UnintendedLogicalMeasurementError
```

两个 `[[4,2,2]]` 块的 `ZZIIZZII`：raw 有两个逻辑类别，必须拒绝。显式因子 `ZZIIIIII`、`IIIIZZII` 的 Fig. 3 输出只读出乘积。若一个因子本身包含两个独立块的 product，连接另一分支仍可能留下额外类别；仅有连接存在不能判定安全。

## 5. 最终 auxiliary 消元与完整逻辑检查

统一用于 raw、Fig. 3 和 mixed。设 `C=[CX|CZ]`：

\[
A=[C_{X,\mathrm{aux}}\mid C_{Z,\mathrm{aux}}],\quad L=\operatorname{rowbasis}(\ker A^T),\quad
R_X=LC_{X,\mathrm{data}},\quad R_Z=LC_{Z,\mathrm{data}}.
\]

`L` 的行选择所有 auxiliary 作用消失的校验乘积。计算

\[
\Gamma=[R_XJ_Z^T\mid R_ZJ_X^T],\qquad
\alpha=[xJ_Z^T\mid zJ_X^T].
\]

成功要求

\[
\alpha\ne0,\qquad
\operatorname{rank}\Gamma
=\operatorname{rank}\begin{pmatrix}\Gamma\\\alpha\end{pmatrix}=1.
\]

另外验证 memory image 与原稳定子对易、全部输出 checks 对易、有符号 witness 等于完整物理目标，以及全部 check dependencies 产生 `+I`。witness 只证明目标可读出；完整行空间才排除额外逻辑测量。失败抛 `UnintendedLogicalMeasurementError(ValueError)`，其属性记录实际逻辑 rank、测得的 signature 基及目标 signature。

## 6. 主程序伪代码

```text
CKBB_DEFORM(code, logical=None, logical_factors=None, type=Z, r=None):
    检查 CSS 对易、输入二进制、参数
    JX,JZ ← STANDARD_LOGICALS(code)
    r 默认 code.distance（已给出时），否则 code.n
    若 factors 已给出：
        解析同型、不相交、至少两个因子；derive/核对有符号目标
        bridge=False 时拒绝 Fig.3
        output ← FIG3_BRANCHES(code, factors, target)
    否则：
        若 logical 未给出：取指定 standard-form X/Z/Y 生成元
        检查目标非零、非稳定子且与存储校验对易
        若目标纯 X/Z：
            F,B,S ← 原始支撑 restriction
            K ← ker(F)；Q ← KS；M ← Q J_conjugate^T
            output ← 原始交替层矩阵
        否则：output ← CKBB_MIXED(code, x,z,bridge)
    生成有符号 witness
    执行最终 auxiliary 消元和完整 signature 检查
    失败立即抛异常；仅在通过后生成适用的 CSS cycle gauges
    返回矩阵、witness、gauges、元数据
```

## 7. Mixed Pauli：Fig. 4/5

```text
CKBB_MIXED(code, x,z,bridge):
    分别建立 X(x)、Z(z) 的原始纯分支
    保留一份存储校验及两个分支提供的对应延伸
    overlap ← supp(x) ∩ supp(z)
    若 overlap 非空：
        按 Fig.4，每个交叠点、每层融合对应 dual checks
        每个内部层添加相对 ZX check
    否则：
        bridge=False 时拒绝所需 Fig.5 connector
        分支生长虚拟端点/连接臂，再融合边界 dual checks
    用 canonical Hermitian Pauli 算术解一致校验符号
    验证 witness、无 -I、完整 logical-readout signature
```

mixed 的 X/Z 支撑交叠是 Fig. 4 的输入；同型 Fig. 3 的不相交限制不适用于此。采用 `P(x,z)=i^(sum xz)X(x)Z(z)`，所以重叠位表示字面 `Y`。mixed 输出通常为非 CSS stabilizer 码，两半不能分别当 `HX,HZ`。

## 8. Gauge、距离和范围

`ker F^T` 产生 opposite-Pauli cycle 候选；模掉输出 opposite-check row space 后，选择独立 gauge 并配对。成功纯 CSS 输出满足

\[
k_{\mathrm{protected}}=k-1,\quad
g=N-\operatorname{rank}H_X'-\operatorname{rank}H_Z'-(k-1),\quad
G_XG_Z^T=I_g.
\]

gauge 与 measured checks 对易，低重量 cycle 不属于受保护的存储逻辑。`r` 计 dual 层，共 `2r−1` 层；原文 `2d−1` 层对应 `r≥d`。输入距离由调用者声明；逻辑读出安全不能代替支撑假设和 fault-distance 证明。距离认证单独检查原文支撑条件，包括 Fig. 3 各分支条件，不满足时给未知下界。

本版输入 CSS stabilizer 码，没有接入原始 subsystem/gauge 或一般非 CSS 存储码。`ckbb_batch` 生成独立输出，不声称同时测量；本版不含 syndrome 电路、解码器或电路级距离仿真。

参考：[CKBB](https://arxiv.org/abs/2110.10794)，[CKBB Fig. 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10926894/#F3)，[Zhang–Li Appendix A](https://arxiv.org/html/2408.01339v3#A1)。
