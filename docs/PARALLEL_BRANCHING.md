# Parallel branching：保留逻辑空间的分支构造

本模块实现 [Parallel Logical Measurements via Quantum Code Surgery，arXiv:2503.05003v3](https://arxiv.org/html/2503.05003v3) 的 branching 构造。默认先估计直接接入测量 gadget 后的行、列重量，仅对会使原码各项预算超限的目标进行分支；未超限的目标留在 memory 上。`branching_mode="always"` 可恢复全部目标分支到互不交叠叶子的完整树。条件选择策略是本项目的开销优化，不是论文中的额外定理。

**branching 本身不测量逻辑算符。** 输出码保留原来的全部 \(k\) 个逻辑 qubit；返回的目标代表仍是逻辑算符。之后若需要测量，可另行使用 gauging measurement 或其他测量构造。本模块与 `ckbb`、`devised_sticking` 分开，不以更改 CKBB 的终止层实现 branching。

论文 §2.2、Lemma 2.5–2.10 说明分支、逻辑保持和适用条件；§3 将分支与测量组合；Appendix A.2 用 mapping cylinder 说明逻辑映射；Appendix E 讨论相反类型代表发生 overlap 时的运输。

以下矩阵按本项目的“check 为行、qubit 为列”约定展开。二进制运算均在 \(\mathbb F_2\) 中，Pauli 的正负号另外保存。

新 check 的符号取 \(+1\)，表示一个固定的参考稳定子扇区。实际执行时，新 junction check 的测量结果可以是随机的；需要用测量记录更新 Pauli frame 和叶子读出的符号。本模块输出静态码，不模拟这些测量结果。

**输入与默认逻辑基**

高层入口是：

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

`code` 是 `CSSCode`。目标有三种互斥的输入方式：

| 方式 | 含义 | 示例 |
| --- | --- | --- |
| `targets` | 长度为物理 qubit 数 \(n\) 的带符号 IXYZ 字符串 | `["-YZZY", "ZIIZ"]` |
| `logical_paulis` | 长度为 \(k\) 的逻辑基坐标字符串 | `["YI", "IX"]` |
| `pauli` 与 `logical_indices` | 对指定的逻辑 qubit 取同一类型；下标从 0 开始 | `pauli="Z", logical_indices=[0, 2]` |

没有给目标时，默认选择所选逻辑基中全部 \(Z\) 逻辑代表，再按重量预算决定是否分支。没有给基时，使用 Guo Zhang–Ying Li 的 simultaneous standard form，参见 [Time-efficient logical operations on quantum LDPC codes，Appendix A](https://arxiv.org/html/2408.01339)。传入基时使用 `logical_basis={"jx": ..., "jz": ...}`，要求恰好 \(k\) 对纯 CSS normalizer，并满足

\[
J_XH_Z^T=0,
\qquad J_ZH_X^T=0,
\qquad J_XJ_Z^T=I_k.
\]

逻辑基的高斯消元仅用于选代表；实际构造仍保留原始稀疏 check 行，包括冗余行。

**QWC 条件与符号保持**

同时分支要求目标在每个物理 qubit 上要么使用相同的非平凡 Pauli，要么其中一方是 \(I\)，即 physical qubit-wise commutation（QWC）。仅仅整体对易不足以满足这个条件。

- `representatives="preserve"`：初始准备时保留输入的物理代表和符号，若不满足 QWC 则报错。
- `representatives="auto"`：初始准备时，已满足 QWC 则原样保留；否则尝试在**同一组逻辑基、同一个逻辑类**内重建标准代表，并再次检查 QWC。
- `representatives="canonical"`：总是在所选基内重建，然后检查 QWC。

这些选项控制初始 canonicalization，后面的 support 优化由 `optimize_logical` 独立控制。若要求精确保留用户输入的物理代表，使用 `representatives="preserve", optimize_logical=False`。

重建失败时不会更换用户的逻辑基，也不会把任意一组对易算符宣称为可同时分支。用户给定代表或自定义基时，可能需要提供另一组经过验证的代表；非 QWC 的 sequential branching 不由这个高层入口自动承诺。

对于物理目标 \((x\mid z)\)，逻辑坐标为

\[
a=xJ_Z^T,
\qquad b=zJ_X^T.
\]

canonical 二进制代表取 \(x_c=aJ_X, z_c=bJ_Z\)。程序解稳定子见证 \(w\)，使

\[
w[H_X^{\mathrm{symp}}\mid H_Z^{\mathrm{symp}}]
=(x+x_c\mid z+z_c).
\]

随后使用完整 Pauli 乘法计算符号，保证

\[
P_{\mathrm{prepared}}
=P_{\mathrm{original}}\prod_{j:w_j=1}S_j.
\]

这里 check 次序是所有原始 \(H_X\) 行后接所有 \(H_Z\) 行。仅验证二进制差属于稳定子行空间不够：例如四比特码中 `YZZY` 乘 `XXXX`、`ZZZZ` 后是 **`-IXXI`**。

逻辑字符串也要转换符号。约定 \(P(x,z)=i^{x\cdot z}X(x)Z(z)\)，那么逻辑 \(i^{a\cdot b}\bar X(a)\bar Z(b)\) 相对物理 \(P(aJ_X,bJ_Z)\) 的相位为

\[
i^{a\cdot b-(aJ_X)\cdot(bJ_Z)}.
\]

有效的 paired CSS basis 使这个相位为 \(+1\) 或 \(-1\)。

每个目标必须是非平凡 normalizer；目标之间必须对易。重复目标、线性相关目标、支撑内含有其他逻辑信息的可约目标都可以分支，独立逻辑秩会单独记录。因为分支不读出逻辑信息，所以没有“支撑内只能包含目标一个逻辑”的前提。

**先减少 support，再构造分支**

默认流程为：选定逻辑基并准备兼容代表 → 在原码稳定子陪集中减重 → 重新核验 QWC → 估计直接测量开销 → 按原码行、列预算选择需要分支的目标。稳定子减重保留每个目标原有的带符号逻辑类，并为最终采用的代表保存原始 check 系数见证。纯 \(X\) 目标只乘 \(X\) 稳定子，纯 \(Z\) 目标只乘 \(Z\) 稳定子；含 \(Y\) 或混合 \(XZ\) 的目标采用完整 Pauli 重量，即 `count(x OR z)`，每个 \(Y\) 计为一个物理比特。

`optimize_logical=True` 是默认值。默认优化设置为 `method="heuristic", threshold="2d", max_trials=128, seed=0`，只有目标重量**严格大于** \(2d\) 时才尝试减重。若原码未给 `distance`，默认规则跳过搜索并记录原因；可设 `threshold=None`（JSON 为 `null`）来始终尝试，也可给非负整数阈值。随机重启采用固定种子以便重现结果。

分别减重可能改变混合目标之间的局部 Pauli 轴兼容性，也可能增加关联 check 数。因此，程序对候选集合重新检查 QWC，并按所选 `branching_mode` 计算实际计划的辅助比特数。整组候选兼容且辅助比特数不增加时一起采用；否则逐个尝试替换，只接受兼容且不增加计划开销的替换，重复到不能继续接受候选。不符合条件的候选退回已经验证的代表。不能仅凭逻辑类仍然对易就直接使用新的物理支撑。

```python
# 默认：重量 > 2d 时先做确定种子的启发式减重。
result = branching_deform(code, logical_paulis=["ZI", "IZ"])

# 无论初始重量多大都尝试；优化选项与 reduce_logical_weight 一致。
result = branching_deform(
    code,
    logical_paulis=["ZI", "IZ"],
    logical_optimization_options={
        "method": "heuristic", "threshold": None,
        "max_trials": 128, "seed": 0,
    },
)

# 完全保持输入物理代表。
result = branching_deform(
    code, targets=["ZZII", "ZIZI"],
    representatives="preserve", optimize_logical=False,
)
```

`logical_optimization_options` 还可使用现有减重器的 `method="auto"` 或 `"milp"`、`time_limit` 等参数；禁止在 `optimize_logical=False` 时同时传入该选项字典，以免静默忽略用户设置。默认启发式通常没有全局最优证明；若结果恰好达到输入声明的码距，下界仍依赖该输入声明的可信性。一个 sticker 的开销是支撑大小 \(s\) 加上关联 check 数 \(c\)，即 \(s+c\)。候选搜索最小化支撑重量，接受阶段另外保证所选模式下实际分支计划的新增比特数不增加；这不构成最小 branching 开销的全局最优证明。

`result.metadata["support_optimization"]` 保存启用状态、优化选项、优化前/候选/采用后的重量、分支计划辅助比特数、候选的稳定子见证和接受决定。非 QWC 的整组候选没有有效 branching 开销，其 `candidate_ancillas` 记为 `None`。这些信息可区分“减重器找到候选”和“候选实际被构造采用”。若自定义的严格预算使初始代表不可行，仍会按配置先尝试减重；此时 `before_ancillas=None` 表示没有可行的初始路由，而不是无限大数值。最终报错仅针对尝试后的代表，不证明该逻辑类的所有代表都不可行。

support 优化只改变代表选择；每个实际添加的 sticker、终止 cap checks 及逻辑保持验证沿用原有构造。support 优化的后端与码距求解相互独立，不改变现有 Gurobi 码距默认设置和 60 秒预算。

**一步 CSS \(Z\)-branch 的矩阵**

设当前码有 \(n\) 个比特，\(H_X\in\mathbb F_2^{m_X\times n}\)、\(H_Z\in\mathbb F_2^{m_Z\times n}\)。对一个待分支的目标组，取其当前代表支撑的并集 \(U\)，大小为 \(s\)。

- \(S\in\mathbb F_2^{s\times n}\)：按确定的坐标顺序选择 \(U\)。
- \(\chi\)：与 \(U\) 相接的**全部** \(X\)-check 行，保留原有冗余，令 \(c=|\chi|\)。
- \(T\in\mathbb F_2^{m_X\times c}\)：将 \(\chi\) 的坐标嵌回原 check 集。
- \(F=T^TH_XS^T\in\mathbb F_2^{c\times s}\)。因此 \(TF=H_XS^T\)。

添加两组比特 \(a,b\)，大小分别为 \(c,s\)。按 \((\mathrm{old},a,b)\) 排列列，按 CSS 类型重排 checks 后有

\[
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
\]

对易来自 \(TF=H_XS^T\)。最后的 \(I_c,F\) checks 必须保留：它们是分支的终止结构，不能把这层替换为测量终止层。

这里无需要求 \(F\) 行重为 2、满秩、连通或满足扩展阈值，也不需要额外计算 cycle basis 来完成这一 sticker。\(c=0\) 时公式仍成立，只剩复制比特和 \(ZZ\) links。

若 \(z\) 是支撑包含于 \(U\) 的目标，令 \(r=zS^T\)。乘第 \(r\) 组新增 \(Z\)-checks后，

\[
(z,0,0)+(rS,rF^T,r)=(0,0,r),
\]

因为 \(rF^T=0\)。这给出仅作用于 \(b\) 的叶子代表。

**必须运输完整的成对逻辑基**

一次 \(Z\)-branch 后，通用运输是

\[
\boxed{
J_X'=(J_X,0,J_XS^T),
\qquad J_Z'=(J_Z,0,0).
}
\]

仅仅把旧 \(J_X\) 补零，通常会与新的 \(Z\)-checks 反对易。程序需要同步更新所有被选和未被选的逻辑算符，才能核验完整逻辑空间的保持。

若对某个 \(X(x)\) 能解出

\[
aF=xS^T,
\]

则 \((x,a,0)\) 是与 \((x,0,xS^T)\) 等价的更紧凑运输。这个条件对应 Appendix E 的第一层清除方法：它要求与区域中全部相关 \(Z\)-逻辑类对易，不能仅由“与本次请求的 \(Z\) 目标对易”推得。当前基础运输使用一般有效的完整 lift。

纯 \(X\)-branch 交换 \(X,Z\) 即可得到；纯 \(X\) 叶子可以通过 `leaf_pauli="X"` 选择。`leaf_pauli="auto"` 按每个 sticker 的当前目标组选择：纯 \(X\) 组选 \(X\) 叶子，其他组选 \(Z\) 叶子。

**一般局部 Pauli 轴，包括 \(Y\)**

QWC 保证并集中的每个位置 \(q_j\) 有一致的局部非平凡 Pauli 轴 \(P_j=(p_{X,j},p_{Z,j})\)。定义

\[
P_X=\operatorname{diag}(p_X)S,
\qquad P_Z=\operatorname{diag}(p_Z)S.
\]

令当前辛 check matrix 为 \([A\mid B]\)，包括所有 check 行。局部反对易矩阵为

\[
R=AP_Z^T+BP_X^T.
\]

选择 \(R\) 的非零行得到 \(F\)，令 \(T\) 为行嵌入矩阵，则 \(TF=R\)。对于纯 \(Z\) 叶子，按实际的 old、junction、cap 行顺序，构造为

\[
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
\]

junction check 在 memory 上使用原来的 \(P_j\)，所以当 \(P_j=Y\) 时输出会含 mixed checks；不能把它硬塞成两个互相独立的 CSS check 矩阵。

任意 Pauli 的 lift 为

\[
u=xP_Z^T+zP_X^T,
\qquad
(x\mid z)\longmapsto(x,0,u\mid z,0,0).
\]

新增叶子因子与旧 memory 分离，literal Pauli 符号不变。选中的 QWC 目标有 \(u=0\)，然后可以乘 junction checks 清到叶子。

这与 Lemma 2.10 的离线局部 Clifford 变换等价：先把目标轴换到 \(Z\)，构造分支，再把 memory 坐标换回。代码直接按局部轴填矩阵，不要求物理实施这些 Clifford 门。选择 \(X\) 叶子时，交换新辅助比特上的 \(X/Z\) 分量即可。

**条件分支：原码各自行、列重量最多增加 2**

预算针对**输入码的每一条原始 check 和每一个原始物理 qubit**，分别比较自己的基线，不是只与一个全局最大重量比较。一般 Pauli check 的行重量和参与次数使用 `check_x OR check_z` 计数，所以 Y 在同一行、同一列中只计一次。设输入辛矩阵的支撑为 \(C_0=C_{0X}\lor C_{0Z}\)，则

\[
w_i^{(0)}=\sum_q(C_0)_{iq},\qquad
\Delta_q^{(0)}=\sum_i(C_0)_{iq}.
\]

程序实际构造待接入的基础 gauging gadget 预览，再核验

\[
\boxed{
 w_i^{\mathrm{preview}}\le w_i^{(0)}+\texttt{max\_check\_increase},\qquad
 \Delta_q^{\mathrm{preview}}\le\Delta_q^{(0)}+\texttt{max\_qubit\_increase}.
}
\]

默认两个增量均为 2；等于上限允许，严格超过才需要调整分支。这里的 preview 包含分支本身和预计的直接测量连接，不能只数当前目标在某个位置出现了几次，也不能只检查 branching 中间码。

预测使用每个目标对应的 **paired-boundary 基础 gauging 图**，按局部反对易点配对构造连接并生成实际矩阵，再计算重量。它与后续更换图、加 expansion 边、decongestion 或 cellulation 后的开销不同。若后续修改测量图，应重新检查完整矩阵。预览不承诺测量码的距离或扩展常数，也没有执行实际测量。

在这个基础连接规则下，一个目标的局部反对易矩阵为 \(F\)，第 \(i\) 行的反对易位置成对连接，所以直接接入会给该原 check 增加 \(\operatorname{wt}(F_i)/2\) 个辅助因子；给目标支撑中每个原 qubit 增加一次新 check 参与。多个目标的贡献累加。因此即使只测一个目标，也可能因某条 \(F_i\) 的重量为 6 而增加 3 个辅助因子，触发默认 +2 上限；整体对易只保证这个行重量为偶数，不保证它至多为 4。程序最终以实际预览矩阵计数为准。

- `branching_mode="adaptive"`（默认）：从所有目标直接接到 memory 开始；满足预算即不分支。否则搜索要移走的目标子集，并允许先把一个子组复制到共同的新根层，再在辅助区域内用完整子树分离该组。这样多个目标可以共享一次 memory 连接。
- 目标数不超过 12 时，使用有限的精确子集搜索；更多目标使用确定性的贪心计划，并在元数据中标明。这里的“精确”只针对实现所枚举的分支结构，不是所有可能 surgery 图的全局最优。
- 留在 memory 的目标可以继续相互 overlap，只要满足 QWC 和最终预算。移到不同末端分支的目标会分离；不再要求全部返回代表两两 disjoint。
- `branching_mode="always"`：恢复原来的完整均衡二叉树，适合需要全部目标分离或比较论文原始构造的场景；仍报告预计重量，但不以 adaptive 的增量限制拒绝完整树。
- 单目标也会检查预算：超限时自动尝试分支。`force_copy=True` 额外强制单目标增加一个 sticker，即使直接测量本来未超限。若预算对可构造计划仍不可满足，则报错，不返回伪装成功的计划。

新增辅助 check 和 qubit **没有原码对应项**，不能套用“各自原重量 +2”。输出另行报告它们的最大行、列重量。原码预算合格不等于整个扩展码已经获得统一的 LDPC 权重上界。

`result.routing` 记录模式、预算、`direct_targets`、`branched_targets`、`root_groups`、分支辅助比特数和 `selection_method`，并保存原码各项的预计增长。`result.measurement_preview` 独立保存实际基础测量预览矩阵、相应参考测量扇区以及重量增量。`result.check_x/z` 始终是**尚未测量的 branching code**，不把预览中的测量 checks 混入返回码。

预览中的 `target_outcome_signs` 选择一组满足目标依赖关系的参考扇区，不是预测真实实验结果；尤其不能把一组有带符号乘积关系的目标强行全部指定为正结果。`memory_row_growth`、`memory_column_growth` 是预览相对 branching code 的增量；与输入原码比较的总增长保存在 `routing.projected_original_row_growth` 和 `routing.projected_original_column_growth`。摘要中的 `projected_measurement_n` 与 `n` 分别指预览总比特数和分支码总比特数。

**树结构与伪代码**

在 `always` 模式下，目标按数量均衡分成两半，每个子组添加一个两层 sticker，再对子组递归。对 \(t\ge2\)，共有 \(2(t-1)\) 个 stickers，最长路径有 \(\lceil\log_2t\rceil\) 个 stickers。adaptive 模式只对选中子集执行相应结构；一个共享根组会先增加一次共同复制，再在该辅助区域中分离。单目标无需分支时，返回代表仍在原码上。

```text
BRANCHING_DEFORM(code, requested_targets, options):
    prepared = PREPARE_TARGETS(code, requested_targets, options)
    # 解析符号、验证 normalizer 和对易；按所选逻辑基处理 QWC

    if optimize_logical:
        candidates = 对超过阈值的代表乘原码稳定子，尝试减少 support
        验证候选的带符号稳定子等价见证
        按当前 branching_mode 估算候选的分支计划开销
        采用兼容且不增加计划开销的候选
        更新 prepared 的实际代表及对原始输入的等价见证

    if branching_mode == "always":
        plan = 全部目标的完整均衡树；单目标遵循 force_copy
    else:
        plan = 全部目标直接测量
        preview = 构造该计划对应的基础 gauging 检查矩阵
        if 原始各 check 行或物理 qubit 列超过自己的 +2 预算:
            搜索分支子集及共享根组
            解析计算候选分支及最终测量连接的行列增长
            只接受原始各行、列都不超预算的计划
            在搜索结构内优先减少 branching 辅助比特
        若没有可行计划则报错
        单目标 force_copy 强制额外复制，并仍核验预算

    checks = prepared.source_checks
    paired_basis = prepared.full_logical_basis
    current_targets = prepared.targets
    根据 plan 添加 stickers、运输完整 paired_basis 和所有目标
    为最终计划生成实际测量预览矩阵，复核解析预测
    移走的目标清到末端叶子；其余目标留在 memory

    验证 check 对易、符号一致、逻辑数 k 不变
    验证完整 paired_basis 的 normalizer 条件和辛配对
    验证每个返回代表与运输后的目标有带符号稳定子等价见证
    验证分支结构、没有新增逻辑读出以及预计行/列重量
    返回 branching checks、完整逻辑基、最终代表、见证、routing 和 preview

SPLIT_AND_BRANCH(group):
    将 group 均衡划分为 left 和 right
    for child in [left, right]:
        APPEND_AND_COPY(child)
        if len(child) > 1:
            SPLIT_AND_BRANCH(child)

APPEND_AND_COPY(group):
    U, local_axes = 该组当前代表的支撑并集和兼容局部轴
    R = 原 checks 与 local_axes 的逐点反对易矩阵
    F, T = 保留 R 的全部非零行及其嵌入
    按块矩阵添加一个两层 sticker
    用同一个 lift 运输完整逻辑基及所有当前目标
    对 group 中每个目标乘对应 junction checks，搬到本 sticker 的 b
    记录 support、F、T、辅助列、check 行和等价见证
```

树深由要分离的目标数量决定，不需要按原码距离 \(d\) 重复 \(d\) 层。这也不同于后续测量图为了 cycle decongestion 而使用的 thickening。

**API 示例与保存**

```python
from codesurgery.examples import four_qubit_code
from codesurgery.parallel_branching import branching_deform

code = four_qubit_code()  # HX = HZ = [1,1,1,1], [[4,2,2]]
result = branching_deform(
    code,
    targets=["ZZII", "ZIZI"],
    representatives="preserve",
    optimize_logical=False,
)
result.validate()
result.save("branched.json")
result.save("branched.npz")

# 混合逻辑 Pauli；选择完整树以确保各目标均分离。
mixed = branching_deform(
    code,
    logical_paulis=["YI", "IX"],
    leaf_pauli="Z",
    branching_mode="always",
)
mixed.validate()
```

第一个例子的两个重叠目标可以直接接入且均满足 +2 预算，因此默认不新增分支，返回原来的 4 个物理比特和 2 个逻辑 qubit。设置 `branching_mode="always"` 后，两条 root branches 各添加 \(1+2=3\) 个比特，得到 10 个比特和两个不重叠代表。两种情况都没有执行逻辑测量；预览中的测量辅助比特也不计入 `result.n`。

`check_x`、`check_z`、`check_signs` 是完整辛 check 描述，`leaf_x`、`leaf_z` 描述最终可接入测量的代表：可能在 memory，也可能在新叶子。保存的数据区分输入目标、实际用于分支的代表、逻辑坐标、稳定子见证和 branching 阶段。对于 \(Y\) 输入，应以辛矩阵为准。

`result.source` 中的代表及见证形成完整的审计链：

| 字段 | 含义 |
| --- | --- |
| `target_x/z/signs` | 用户原始请求转换得到的物理代表 |
| `pre_optimization_x/z/signs` | 按 `representatives` 完成初始 canonicalization 后、减重前的代表 |
| `prepared_x/z/signs` | 完成候选选择后实际用于 branching 的代表 |
| `initial_canonicalization_witnesses` | 原始请求 → 减重前代表的带符号稳定子见证 |
| `support_reduction_witnesses` | 减重前代表 → 实际采用代表的带符号稳定子见证 |
| `canonicalization_witnesses` | 原始请求 → 最终准备代表的合并见证，保留原字段名供兼容 |

这些见证均以原码 check 行编号表示，并在 `validate()` 中使用完整 Pauli 乘法核验。`canonicalization_witnesses` 在开启减重后覆盖两步变化，不能再将它仅解释为初始标准代表重建。结果中的 `equivalence_witnesses` 另外核验运输后的原始请求与最终叶子代表之间的带符号稳定子等价。

**独立命令行入口**

在项目根目录安装当前源码后运行：

```bash
python -m pip install -e .
python -m codesurgery.parallel_branching input.json -o branched.json
```

`input.json` 最小示例：

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

可把 `targets` 换为 `logical_paulis`，或换为 `pauli` 与 `logical_indices`。`logical_basis` 使用与 Python API 相同的 `jx/jz` 格式。当两个 checks 都为空时，用 `"n": ...` 明确物理宽度。

```bash
python -m codesurgery.parallel_branching input.json -o branched.npz \
  --force-copy --leaf-pauli Z --representatives auto
```

可显式设置预算或恢复完整树：

```bash
python -m codesurgery.parallel_branching input.json -o branched.json \
  --branching-mode adaptive --max-check-increase 2 --max-qubit-increase 2
python -m codesurgery.parallel_branching input.json -o full-tree.json \
  --branching-mode always
```

增长上限必须为非负整数；非法设置或无可行计划会报错，不覆盖已有输出。`--force-copy` 是单目标控制，多个目标的树结构由 `branching_mode` 和计划决定。

该入口输出 `stage="branching_only"`、`logical_measurement_performed=false`。CLI 参数覆盖 JSON 中对应选项；它不复用 CKBB 的执行入口。

CLI 默认启用 support 优化，`--no-optimize-logical` 显式关闭，`--optimize-logical` 可覆盖 JSON 中的 `false`。优化细节在 JSON 中指定，例如：

```json
{
  "optimize_logical": true,
  "logical_optimization_options": {
    "method": "heuristic", "threshold": null,
    "max_trials": 128, "seed": 0
  }
}
```

该片段与输入码、目标字段合并使用。若提供 `logical_optimization_options`，同时用命令行关闭优化会报错，输出文件不会被覆盖。

**静态距离证书与可选求解**

BB90 的默认参数实测如下；两种模式均先按同样的阈值和随机种子尝试 support 优化，单目标未启用 `force_copy`：

| 测量目标 | 采用的目标重量 | adaptive 分支码 n | always 分支码 n | adaptive 测量预览 n | adaptive 原行/列最大增长 |
| --- | --- | ---: | ---: | ---: | --- |
| Z1、Z2 | 12、10 | 90 | 145 | 123 | 2 / 2 |
| X1、Z2 | 14、22 | 90 | 177 | 144 | 2 / 1 |
| Y1 | 19 | 90 | 90 | 123 | 1 / 1 |
| Z1、Z2、Z3、Z4 | 12、10、14、14 | 287 | 317 | 362 | 2 / 2 |
| 全部 8 个 Z | 12、10、14、14、12、14、16、12 | 672 | 683 | 828 | 2 / 2 |

前 3 个 adaptive 案例没有添加 branching sticker。四 Z 和八 Z 案例均把 Z3 留在 memory，将其他目标分支。X1、Z2 中，Z2 的减重候选重量为 10，但与选定 X1 代表不满足 QWC，因此保留重量 22 的兼容代表。这些数值是当前有限计划族和启发式减重的结果，不声称全局最小。全部 core 都保留 k=8，core 与 preview 均通过对易检查。

**下面的距离保持结论仅适用于分支码 core，不适用于测量预览。**

一次 sticker 添加 \(c+s\) 个比特并增加同样的稳定子秩，因此 \(k'=k\)。更强的保持由逻辑 lift 与 retraction 共同验证。对于上面的 \(Z\) 叶子一般构造，定义二进制 retraction

\[
\rho(x_o,x_a,x_b\mid z_o,z_a,z_b)
=(x_o+z_bP_X\mid z_o+z_bP_Z).
\]

它把新 normalizer 映回旧 normalizer，把新稳定子映回旧稳定子，并在逻辑商空间上给出 lift 的逆。辅助 \(b_j\) 的一个因子至多映到原来的一个物理 qubit，故

\[
\operatorname{wt}_{\mathrm{Pauli}}(\rho(P))
\le\operatorname{wt}_{\mathrm{Pauli}}(P),
\qquad d_{\mathrm{branched}}\ge d_{\mathrm{source}}.
\]

这个 retraction 用来证明逻辑类和重量的性质，不声称是作用于任意 Pauli 的物理 Clifford 电路。这里的距离是静态码距；\(Y\) 的物理重量计为 1。

分支默认不启动额外的距离搜索。需要实际求解时可显式调用：

```python
from codesurgery.distance_preservation import code_distance

report = code_distance(result, method="gurobi", time_limit=60.0)
result.metadata["distance_calculation"] = report
result.save("branched-with-distance.json")
```

Gurobi 需要 `gurobipy` 与可用许可证。`time_limit=60.0` 是求解器阶段的秒数预算，包含输入处理、构造矩阵等步骤的总墙钟时间可以更长。超时报告可能只有上下界，不能把找到的低重量逻辑代表当成已经证明的精确码距。

CLI 显式开启：

```bash
python -m codesurgery.parallel_branching input.json -o branched.json --compute-distance
```

未给 `distance_options` 时调用现有 `code_distance` 默认设置，即 `method="auto"`、60 秒求解预算；auto 可以尝试 Gurobi并在不可用时使用其他后端。若要明确指定 Gurobi，在 JSON 添加：

```json
{
  "compute_distance": true,
  "distance_options": {"method": "gurobi", "time_limit": 60.0}
}
```

以上片段应与 `hx/hz` 和目标合并。原码中给出的 `distance` 是用户提供的参数；继承下界的可信程度取决于这个输入是否已有证明，程序不会把一个未经核验的声明升级为精确证书。

论文 Lemma 2.9 的 fault-distance 结论还依赖 syndrome extraction rounds；Appendix D 给出了时间边界和 padding 条件。当前矩阵导出及静态距离验证没有构造并验证具体 syndrome circuit，所以结果不等于电路级 fault-distance 证书。
