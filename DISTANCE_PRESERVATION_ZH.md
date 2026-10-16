# 自动计算码距、按需扩展与校验稀疏化

Python 的四个构造入口现在默认 `require_distance=True`：先计算原码和基础 deformed code 的静态码距，再根据实际结果决定是否增强扩展、降低 cycle 拥塞或分解长 cycle。可用 `require_distance=False` 显式关闭，返回基础构造。

码距后端默认 `auto`：可用时优先 Gurobi，单次 MILP 求解默认限时 60 秒。也可用 `distance_options={"distance_method": "gurobi", "distance_time_limit": 60.0}` 明确指定。超时立即返回该次计算的有效上下界，不继续做簇搜索。`distance_statement` 会显示 `8 < d <= 10` 这类结果；此时 `distance=None`，不能声称已经保证 d≥10。严格保距流水线仍按既有目标与图证书规则处理未确定结果。60 秒是每次求解器调用的限制，不是包含源代码距、前处理及所有图重建的整个 surgery 总时限。

默认稀疏度阈值取**水平拼接的 memory 校验块的最大行重量**，即用户指定的 `max rowwt([HX | M])`，不会把 `max rowwt(HX)` 与 `max rowwt(M)` 分别取最大再合并。完整输出校验超过该阈值时需要稀疏化；实际码距低于目标时进行 expansion，未确定时按后文的证书策略处理。仅有 `h<1` 或 `beta_d<1` 不足以证明码距降低。

成功返回意味着静态码距已证明至少达到目标，且完整校验矩阵的重量与参与次数满足上限。搜索预算不足而又没有有效距离定理证书、或构造达不到稀疏度限制时，直接报错。这些检查不等同于带噪声测量电路的故障距离。

## Python API

以下四个接口均默认自动处理：

- `gadget_deform`：单个完整目标的 explicit 构造；
- `gadget_pair_deform`：用 adapter 联合测量两个因子的乘积；
- `gauging_deform`：单个完整目标的 gauging 构造；
- `gauging_pair_deform`：两个 gauging gadget 的 adapter 乘积测量。

```python
import numpy as np
from codesurgery import CSSCode
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.distance_preservation import code_distance

h = np.array([
    [(j >> bit) & 1 for j in range(1, 16)]
    for bit in (3, 2, 1, 0)
], dtype=np.uint8)
code = CSSCode(h, h, name="Hamming [[15,7,3]]")  # 可以不填 distance
logical = "-YXXZZ" + "I" * 10

result = gauging_deform(code, logical)  # 默认计算、检查并按需处理
basic = gauging_deform(code, logical, require_distance=False)

print(code_distance(code))
print(result.metadata["source_distance_calculation"])
print(result.metadata["distance_calculation"])
print(result.metadata["distance_postprocessing"]["sparsity_threshold"])
result.save("deformed-with-computed-distance.json")
```

API 不会等待交互输入。`distance_options` 为字典，仅在自动模式生效；关闭自动处理时忽略其中的可选上限。未给逻辑基时仍使用 Zhang–Li standard form，保留原有 Y、混合 XZ、目标整体负号及实际逻辑读出核验。

```python
result = gauging_deform(
    code, logical,
    distance_options={
        "distance_max_candidates": 2_000_000,
        "max_base_degree": 12,
        "max_layers": 128,
    },
)

# 对已有基础结果执行同样的处理。
from codesurgery.distance_preservation import ensure_distance
result = ensure_distance(basic, distance_max_candidates=2_000_000)
```

## 真正计算的码距

设完整辛校验矩阵为 \((C_X\mid C_Z)\)，其 stabilizer 行空间为 \(\mathcal S\)。程序计算

\[
d=\min_{(x,z)\in\mathcal S^{\perp_s}\setminus\mathcal S}
\operatorname{wt}(x\lor z).
\]

`x OR z` 使 Y 在一个物理比特上只计一次重量。先验证校验对易，再用 GF(2) 行空间检验剔除 stabilizer；不能把最小非零 stabilizer 的重量当作码距。

公开计算接口为：

```python
from codesurgery.distance_preservation import code_distance, stabilizer_distance

report = code_distance(code, max_weight=None, max_candidates=2_000_000)
report = stabilizer_distance(CX, CZ, max_weight=None, max_candidates=2_000_000)

# 只需要证明 d >= 3 时，可检查重量 1、2；也可能通过上下界相遇得到精确 d。
report = code_distance(result, max_weight=2)
```

枚举/簇后端按重量从小到大进行。CSS 情况只需枚举纯 X、纯 Z 逻辑候选；一般带 Y 的校验使用完整的局部 X/Y/Z 枚举。某个重量层仅搜索了一部分就耗尽预算时，不能宣称已排除整层。Gurobi 后端对完整辛模型进行整数优化；正常限时结束时可以用有效的最终目标下界建立距离下界。所有后端从 centralizer 基及 RIS 搜索寻找并核验非平凡逻辑代表，给出距离上界。

| 报告字段 | 含义 |
| --- | --- |
| `status="exact"` | 上下界相遇，或按重量搜索找到最小逻辑算符；`distance` 是精确值 |
| `status="bounded"` | 已有非平凡下界，但上下界尚未相遇；`distance=None` |
| `status="unknown"` | 尚未建立非平凡下界；仍报告已知上下界 |
| `status="no_logicals"` | 输出没有剩余逻辑比特；`distance=None`，`distance_is_infinite=True` |
| `proven_lower_bound` / `upper_bound` | 已证明的距离下界／由实际逻辑见证给出的上界 |
| `distance_interval` / `distance_statement` | 含端点的整数上下界／如 `8 < d <= 10` 的可读区间 |
| `witness` | 上界见证的 X、Z 向量、Pauli 字符串和重量 |
| `completed_weight` / `checked_candidates` | 完整排除到哪个重量，以及枚举候选数 |

这是指数复杂度的有限预算算法，适合小尺寸精确核验；不会把搜索超时包装成精确答案。`max_candidates` 限制枚举候选数，矩阵验证及上界见证的线性代数工作另行执行。对带 gauge qubits 的对象，该接口计算其 stabilizer-code distance，不能直接作为 subsystem 的 dressed distance。

### 如何确定目标距离

1. **原码未提供 `distance`**：计算原码距离，只有得到精确值才自动将其作为目标。预算内不能确定时，报错并提示增加预算或提供可信距离下界。
2. **原码已提供 `distance=d`**：把它当作用户声明的可信距离下界，仍实际计算原码并报告结果。若发现逻辑算符重量小于该声明，则报错；不会继续使用已被反证的距离。
3. **输出码距**：计算真实完整 deformed code。已证明的下界达到目标就通过；找到小于目标的实际逻辑见证才确认距离不足并触发 expansion。
4. **计算尚未确定**：若现有图距离定理能严格证明达到目标，可凭该证书通过，但 `distance_calculation["distance"]` 仍保持 `None`。若还没有证书，默认 `unknown_distance_policy="certify"` 继续在固定预算和上限内寻求可认证的辅助图。此时补边的原因明确记录为“证书尚未建立”，不是声称已经证明距离不足。仍无法认证则报错；`unknown_distance_policy="error"` 保留遇到未知即停止的行为。

`source_distance_calculation` 与 `distance_calculation` 始终区分输入声明、实际计算结果和图定理提供的下界。

## 稀疏度阈值：水平拼接的最大行重量

对写成 \([H_X\mid M]\) 的 dressed memory 校验，阈值为

\[
\tau=\max_i\operatorname{wt}([H_X\mid M]_i)
=\max_i\bigl(\operatorname{wt}((H_X)_i)+\operatorname{wt}(M_i)\bigr).
\]

这里的两项必须属于**同一行**。比如 HX 两行重量为 4、8，M 两行重量为 3、0，则阈值是 `max(7, 8)=8`，不是 `max(4,8)+max(3,0)=11`。

纯 Z 目标严格采用 `[HX | M_X]`，纯 X 目标采用对偶块 `[HZ | M_Z]`。这里 M_X、M_Z 是完整 matching 中分别对应 HX、HZ 的行；未 dressing 的另一类 CSS 校验不会暗中抬高阈值。对 Y 或混合 XZ 目标，把所有原码校验按统一 memory 行顺序排列为 \((C_X\mid C_Z)\)，对应 matching 为 \(M\)，采用

\[
\tau=\max_i\left[\operatorname{wt}((C_X)_i\lor(C_Z)_i)
+\operatorname{wt}(M_i)\right].
\]

对于 CSS 输入，完整 memory 行顺序为 HX 的所有行，然后 HZ 的所有行；`sparsity_reference_rows` 记录实际选取的参考行。纯目标未纳入参考块的另一类校验仍必须通过最终完整行重量检查：若它已经超过阈值，而图处理无法降低它，就报错，不用它来放宽阈值。这个阈值在**初始构造完成后、任何扩展或稀疏化之前**固定，后续不会随新增校验变重而自动放宽。

默认要求最终所有 memory、vertex 和 cycle 校验均满足

\[
\max_i\operatorname{wt}((\widetilde C_X)_i\lor(\widetilde C_Z)_i)\le\tau.
\]

默认 cycle 行重量上限同样取 \(\tau\)。用户显式传入 `max_cycle_weight` 时取其与完整行上限的较小值；显式 `max_check_weight` 可以覆盖默认完整行上限，并在审计信息中同时保留原始 \(\tau\) 和实际采用的上限。

列参与次数另设上限，不能仅凭行重量有限就声称是 LDPC。设输入 memory 比特最大参与次数为 \(q_0\)，matching 列上限为 \(m_c\)，cycle 列上限为 \(c_N\)，默认完整列上限为

\[
q_{\mathrm{check}}\le\max\{q_0+1,\ 2+m_c+c_N\}.
\]

输出逐行、逐列检查完整 Pauli 支撑 \(\widetilde C_X\lor\widetilde C_Z\)，同时检查图、M、N 的各项指标。仅 cellulation 未必能解决过重的 vertex 校验；如果分层或补边后仍超出固定阈值，明确报错，不自动提高 \(\tau\)。输入完全没有 memory 校验时阈值为零，必须显式提供可行的 `max_check_weight`。

## 三个处理步骤何时执行

### 1. Expansion：距离不足，或尚需建立保距证书

若实际距离满足目标，即使图的 \(\beta_d<1\)，也不会只为了使充分条件成立而补边。若实际逻辑见证证明距离低于目标，或数值距离仍未知且尚无保距证书（默认策略），则保留已有边，在度数预算内按固定随机种子逐级尝试提高度数，并对不足的割进行定向修复。每个候选先重建完整稀疏校验、计算实际距离；数值结果仍未知时可以用严格的图证书补足下界。报告始终区分“已证明距离不足”和“证书尚未建立”。新增边在 matching 中对应零列。旧的 Hamilton 环策略可用 `expansion_strategy="hamiltonian"` 显式选择。

增强阶段的度数上限同时受 `max_base_degree` 和 `完整行上限 - 1` 限制，为端口上的物理 Pauli 留出重量。无法在预算内得到有效图，直接报错。这是本实现选择的有限搜索算法，依据 [Gauging 论文的常数度数扩展增强思路](https://arxiv.org/html/2410.02213)，不声称论文指定了这一种补边次序。

单目标 explicit 可以保留一般偶数 Tanner 行；只有确实需要图后处理、且当前 incidence 不是连通普通图时，才转换为 gauging 配对图，并重新检查实际码距。这不会绕过 explicit 或 adapter 的原始逻辑读出与图条件检查。

### 2. Decongestion：降低每条边参与的 cycle 数

若完整 cycle 生成集已有合格的行、列重量，就保留它。否则默认先在原图上搜索等价的稀疏 cycle 基：从多个根的 BFS 生成树收集候选，优先选择较短的线性独立 cycle；必要时再用边参与频次排序尝试降低拥塞。只有新基完整张成同一核空间，且行重量、边拥塞同时达标时才采用。图、端口、matching 和辅助比特数不变，adapter 组合距离证书也会保留并从实际图重新核验。

此搜索是确定性的有限尺寸启发式，不保证最优。默认最多搜索 64 个根、保留 10000 个不同候选，候选存储同时受 `max_entries` 限制；预算耗尽不会返回缺秩的 cycle 集。设置 `optimize_cycle_basis=False` 可关闭这一步。

如果没有找到同时达标的基，则保留原回退策略：重新计算简单 fundamental cycle basis，用确定性贪心着色把共享顶点的 cycles 放到不同层，构造 \(B\square P_L\)。每个原 basis cycle 只放入一层，再加入相邻层之间的全部四边形校验。

相同层待分解的 cycles 不共享顶点，因此后续 chord 不会在同一顶点无限堆积。层数受到 `max_layers` 和矩阵大小预算的硬限制。这个有限尺寸构造不实现 Freedman–Hastings 的最优低拥塞算法，不承诺多对数层数或论文最优开销。

### 3. Cellulation：分解超过固定阈值的长 cycle

对超过 cycle 行重量上限的简单 cycle，使用交替耳切割，添加新 chord 并用三角形校验替换长 cycle。每个原顶点至多增加两条 chord；即使端点间已有边，也添加独立的新辅助边。

不超过阈值的 cycle 保留。分层用到重量 4 的四边形，因此需要这一操作时，阈值必须至少容纳重量 4；更低阈值只有在原生成集或自动优化后的等价 cycle 基直接满足限制时才能保留。

设增强后的基图最大度数为 \(\Delta\)，则该图构造满足

\[
\Delta_{\mathrm{final}}\le\Delta+4,\qquad
\max\operatorname{rowwt}(N)\le\text{cycle 行上限},\qquad
\max\operatorname{colwt}(N)\le\max(3,\Delta).
\]

这些辅助图界只是中间保证，仍须通过完整输出矩阵的固定行阈值和列上限。每次重建后再次计算实际码距；若稀疏化之后出现距离不足，再进行一次受预算限制的修复检查。最后未满足要求则报错。

## 图距离证书与加层后的核验

辅助图 \(B\) 的物理端口为 \(U\)，使用端口相对扩展充分条件

\[
\beta_d(B,U)=\min_{A:\,\min(d,|A\cap U|,|U\setminus A|)>0}
\frac{|\partial_B A|}{\min(d,|A\cap U|,|U\setminus A|)}\ge1.
\]

无需严格大于 1；加 dummy vertices 或分层后也无需整张图具有全局 \(h\ge1\)。距离条件见 [Universal adapters 第 2.3 节](https://arxiv.org/html/2410.03628#S2.SS3)。它是充分条件，实际距离计算可以证实某个不满足它的小图仍保持码距。

小图精确枚举割。较大图在预算允许时使用任意精度整数 Bareiss 消元，对

\[
A=nL-2nI+2J
\]

的一个 \((n-1)\times(n-1)\) 主子矩阵验证严格正定，从而证明 \(\lambda_2(L)>2\)，得到全局扩展再推出相对扩展。浮点特征值、随机抽样割均不直接作为距离证书。

分层保留第 0 层的基图和全部原端口。任意最终割的边界至少包含它在第 0 层基图中的边界，而相对扩展分母只依赖这些不变端口。因此已证实的基图相对扩展下界可以保留。程序重新核验实际矩阵里的基图、端口及证书，不信任传入的一个布尔值。若距离枚举发现与图证书冲突的实际逻辑见证，直接报错。

## 参数

`distance_options` 或 `ensure_distance` 支持：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `distance_max_candidates` | 2,000,000 | 每次实际码距搜索的 Pauli 候选预算 |
| `distance_max_weight` | `None` | 不另设重量截断；仍受候选预算限制 |
| `max_check_weight` | `None` | 默认固定为初始 dressed memory 的 \(\tau\) |
| `max_cycle_weight` | `None` | 默认采用完整行上限；显式值只会进一步收紧 cycle 限制 |
| `max_qubit_degree` | `None` | 按完整列参与次数公式自动设置 |
| `max_base_degree` | 12 | 扩展阶段的度数上限，同时受完整行重量限制 |
| `max_cycle_congestion` | `None` | 默认 `max(4, max_base_degree)` |
| `optimize_cycle_basis` | `True` | 分层前先搜索完整等价的稀疏 cycle 基；`False` 复现旧回退流程 |
| `cycle_basis_max_roots` | `64` | 最多尝试的 BFS 根数 |
| `cycle_basis_max_candidates` | `10000` | 最多保留的不同 cycle 候选数；同时受 `max_entries` 约束 |
| `unknown_distance_policy` | `"certify"` | 距离未知且无下界证书时继续受预算限制的图修复；`"error"` 则立即报错 |
| `max_layers` | 128 | 分层硬上限 |
| `max_matching_weight` | `None` | 默认固定为初始 M 的最大行重量 |
| `max_matching_congestion` | 1 | 每条辅助边参与 memory dressing 的次数上限 |
| `seed` / `attempts` | 0 / 12 | 扩展搜索种子与重试预算 |
| `exact_limit` | 20 | 精确图割枚举的顶点上限，与实际码距枚举不同 |
| `certificate_limit` | 128 | 整数谱证书的顶点上限 |
| `max_entries` | 50,000,000 | 稠密矩阵及相关图运算的资源预算 |

有限例子满足行列上限不自动证明一个渐近码族是 LDPC；需要输入族本身为 LDPC，并且这些上限在整个码族中统一有界。

## 命令行

交互终端中未提供开关或 JSON 选择时，仍询问是否保证码距；选择 `y` 后执行自动计算和按需处理，`n` 或回车保持基础模式。**不再询问一个缺失的原码距离**，而是尝试计算。非交互调用未选择时默认基础模式并输出提示。

```bash
python -m codesurgery.gauging_measurement input.json --require-distance -o certified.json
python -m codesurgery.gauging_measurement input.json --no-require-distance -o basic.json
```

`input.json` 由用户提供，包含原码 `hx,hz` 及目标算符。仓库的完整 BB90 使用入口为 [examples/bb90_compare.py](examples/bb90_compare.py)。`--require-distance` / `--no-require-distance` 覆盖 JSON 的 `require_distance`。`--distance d` 仍可声明可信原码距离下界，但本身不打开自动处理。JSON 可添加如下字段：

```json
{
  "require_distance": true,
  "distance_options": {
    "distance_max_candidates": 2000000,
    "max_base_degree": 12
  }
}
```

还需提供原有 `hx`、`hz`、`logical` 或双因子 `logicals`。缺失距离且搜索不能确定时，命令行和 API 都给出明确错误，不冒充已知码距。

## 总伪代码

```text
BUILD(code, target_or_two_factors, require_distance=True, options=None):
    result = 原有 explicit / gauging / adapter 构造
    核验对易、相位、实际读出空间和剩余逻辑维数
    若 require_distance=False: 返回基础 result

    source_report = 计算原码真实静态距离（有预算）
    若 code.distance 未提供:
        要求 source_report 有精确有限距离；令它为目标 d
    否则:
        令可信输入下界为目标 d
        若实际逻辑见证反证输入下界：报错

    纯 Z: tau = max rowwt([HX | M_X])
    纯 X: tau = max rowwt([HZ | M_Z])
    Y / 混合 XZ: tau = max_i(wt(CX_i OR CZ_i) + wt(M_i))
    # 以上均取初始水平 memory 拼接块，不暗中用其他行抬高阈值
    固定完整校验行上限 = 显式 max_check_weight 或 tau
    另外固定 cycle、matching、列参与次数和资源上限

    result_report = 计算 result 的真实静态距离
    若距离已达标且完整矩阵满足全部稀疏度上限:
        返回 result 与计算报告（不补边、不加层）
    若距离未知且没有有效定理下界:
        unknown_distance_policy="error"：报错
        默认 "certify"：保持 unknown 记录，继续寻找保距证书

    必要时把一般 Tanner incidence 转为普通图，重新计算距离
    若找到重量小于 d 的实际逻辑，或默认策略下仍缺少下界证书:
        在度数和固定行上限内尝试增强扩展，记录实际触发原因
    若 cycle 重量或拥塞超过上限:
        先在原图上优化完整 cycle 基；只有行重和拥塞同时达标才替换
        若优化未能达标，再执行 decongestion 分层和 cellulation

    重建全部带相位校验，重新计算真实静态距离
    如有新的实际距离不足，作一次受预算限制的修复检查
    验证 NB=0、cycle 空间完整、对易、相位与唯一目标逻辑读出
    检查完整校验的每行重量、每列参与次数及所有辅助界
    证书不足、超重或超预算：报错，绝不自动放宽 tau
    返回 deformed code、真实距离报告、定理下界及稀疏度审计
```

审计位于 `metadata["distance_postprocessing"]`：保存 `sparsity_threshold`、实际采用的行上限、初始和最终码距报告、`expansion_trigger`、补边数量、层数、cellulation 及最终完整矩阵指标。

其中 `sparsification.cycle_basis_optimization` 保存优化预算、候选数量、是否截断、是否换基、前后指标和空间核验。仅换基时 `sparsification.applied=False`（无图拓扑变化）、`layers=1`，但嵌套的 `cycle_basis_optimization.changed=True`。保留的 `adapter`/`graph_design` 旧 cycle 行号描述原连接构造；实际输出校验以当前 `metadata.cycle_basis`、`row_layout.cycles` 为准。

## 小码构造结果记录

以下是此前的小码构造记录，用于说明距离修复与稀疏化的不同触发条件；发布目录不包含这些历史运行的矩阵输出。运行自己的构造时，应以本次返回的证书和完整矩阵审计为准。

| 例子 | 基础 → 自动的总比特数 | 实际码距 | 处理结果 |
| --- | ---: | --- | --- |
| Hamming 15 比特码，带负号 Y 混合目标 | 19 → 19 | 3 → 3 | 相对扩展为 1/2，但真实码距足够；阈值 9，保留原矩阵 |
| 两个 4 比特码的 XZ 乘积，单边薄 adapter | 11 → 15 | 1 → 2 | 实际距离不足才增强扩展；固定行阈值 5，最终最大列参与次数 6 |
| 4 比特码的 XX，8 顶点环、相邻两个端口 | 12 → 17 | 2 → 2 | 固定行阈值 5；长度 8 cycle 分解为三角形，加入 5 条 chord，无需 expansion |

第三个例子只需要一层；应从实际报告读取层数。更复杂 cycle 拥塞可触发多层；若最后 vertex 校验超出固定行阈值，程序按约定报错。
