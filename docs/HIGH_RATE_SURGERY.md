# High-rate surgery：共享辅助系统的多逻辑测量

独立模块：`codesurgery.high_rate_surgery`。本模块对应 Guo Zheng、Liang Jiang、Qian Xu 的 [High-Rate Surgery: towards constant-overhead logical operations, arXiv:2510.08523](https://arxiv.org/abs/2510.08523)，主要采用通用码的随机辅助系统构造思路。它与 `ckbb`、`explicit_gadgets`、`gauging_measurement`、`parallel_branching` 分开，也不是 linear-PCPP surgery 的实现。

论文利用一个共享辅助系统同时测量一组逻辑算符；附录 E 的 Algorithm 1 包含初始化、增加辅助连接、按需分层等步骤，Algorithm 2 用来寻找低重量 dressed logical。论文中的有限码长计算依赖随机搜索，不能把未找到低重量错误直接解释为严格的码距下界。这里将代数验证、搜索结果和严格距离认证分别记录。

## 输入和测量含义

Python API 接受 `CSSCode(hx, hz, distance=None, name=...)`。未提供 `logical_basis` 时，使用现有的 Zhang–Li standard form 成对逻辑基。

```python
from codesurgery.examples import four_qubit_code, ibm_bb90_code
from codesurgery.high_rate_surgery import high_rate_deform

code = four_qubit_code()

# 分别测 Z1 和 Z2：输出两个逻辑测量结果。
parallel = high_rate_deform(code, logical_paulis=["ZI", "IZ"])

# 只测一个乘积 Z1 Z2：输出一个结果；不能因此读出各因子。
joint = high_rate_deform(code, logical_paulis=["ZZ"])

parallel.validate()
parallel.save("parallel.json")
joint.save("joint.npz")
```

`targets` 与 `logical_paulis` 的字符串长度不同：

| 输入 | 含义 |
| --- | --- |
| `targets=["ZZII"]` | 一个长度为物理比特数 `n` 的物理 Pauli 代表 |
| `targets=[{"x": [...], "z": [...], "sign": 0}]` | 显式物理二进制向量；`sign=1` 表示负号 |
| `logical_paulis=["ZZ"]` | 一个长度为逻辑比特数 `k` 的逻辑 Pauli 乘积 |
| `logical_indices=[0, 2], pauli="Z"` | 分别测量零起始编号的第 0、2 个 Z 逻辑基算符 |
| 未给任何目标 | 默认分别测量全部 Z 逻辑基算符 |

多个目标列表始终表示多个可同时读出的观测量；联合乘积测量必须写成**一个**完整 Pauli 字符串或一个完整 `x/z` 对。单个 `Y1` 也应作为一个目标输入，不能拆成 `X1` 和 `Z1` 两次测量。

当前构造接受共同纯 X、共同纯 Z，以及 qubit-wise commuting（QWC）的混合 Pauli 集。QWC 要求任意两个目标在每个物理比特上作用相同 Pauli，或至少一个作用 I。单个目标自动满足 QWC，包括 Y 或 XZ 乘积。仅整体对易但不满足 QWC 的集合会报错，不会自动改成分别测量其 X/Z 分量。

QWC 混合构造使用局部 Pauli 轴描述和二进制辛矩阵；这一步属于离线矩阵构造，不表示硬件必须执行相应换基门。混合构造不会自动继承纯 CSS 路线的距离证明，但当 `gauge_qubits == 0` 时，可以直接计算完整辛 stabilizer code 的静态距离，因为此时所有剩余逻辑自由度都是要保护的原码逻辑。

如果混合构造仍有 `gauge_qubits > 0`，则普通 stabilizer distance 可能把辅助 gauge 逻辑误当作受保护逻辑。当前尚未实现这种一般辛子系统的 dressed-distance 后端，此时明确报告未认证；不能把 `g=0` 的做法无条件套用。

## 输出矩阵和完整读出核验

对于纯 X 测量，采用以下维度约定说明共享结构。`S` 的每一行选出支撑并集中的一个原码比特；`F` 的列对应共享的连接校验，行对应辅助比特；`M` 将原 Z 校验连接到辅助比特；`G` 给出辅助 Z 校验。

\[
\widetilde H_X=\begin{pmatrix}H_X&0\\S&F^T\end{pmatrix},\qquad
\widetilde H_Z=\begin{pmatrix}H_Z&M\\0&G\end{pmatrix}.
\]

对易要求

\[
MF=H_ZS^T,\qquad GF=0.
\]

后续分层时 `S` 会扩展为相应的连接映射，上式中的共享矩阵也随之扩展。纯 Z 测量交换 X、Z 的角色。一般 Pauli 输出统一用 `check_x`、`check_z`、`check_signs` 表示；含 Y 的行在重量计算时只算一个非 I 位置：

\[
w_i=\sum_q (X_{iq}\lor Z_{iq}),\qquad
\deg(q)=\sum_i (X_{iq}\lor Z_{iq}).
\]

初始化不能仅仅对每个目标取受限矩阵后独立拼接。必须检查共享支撑上是否还存在未请求的逻辑读出，并增加逻辑筛选约束，使实际读出的逻辑空间恰好是输入目标的张成空间。

最终验证也必须直接针对完整输出矩阵进行：

1. 校验两两对易，且依赖关系不产生 `-I`。
2. 消去所有辅助比特支撑，得到仅作用于原码的校验乘积。
3. 计算这些乘积在输入成对逻辑基下的签名。
4. 验证签名行空间恰好等于请求目标的签名行空间。
5. 验证目标测量见证及符号。

因此，输入 `logical_paulis=["XX"]` 时不允许额外读出 `XI` 或 `IX`；输入 `["XI", "IX"]` 则明确允许读出这两个独立结果和它们的乘积。输出的 `measurement_witnesses[i]` 选择第 i 个目标的校验乘积，`readout_sign_corrections[i]` 给出其符号修正：

\[
\text{目标 }i\text{ 的结果}=(-1)^{c_i}\prod_{j:w_{ij}=1}\text{带符号校验 }j\text{ 的结果}.
\]

这里输出测量规则和一个一致的符号约定，并未实际运行设备或产生测量样本。重复目标和带相反符号的同一目标也必须满足相应的结果依赖关系。

## 默认优化、稀疏度与距离

```python
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
```

默认先尝试在同一逻辑类内减小目标支撑，再构造共享辅助系统。减重不能改变目标逻辑类或符号，也不能把目标集变成不兼容的测量集合。

未指定稀疏度阈值时，以输入原码的**最大**行重量和**最大**物理比特列度数分别加 2：

\[
w_{\max}^{\rm limit}=\max_i w_i(H_X,H_Z)+2,\qquad
\Delta_{\max}^{\rm limit}=\max_q\deg_q(H_X,H_Z)+2.
\]

这两个阈值检查完整 merged code，包括新辅助校验和新辅助比特。它们与 `parallel_branching` 的“每条原始行/列各自最多增长 2”是不同的策略。

`ensure_distance=True` 表示启动距离检查及按需修复；**不是调用开始时就已经保证码距**。`distance_time_limit=60` 是完成初始化后，整个距离检查及增长循环共享的 60 秒预算；多个求解器调用会分用剩余时间。初始化、支撑优化和最终验证不计入该预算，正在执行的矩阵操作也不会被强制中断，因此它不是整个 API 的严格一分钟超时。这里使用限时的 protected-distance 后端：纯 CSS 路线计算 dressed distance；gauge-free 混合路线计算完整 Pauli 的静态距离。辅助 gauge 逻辑不应误计为受保护逻辑。`distance_method="auto"` 可用时优先 Gurobi；`"gurobi"` 则只选择该后端，不自动改用其他求解器。若缺少安装包或可用许可证，保留已有上下界并报告认证未完成；严格模式据此失败。另可选择 `"enum"` 逐重量枚举或 `"heuristic"` 随机寻找低重量代表。

距离报告必须区分：

- 找到重量 `u` 的非平凡逻辑代表，只能证明 `d <= u`。
- 严格排除所有重量 `<= l` 的相关逻辑错误，才能证明 `d > l`。
- 超时得到界时保留上下界，不能将“搜索未找到”改写为已达到目标距离。
- 基础构造、带 gauge 自由度且尚无对应距离后端的混合构造、未完成认证的搜索结果应明确显示未认证状态。

对于 gauge-free 的 QWC 混合构造，距离后端使用完整 `(x,z)` Pauli 错误，重量为 `wt(x OR z)`，每个 Y 只算一次。增长评分使用这些已验证短错误的辅助 X 分量；评分之后仍需重新计算完整 Pauli 距离，而不是仅用 X 分量重量宣称距离达标。该报告的 `scope` 是 `protected_mixed_stabilizer_gauge_free`。

`strict=False` 允许返回代数合法、满足行列重量上限、但距离仍未完成认证的候选结果，调用方应查看 `result.metadata["status"]` 和摘要。`strict=True` 用于要求完成所请求距离认证的工作流；若仍无法认证，Python API 抛出 `DistanceCertificationError`，其 `.result` 保存有效候选。稀疏度是返回结果的硬约束，不因 `strict=False` 放宽。

CLI 严格模式失败时返回退出码 2；如果显式提供了 `-o` / `--output`，仍会将经过验证的未认证候选保存到该路径，摘要含 `strict_failed=true`、`candidate_saved=true`。没有显式输出路径时不创建候选文件。这一机制保留 deformed code，同时让自动化脚本明确看到认证失败。

摘要中的 `protected_distance_certified` 只对应幸存原码逻辑子系统的静态 dressed distance；`distance_lower_bound`、`distance_upper_bound` 给出求解器当前的界。`measurement_fault_distance_certified` 单独保留为 false：本模块没有仅凭这些矩阵认证整个测量记录或门调度的故障距离。即便测量所有原码逻辑后 `protected_k=0`，也不能据此声称测量记录的可靠性已得到证明。

常见 `metadata["status"]` 含义如下：

| 状态 | 含义 |
| --- | --- |
| `constructed_distance_disabled` | 用户关闭距离检查，只返回通过代数与稀疏度核验的构造 |
| `protected_distance_certified` | 当前候选的 protected dressed distance 下界达到目标 |
| `constructed_mixed_distance_unverified` | 混合 Pauli 构造有效，但仍有额外 gauge 自由度，当前后端不支持其一般辛子系统距离 |
| `constructed_all_logicals_measured_readout_distance_unverified` | 已测完全部原码逻辑；测量结果记录的故障距离仍未认证 |
| `constructed_no_distance_target` | 输入没有可信距离目标，返回计算到的界 |
| `constructed_algebraic_distance_unverified` | 使用显式代数辅助块和连接映射构造，但所请求距离尚未认证 |
| `distance_inconclusive` / `distance_time_limit` | 当前界不足以完成认证，或距离/增长预算已耗尽 |
| `step_limit` / `no_admissible_growth` | 达到步数上限，或未找到满足重量限制的有效增长 |

本模块增长候选的评分依据是能检测多少个已找到的低重量代表，之后必须重新计算距离。这里没有单独计算并声称辅助超图的扩展常数 `h > 1`；随机加连接本身不构成扩展性或距离证书。初始化若找不到满足重量上限、且排除了额外逻辑读出的分离约束，会直接报错；这只是有限搜索失败，不是不存在这种构造的证明。

增长时默认给末层校验预留一个连接位置，防止在层内用尽重量预算后无法再增加 bridge。按论文式 E7，层内组合增长只在该层 kernel 已与首层一致时使用；未完成的新层可以通过相邻层复制补齐，即使该次复制没有直接命中当前错误见证。这里仍是有界启发式搜索，没有实现完整的 HGP 最坏情况回退，也不承诺任意输入都能在给定重量和时间预算内达到目标距离。

## 伪代码

```text
HIGH_RATE(code, targets, options):
    basis = supplied_basis or standard_form(code)
    parse targets with phases; reject trivial/non-normalizer targets
    check target commutation and supported QWC condition
    optionally reduce support in the same signed logical classes

    support = union of target supports
    build shared restricted boundary and transversal attachment maps
    add selector constraints to remove unwanted logical readout
    build compatible auxiliary checks
    verify exact requested logical readout space

    repeat within max_steps / max_layers:
        construct full merged check matrices
        verify commutation, phases, exact logical readout
        measure all row weights and column degrees
        if distance preservation requested:
            compute rigorous distance bounds and short-error witnesses
        if requested bounds and certificates are satisfied:
            stop
        propose target-preserving auxiliary connections
        score proposals against available short-error witnesses
        reject any proposal violating the configured weight limits
        if useful degree-feasible proposal exists:
            add it and revalidate
        else if another layer is permitted:
            extend the shared ancilla layer
        else:
            stop with explicit unmet conditions

    fail if no weight-feasible, algebraically valid construction is available
    strict mode: fail if requested distance guarantees are not established
    return validated matrices, target witnesses, bounds, and search history
```

上述搜索、候选选择和停止预算是本实现的有限规模算法选择。不能仅因采用论文的构造框架，就把所有随机候选标为已经满足论文的充分条件；认证以实际输出的证据为准。

## 代数 HGP 辅助块

另提供底层工具 `hgp_ancilla(HB, HD)`，返回两个二进制矩阵 `(B, C)`：

\[
B=\begin{pmatrix}I_s\otimes H_D\\H_B\otimes I_u\end{pmatrix},\qquad
C=\begin{pmatrix}H_B\otimes I_t&I_r\otimes H_D\end{pmatrix},
\]

其中 `HB.shape=(r,s)`、`HD.shape=(t,u)`，因此 `CB=0`。此工具只构造辅助边界块；它不会自动为任意原码寻找合适的 HGP 分解、连接映射或验证论文的 soundness 条件。`high_rate_deform` 默认走上文的通用随机路线，不能把一次 `hgp_ancilla` 调用等同于已经完成整个 surgery 或距离认证。

已有辅助块和连接映射时，可以使用独立代数入口：

```python
high_rate_from_ancilla(
    code, boundary, cycles, gamma0, gamma1,
    targets=None,
    pauli="X",                         # 只支持纯 X 或纯 Z；这里默认 X
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
```

设 `boundary.shape=(a,s)`，则 `gamma1.shape=(n,s)`，`gamma0.shape=(m_opposite,a)`；纯 X 时 `m_opposite=len(code.hz)`，纯 Z 时为 `len(code.hx)`。此入口接受原码相反类型校验对应的 `gamma0` 行，区别于随机构造内部 state 存储的完整原校验行。

它会验证链映射条件、完整 Pauli 对易和符号、实际逻辑读出空间以及全部行列重量，并按选项计算距离；**不会自动搜索或修复输入连接映射，也不启动随机增长循环**。例如纯 X 情况要求

\[
\Gamma_0B=H_Z\Gamma_1,\qquad CB=0.
\]

下面的可运行例子把已有随机构造的辅助块及映射重新送入代数入口，演示二者如何衔接；它不表示已经为任意 HGP 因子自动找到了连接：

```python
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
    shared.state.gamma0[len(code.hx):],  # 原 Z 校验对应的行
    shared.state.attachment_x.T,        # n × s
    logical_paulis=["XI"], pauli="X",
    ensure_distance=False,
)
rebuilt.validate()
rebuilt.save("algebraic-deformed.json")
```

若用 `hgp_ancilla(HB, HD)` 产生 `(boundary, cycles)`，仍需调用方提供满足上述条件、且精确实现目标逻辑空间的 `gamma0/gamma1`。不满足时直接报错，不能只依据 `CB=0` 认定 surgery 正确。

## 命令行

```bash
# 小码基础构造：分别测 Z1、Z2，不运行距离修复。
python -m codesurgery.high_rate_surgery --example four \
    --logical-pauli ZI --logical-pauli IZ \
    --no-ensure-distance -o four-two-z.json

# 只测 Z1 Z2，而不是分别读出两者。
python -m codesurgery.high_rate_surgery --example four \
    --logical-pauli ZZ --no-ensure-distance -o four-zz.npz

# BB90：同时测 Z1、Z2，距离检查/增长共用 60 秒预算。
python -m codesurgery.high_rate_surgery --example bb90 \
    --logical-pauli ZIIIIIII --logical-pauli IZIIIIII \
    --ensure-distance --distance-method gurobi \
    --distance-time-limit 60 --seed 0 --trials 32 -o bb90-two-z.npz
```

本机验证记录（2026-10-14，`seed=0, trials=32`）：BB90 分别测 `Z1,Z2`，初始共享构造为 114 比特；自动增长后为 **154 比特、2 层**，最大校验重量和最大比特度数均为 **8**。Gurobi 得到受保护静态距离 **d=10**，保留 **6** 个逻辑比特，总用时约 **44.9 秒**。该记录验证一个有限实例，不是任意输入的常数开销保证，也未认证门调度或测量记录的故障距离。求解器限时和平台性能会影响搜索轨迹；随机种子和完整增长记录都在结果中保存。

若命令行和 JSON 都未指定 `ensure_distance`，交互终端会询问“是否需要保证码距”，默认回答是；非交互调用默认开启，并在 stderr 提示。使用 `--ensure-distance` / `--no-ensure-distance` 可以完全避免询问。

JSON 输入示例：

```json
{
  "example": "four",
  "logical_paulis": ["ZI", "IZ"],
  "optimize_support": true,
  "ensure_distance": false,
  "max_check_weight": 6,
  "max_qubit_degree": 4,
  "seed": 0
}
```

```bash
python -m codesurgery.high_rate_surgery input.json -o deformed.json
```

也可将 `example` 换为 `hx`、`hz`（或 `H_X`、`H_Z`）及可选 `distance`、`name`。若两类校验均为空，须提供正整数 `n` 来确定物理比特数。不要同时提供示例名与输入矩阵。

命令行显式给出的目标选择替换 JSON 中的目标选择。`targets`、`logical_paulis`、`logical_indices` 三种选择方式互斥；每种方式中的多项表示多个同时测量的观测量。

输出 `.json` 便于直接检查矩阵、见证和 metadata；`.npz` 适合 NumPy 读取。终端 stdout 输出 JSON 摘要，交互提示和诊断写入 stderr。
