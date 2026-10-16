# Explicit gadget、Universal adapter 与 gauging measurement

本版完善 `codesurgery.explicit_gadgets`，并增加 `universal_adapters` 和
`gauging_measurement`。CKBB 与 devised sticking 使用原有实现。

论文依据：

- [Webster–Smith–Cohen，2025，Sec. II.1](https://arxiv.org/html/2511.15989v1#S2.SS1)：单算符基础 gadget。
- [Swaroop–Jochym-O’Connor–Yoder，2024 初稿](https://arxiv.org/html/2410.03628v1)：Universal adapters，SkipTree 与 repetition adapter。
  [v4 的 Sec. III、IV](https://arxiv.org/html/2410.03628#S4) 给出同一核心构造及更一般的条件；本文的 Lemma 9/10 编号采用 v4。
- [Williamson–Yoder，2024 初稿](https://arxiv.org/html/2410.02213v1)：gauging logical measurement。

**Python API 默认自动计算码距并按需处理**，可用 `require_distance=False` 关闭。基础构造本身不自动补扩展边；自动处理只在实际距离不足时增强扩展，且按初始水平 memory 拼接块 `[H | M]` 的最大行重量检查稀疏度。历史上的 `expansion='auto'/'star'/'complete'` 仍是显式选择的附加启发式，
不是 WSC 论文规定的补边算法；它们也不能绕过原始 F 的逻辑读出检查。双因子联合测量只使用下述 adapter。

## 运行

在 PyCharm 中可运行 [BB90 示例](examples/bb90_compare.py)，比较输入代表减重前后的构造与 overhead。也可以通过包的命令行处理自己的 `input.json`：

```bash
python -m codesurgery.explicit_gadgets input.json -o deformed.json
python -m codesurgery.gauging_measurement input.json -o gauging.json
```

输入 JSON 需包含原码的 `hx,hz` 和待测算符。
`scheme` 可为 `explicit` 或 `gauging`；`logical` 表示一个待测可观测量，`logicals:[L1,L2]` 表示只测量其乘积。
这里旧称“双目标”的接口实际是**双因子联合测量**：输入 L1、L2，读出一个 ±1 结果，
测量投影为 `(I ± L1 L2)/2`。例如输入逻辑 X1 与 Z2，测量的是逻辑 X1Z2，
不会分别获得 X1、Z2 的两个结果。单目标自身也可以是逻辑乘积，例如直接输入 X1Z2 的完整物理代表。

## 输入逻辑基、Y 和混合 XZ

```python
from codesurgery import CSSCode
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform

code = CSSCode(HX, HZ, distance=d)
result = gadget_deform(code, logical="YXXZZIIIIIIIIII")
pair = gadget_pair_deform(code, left=L1, right=L2)
```

不提供逻辑基时使用 Zhang–Li standard form。可传入
`logical_basis={"jx": JX, "jz": JZ}`，其中两者都是 k×n 的纯 CSS 配对逻辑基；
程序验证 normalizer 条件和 `JX @ JZ.T = I`。物理列始终保持输入 code 的原始顺序。

不提供目标时，单目标默认 Z0；可指定 `pauli='X'/'Y'/'Z'` 和 `logical_index`。
联合测量的两个因子默认 Z0、Z1，使用 `paulis` 和 `logical_indices` 修改。默认代表可能有物理重叠；此时 adapter 接口报错。

目标可以是带正负号的 IXYZ 字符串、物理 `{x,z,sign}`、二进制向量加 `pauli`，或者逻辑基系数：

```python
# 测量配对逻辑基中的 X0 Z1，系数长度为 k。
logical = {"logical_x": [1,0,...], "logical_z": [0,1,...]}
```

逻辑系数采用 `i^(alpha·beta) Xbar(alpha) Zbar(beta)` 的 Hermitian 约定。
特别是 `pauli='Y'` 表示 `i Xbar_j Zbar_j`，物理 Pauli 字符串前可能带负号。
例如物理代表 XXX 与 ZZZ 对应 `i XXX ZZZ = -YYY`。符号单独保存在 `check_signs` 和 `measurement_sign` 中。

一个 Y 或混合 Pauli 目标直接按 `(x|z)` 构造。双因子 adapter 当前实现要求物理支撑不重叠，
因此可处理不重叠的 XZ、YZ 等乘积。重叠的因子会提示使用完整 Hermitian 目标调用 `gauging_deform`；
例如逻辑 Y 应使用 `pauli='Y'`，而不是把反对易的 X、Z 因子直接当作一个可测乘积。

## 单目标 F 的检查

将 memory 全体校验写成 `C=(CX|CZ)`。目标为 `(x|z)`，支撑 V 的大小为 w。
S 为 w×n 的支撑选择矩阵，定义

```text
F_full[j,u] = CX[j,V[u]] z[V[u]] + CZ[j,V[u]] x[V[u]]  (mod 2)
F = F_full 的所有非零行
```

这里不对 F 作行基约简；重复行对应不同辅助比特。纯 X 时就是 HZ 限制到目标支撑，纯 Z 时取 HX。

单目标 explicit 路径按以下规则运行：

1. 目标与 memory 校验对易，因此 F 的每个非零行具有偶数重量；**不要求重量恰好为 2**。
2. 若输入了可信的原码距离下界 d 且 w<2d，免去本步骤的 kernel 逻辑信息预检。
3. 否则计算 K=ker(F)、QX=(KS)diag(x)、QZ=(KS)diag(z)。其逻辑签名空间必须等于目标的一维 span。
   发现额外独立逻辑信息时抛出 `GaugeMeasurementRequired`。
4. 使用完整的辅助核 `N=ker(Fᵀ)`，秩为 `行数(F)-rank(F)`。
   不要求普通图连通；额外 kernel 方向若只对应原码 stabilizer，不应误报为额外逻辑测量。

只有双因子 adapter 路径额外要求：每个 F_i 的非零行重量恰好为 2，并且普通图连通。
这些条件使 SkipTree 可以运行；失败会报错并提示使用 gauging measurement。

第 2 步只针对与目标局部 Pauli 对齐的子算符。若存在独立逻辑 Q，它的补算符 LQ 也是非平凡逻辑，
二者支撑不交且重量之和为 w，于是 w≥2d，矛盾。这不排除支撑内的 stabilizer；单目标接受这些方向，adapter 路径仍需检查普通图连通性。
`w=2d` 不满足快捷条件；没有 d 也会完整做预检。

**w<2d 不会免去 adapter 路径的图条件、对易、相位以及最终实际读出检查，也不是距离保持证书。**

## 基础矩阵与一般 Pauli

纯 X 情况，T 将 F 的每一行映射到对应的 memory Z 校验；N 是 ker(Fᵀ) 的行基：

```text
HX_new = [ HX   0  ]       HZ_new = [ HZ  T ]
         [ S   Fᵀ ]                [ 0   N ]
```

通用辛形式采用辅助关联矩阵 B（辅助比特×顶点）、port selector S（顶点×memory）及 dressing M。
单目标中 B 可以具有重量大于 2 的偶数行；普通图只是特例：

```text
M B = (CX diag(z) + CZ diag(x)) Sᵀ
N B = 0,    rank(N) = 行数(B) - rank(B)

check_x = [ CX        0  ]       check_z = [ CZ        M ]
          [ S diag(x) Bᵀ ]                 [ S diag(z) 0 ]
          [ 0         0  ]                 [ 0         N ]
```

每个 memory 支撑位置恰好接到一个 port；允许图中有不接 memory 的 dummy 顶点。
所有顶点校验相乘得到目标。纯 Z 时实现对所有辅助比特作 Hadamard 对偶，使输出仍然是 CSS。
Y 和混合 XZ 使用完整 `check_x/check_z/check_signs`，两半同一行表示同一个 Pauli 校验。

## 双因子联合测量：SkipTree repetition adapter

分别检查 L_l、L_r 的逻辑读出，再要求 F_l、F_r 为连通的重量 2 普通图。
得到各自 dressing M_l、M_r。memory 校验只有一份；
同一原始校验可能同时接到左右两个辅助图。

选择左右各 A 个顶点，其诱导子图必须连通。默认 A=min(|V_l|,|V_r|)，用 BFS 前缀选择连通顶点集。
可显式指定 `adapter_size`、`adapter_vertices=(left_vertices,right_vertices)`。

SkipTree 从生成树出发，交替使用先序/后序标签；相邻循环标签之间取树路径。
得到 T_l、T_r、P_l、P_r 和循环 repetition check matrix H_C，使得

```text
T_l F_l = H_C P_lᵀ,      T_r F_r = H_C P_rᵀ.
```

T 的每行重量至多 3、每列重量至多 2。P_i 是 |V_i|×A 的选点排列矩阵。
按左右相同标签连接 A 条 adapter 边，每条边对应一个真实辅助比特。列顺序为
`(memory, left_edges, adapter_edges, right_edges)`。

```text
B = [ F_l   0   ]        N = [ N_l   0    0   ]
    [ P_lᵀ P_rᵀ ]            [ T_l  H_C  T_r ]
    [ 0     F_r ]            [ 0     0    N_r ]

M_joint = [ M_l   0_A   M_r ]
```

新增 cycle 校验 `[T_l H_C T_r]` 的重量至多 8，每条旧边被这些新校验使用至多两次。
保留这些稀疏生成元，不用任意 RREF 核基替换；A 个循环生成元通常只有 A−1 个独立。
A=1 时省略唯一的零 cycle 行。

最后用 B、M_joint、N 和联合 port 构造 deformed code。程序检查整个实际校验群，
要求只能读出 L_l L_r，不能分别读出 L_l 或 L_r。

## 基础构造伪代码

下面先展开基础矩阵构造。公开 Python API 默认再调用本文末尾的自动码距与稀疏化流程；`require_distance=False` 才直接返回这里的基础结果。

```text
EXPLICIT(code, targets, optional_basis):
    basis = optional_basis 或 Zhang–Li standard form
    解析带符号的目标，检查非平凡 normalizer 条件

    对每个目标 L_i:
        提取 F_i：局部反对易矩阵的全部非零行
        如果 d 未知，或者 wt(L_i) >= 2d:
            K_i = ker(F_i)
            Q_i = 与目标局部 Pauli 对齐的 K_i S_i
            Gamma_i = logical_signature(Q_i, basis)
            如果 rowspace(Gamma_i) != span(signature(L_i)):
                ERROR("测量了额外逻辑信息；请使用 gauging measurement")

    如果只有一个目标:
        B = F_1
        M = 原始校验到 F_1 行的选择矩阵
        N = ker(Bᵀ) 的完整行基，维数 = 行数(B) - rank(B)
        target = L_1
    否则:  # 输入两个因子，仅测量它们的乘积
        对 F_1、F_2:
            如果任一非零行重量不是 2:
                ERROR("adapter 要求重量 2；请使用 gauging measurement")
            如果普通图不连通:
                ERROR("SkipTree 需要连通图；请使用 gauging measurement")
        如果两个目标有物理支撑交叠:
            ERROR("用完整 Hermitian 目标进入 gauging measurement")
        为 F_1、F_2 选择等大小的连通诱导子图
        运行 SkipTree，得到 T_1,P_1,T_2,P_2,H_C
        按上面的块矩阵构造 B、M_joint、N
        target = 带正确符号的 L_1 L_2

    构造 memory、vertex、cycle 三组辛校验
    解符号：memory/cycle 正号，无 -I 依赖，顶点乘积等于 signed target
    VALIDATE_ACTUAL_READOUT(全部校验, target, basis)
    检查剩余逻辑比特数 k-1
    普通图报告相对扩展；一般单目标 Tanner 矩阵报告奇偶边界 h(F)
    基础步骤不自动补扩展边
    封存构造与证书，返回基础 deformed code

VALIDATE_ACTUAL_READOUT(CX_new, CZ_new, target, basis):
    A = [CX_new[:,aux] | CZ_new[:,aux]]
    U = ker(Aᵀ)
    RX = U CX_new[:,memory]
    RZ = U CZ_new[:,memory]
    Gamma = row_basis([RX JZᵀ | RZ JXᵀ])
    如果 rowspace(Gamma) != span(signature(target)):
        ERROR("实际构造读出额外逻辑；请使用 gauging measurement")
```

以上实际读出检查在构造、验证及保存时执行。修改结果对象的矩阵或证书后，summary/validate/save 会拒绝旧快照。

## Gauging measurement 框架及 adapter

```python
# 单目标读出额外逻辑信息，或联合测量的 F 不符合 adapter 图条件时，调用这条路线。
r = gauging_deform(code, logical)

# 指定自己的图、port、dressing 和稀疏 cycle 基。
r = gauging_deform(code, logical, graph=B, ports=ports,
                   matching=M, cycle_basis=N)

# 左右各自构造辅助图，再按同一个 SkipTree adapter 连接。
r = gauging_pair_deform(code, L1, L2, graphs=(B1, B2))
```

未给图时，对每个原始校验的偶数个反对易位置排序后两两配对，每对增加一个图边。
若形成多个连通分量，再将分量代表串成链。每个原始校验在其配对边上获得辅助 Z 作用。
这是论文 auxiliary-graph construction 的一个确定性配对选择；它只保证连通与正确读出，不声明扩展性或最优开销。

```text
GAUGING(code, target, optional_graph):
    解析 target 和 basis
    如果未给图:
        将各校验的偶数反对易边界分成二元配对边
        连接图的不同分量
        保存 dressing M 的每一行所使用的边
    否则:
        检查图的每行重量为 2，且连通
        检查 port 为目标支撑到图顶点的单射
        求解或核验 M B = 原始校验的 port 反对易边界
    选择或核验完整 cycle 生成元 N
    用统一辛矩阵构造并执行全部实际读出与符号检查
    返回基础 flux-fixed deformed code 和协议框架

GAUGING_PAIR(code, L1, L2):
    分别构造两个 connected auxiliary graphs 和各自 dressing
    检查物理支撑不交叠
    使用与 EXPLICIT_PAIR 相同的 SkipTree adapter
    对最终乘积构造执行完整实际读出检查
```

这里实现的是 gauging 的静态、flux 已固定的 deformed code，而非任意非对易 gauge-generator 测量器。
协议 metadata 包括辅助初始化、顶点测量的乘积读出以及辅助测出方式；尚未编译带噪声的 syndrome 电路。

## 距离和 LDPC 的界限

普通连通图路径检查相对扩展

```text
beta_d(B, ports) = min wt(Bv) / min(d, |v∩ports|, |ports|-|v∩ports|)
```

只考虑分母非零的 v。小图精确枚举；图过大时保留 unknown，或使用已验证的组合证书：
左右 beta_d≥1、adapter 所选点都在 ports 中且 A≥d，则由 Universal adapters Lemma 9 得到合并图 beta_d≥1。
`wt(L)<2d` 与这个距离检查不是同一个条件。

一般单目标 Tanner 矩阵另外检查 `h(F)=min wt(Fv)/wt(v)`，其中 `0<wt(v)≤w/2`，
采用奇偶边界而非普通超图边界。使用 `h(F)≥1` 作为静态距离保持的充分条件；不套用普通图连通性推导的界。

上述是图的充分条件；不满足它们不代表实际距离下降。默认自动模式另外计算原码及完整输出码的真实静态距离，未提供原码 distance 时可从精确计算得到目标。若提供 distance，则仍将它视为可信下界，并拒绝与实际逻辑见证冲突的声明。
只有实际逻辑见证证明距离不足才增强扩展；未知且没有有效距离定理证书时提示增加预算。
完整校验默认必须满足初始水平 memory 拼接块 `[H | M]` 的最大行重量阈值，cycle 过长或拥塞过大时按需稀疏化。
详见 [自动码距与稀疏化模块](DISTANCE_PRESERVATION_ZH.md)。还会检查完整列参与次数、matching 和 cycle 的拥塞；
一个有限例子通过这些检查不自动证明整个码族 LDPC。SkipTree 保证的是连接新增部分的稀疏性。

## 已运行的例子

- Hamming 单 Y：`YXXZZIIIIIIIIII`，原始 F 为路径；19 比特，13 个独立校验，保留 6 个逻辑比特。h=1/2。
- 同一 Hamming code 的 X1X2X3 与 Z4Z8Z12：分别提取两个 F，各 2 条边；adapter 增加 3 条边。
  最终 22 比特，17 个校验生成元、秩 16，保留 6 个逻辑比特，只读出 XZ 乘积。
  对全部 2145 个重量 1、2 的非恒等 Pauli 错误检查，没有遗漏的非平凡逻辑；静态距离下界为 3。
- Gauging demo：原始 F 导致额外独立逻辑读出，single explicit 报错；独立 gauging 路径构造 Y 测量，并通过 adapter 测量带符号的 YZ 乘积。

## 单目标重量 4 行的合法例子

```text
HX = [[1,1,1,1,0], [1,1,0,0,0], [1,0,1,0,0]]
HZ = 空矩阵 (0×5)
L  = ZZZZI
F  = [[1,1,1,1], [1,1,0,0], [1,0,1,0]]
```

`ker(F)=span(1111)`，因此单目标能构造，保留重量 4 的原始行。
基础结果从 5 比特、2 个逻辑比特变为 8 比特、1 个逻辑比特；h(F)=1/2 本身不足以给出图距离证书。自动模式再计算实际距离，不能仅凭这个 h 值判断距离下降。
将同一个 L 与 Z5 分别作为两个 adapter 因子时，会因左 F 含重量 4 行而报错。
这区分的是构造路径，不是“一个逻辑 Pauli”与“逻辑 Pauli 乘积”的区别。

## 默认自动计算码距与稀疏化

Python 的四个构造接口现在默认 `require_distance=True`，显式 `False` 保留基础构造。
CLI 未指定时仍在交互终端询问；使用 `--require-distance` / `--no-require-distance` 可显式选择。
非交互模式未指定时保留基础构造，并打印说明。开启后没有原码 distance 时尝试计算，不再要求先手动输入。

```python
from codesurgery.distance_preservation import code_distance, stabilizer_distance

result = gadget_deform(code, logical)  # 默认自动处理
pair = gadget_pair_deform(code, L1, L2)
basic = gauging_deform(code, logical, require_distance=False)
result = gauging_deform(code, logical,
    distance_options={"distance_max_candidates": 2_000_000})
report = code_distance(result)
# 或直接提供完整辛校验：stabilizer_distance(CX, CZ)
```

计算的是 centralizer 中、stabilizer 行空间外的最小 Pauli 支撑重量。报告区分精确距离、已证上下界和预算内未知，不用输入声明替代计算。若枚举不完整但已有有效图定理证明下界，则可通过距离检查，精确距离字段仍保持 `None`。

默认行阈值：纯 Z 目标为 `max rowwt([HX | M_X])`，纯 X 用对偶块 `[HZ | M_Z]`，Y/XZ 用全部 memory 行的 `tau=max_i(wt(CX_i OR CZ_i)+wt(M_i))`。各项必须按同一行相加；纯目标的另一类未 dressing 校验不会暗中抬高阈值，但仍须通过最终完整矩阵检查。cycle 默认也用这个阈值，完整列参与次数另设上限。后处理不自动增大 tau；可用显式 `max_check_weight` 覆盖。

实际距离和全部稀疏度界已达标就保留原矩阵。距离不足才增强扩展；超重 cycle 或拥塞才触发 decongestion/cellulation。最后重建并核验实际读出、相位、完整行列重量及实际距离。不能在指定上限内完成则报错。
一般偶数 Tanner 行在确实需要图后处理时转换为普通辅助图，并保留原始构造记录。
完整流程、参数、实例和伪代码见 [DISTANCE_PRESERVATION_ZH.md](DISTANCE_PRESERVATION_ZH.md)。
