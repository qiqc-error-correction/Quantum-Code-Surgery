# 用 Pauli 测量实现逻辑 H、S、CNOT

`codesurgery.pauli_gates` 是独立的门分解与测量构造层。它把逻辑门转为有先后顺序的完整 Pauli 测量，为每个测量调用已有 surgery 后端，输出对应的 deformed code，并根据实际测量结果计算 Pauli correction/frame。默认后端是 `gauging`。

默认保留**原码的第一个逻辑比特**作为辅助比特，Python 下标为 `ancilla=0`。它不是第一个物理比特，也不是新增加的编码块。其他逻辑标签不重新编号：原来的逻辑下标 1 仍然写 1。辅助比特会被准备、测量和重置，其原有未知信息会丢失；不得把它同时作为数据门的 target/control。

## 门的测量顺序

下面的下标 `a` 表示辅助逻辑比特，`d` 表示数据，`c,t` 表示 CNOT 的 control、target。一个 Pauli 乘积表示一次联合测量，不能拆成分别测量各因子。

测量结果位满足 `b=0` 表示本征值 `+1`，`b=1` 表示本征值 `−1`。表中的 `b_i` 是已经扣除当前 Pauli frame 影响的**有效结果**；有未执行的 correction 时不能直接代入原始硬件结果。

| 门 | 准备辅助比特 | 按顺序测量 | 数据 correction，忽略整体相位 |
|---|---|---|---|
| `H(d)` | `Z_a=+1` | `X_d X_a → b1`；`Z_d X_a → b2`；`Z_a → b3` | `X_d^(b1 XOR b2) Z_d^(1 XOR b1 XOR b2 XOR b3)` |
| `S(d)` | `X_a=+1` | `Z_d Y_a → b1`；`Z_a → b2` | `Z_d^(b1 XOR b2)` |
| `CNOT(c,t)` | `X_a=+1` | `Z_c Z_a → b1`；`X_a X_t → b2`；`Z_a → b3` | `Z_c^b2 X_t^(b1 XOR b3)` |

这里 `S=diag(1,i)`。H 使用两个不同的联合 Pauli 测量，使数据始终留在原逻辑标签上；它的 correction 中含常数 `1`，所以不能把所有零结果误认为“没有 correction”。每次门结束后还要处理辅助比特的 `Z` 读出，使它重置到 `Z_a=+1`，或把相应 `X_a` correction 记录在 frame 中。

准备步骤也是真实测量：测得 `Z_a=(-1)^p` 后用 `X_a^p` 准备 `|0>`；测得 `X_a=(-1)^p` 后用 `Z_a^p` 准备 `|+>`。因此 H/CNOT 各有 **4 次**测量，S 有 **3 次**，均已包含准备测量。没有给出 outcomes 时，编译器只输出计划，不虚构运行结果。

Guo Zhang–Ying Li 的 [Time-efficient logical operations on quantum LDPC codes，附录 I.3](https://arxiv.org/html/2408.01339#A9.SS3) 给出通过逻辑测量实现 Clifford 门的框架；Krishna–Poulin 的 [Fault-tolerant gates on hypergraph product codes，Lemma 21](https://arxiv.org/html/1909.07424#S5.SS2) 也讨论用 Pauli 测量实现 Clifford 门。本文模块使用保持原数据标签的 H 投影恒等式，并直接利用可测 Y 的后端实现 S；这两点并不声称是 Zhang–Li 论文原图的逐步复现，也不声称测量次数最优。

CNOT 使用标准的辅助比特联合奇偶测量序列，参见 Litinski–von Oppen 的 [Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes，图 3](https://arxiv.org/html/1709.02318#S3)。实现中的每个分支都通过完整投影算符核验到目标酉门，允许整体相位；核验包括辅助准备、重置、初始 Pauli frame 和数据与外部系统纠缠的情形。

## Python API

```python
from codesurgery.examples import ibm_bb90_code
from codesurgery.pauli_gates import compile_gates

plan = compile_gates(
    ibm_bb90_code(),
    [("H", 1), ("S", 1), ("CNOT", 1, 2)],
    ancilla=0,
    backend="gauging",
    # 留空时保留 gauging 原有的距离处理默认值。
)

print(plan.summary())
plan.save("pauli-gates-output", matrix_format="npz")
```

如果只查看分解，而不启动矩阵构造和距离计算：

```python
plan = compile_gates(code, [("H", 1), ("S", 1)], build=False)
for step in plan.measurements:
    print(step.id, step.role, step.logical_pauli)
```

单门也可以调用：

```python
from codesurgery.pauli_gates import compile_gate

h_plan = compile_gate(code, "H", target=1)
cnot_plan = compile_gate(code, "CNOT", control=1, target=2)
```

没有给 `logical_basis` 时使用项目中已有的 Zhang–Li standard form。用户可以传入配对的 CSS 逻辑基；整个程序保持这套基和逻辑标签一致。逻辑 Y 按 `i Xbar Zbar` 构造，包含物理 Pauli 代表所需的符号，而不是仅取 X/Z 支撑并集。

快速检查小码矩阵时可以显式关闭距离处理：

```python
from codesurgery.examples import direct_sum, four_qubit_code

code = direct_sum(four_qubit_code(), four_qubit_code())  # [[8,4,2]]
plan = compile_gates(
    code,
    [("H", 1), ("S", 1), ("CNOT", 1, 2)],
    backend_options={"require_distance": False},
)
```

这个快速设置不会自动得到码距保证；查看具体后端结果中的证书和未认证状态。编译器不会为了使所有方案都“能运行”而自动改变测量目标、拆分联合测量、换用另一后端或关闭距离选项。

## 实际结果与 Pauli frame

每个测量都有稳定的 ID，例如第一扇门的 `g0_prep`、`g0_m1` 等。准备结果必须提供。硬件得到的测量结果应当先使用各 gadget 自带的 readout witness 和符号修正转换成**请求逻辑 Pauli 的结果**，再传给 `resolve`。

```python
# measured_bits 是来自真实运行或明确模拟的 ID -> 0/1 字典。
resolution = plan.resolve(measured_bits, outcome_format="bits")
print(resolution["correction_pauli"])
print(resolution["data_correction_pauli"])
print(resolution["ancilla_correction_pauli"])
```

也接受按 `plan.measurements` 顺序排列的完整结果列表。若输入本征值，明确使用 `outcome_format="eigenvalues"`，值只能为 `+1/-1`。返回值给出原始结果、有效结果、结果翻转和最终 frame。`correction_pauli` 是原码全部 `k` 个逻辑位置的 Pauli 字符串。

`resolve` 默认采用 **frame 模式**：准备、门后 correction 和辅助重置的 Pauli 都记录下来，不假定硬件已经施加；下一次测量的有效结果会自动修正。如果硬件在中途实际执行了这些 correction，应按自己的执行边界消费或清除 frame，不要把同一 correction 再算一次。

例如当前 frame 为 `P`，请求测量为 `M`，原始结果位是 `r`，则

```text
effective_bit = r XOR symplectic_product(P, M)
```

随后根据有效结果逐项更新条件 Pauli frame。这种逐测量的处理已经包含整扇门的 frame 传播，不需要在门边界再手动共轭一次。使用 Y 测量时同样按辛内积计算，不使用仅适用于纯 X/Z 的简化翻转规则。

## 后端

| `backend` | 构造接口 | 注意事项 |
|---|---|---|
| `gauging`（默认） | gauging measurement | 支持完整的混合 Pauli；保留默认距离处理 |
| `ckbb` | CKBB | 保留现有支撑与层数检查 |
| `explicit` | explicit gadgets | 可能因受限支撑、F 或额外逻辑信息检查失败；不会自动拆成因子 |
| `devised_sticking` | devised sticking | 当前只支持纯 X 或纯 Z；H/S 所需混合目标会明确报错 |
| `high_rate` | high-rate surgery | 保留其随机增长、距离预算和 strict 语义 |

`backend_options` 原样传入每个**不同的测量 gadget**，但禁止覆盖目标、逻辑基或原码。同一 observable 的矩阵可以复用，测量步骤和结果仍各自独立。后端失败会报告方案和出错目标，不会返回一个伪装为成功的完整门。

各后端的距离预算作用于每次不同 gadget 的构造；多门程序的总运行时间可能超过单个 gadget 的一分钟预算。关闭距离处理必须显式指定，不会因编译多个门而自动关闭。

## 编译伪代码

```text
COMPILE_GATES(code, gates, ancilla=0, basis=None, backend="gauging"):
    basis = basis 或 standard_form(code)
    检查 ancilla 是合法逻辑标签
    检查 H/S 数据标签、CNOT control/target 合法且不等于 ancilla

    对每扇门依次：
        生成准备、联合测量、辅助读出的有序步骤
        为每个步骤给出独立 ID 和 Pauli-frame 更新规则

        对该步骤的完整逻辑 Pauli L：
            若构造矩阵且尚未缓存 L：
                physical_L = basis 中 L 的带符号物理代表
                gadget = 选定后端(code, physical_L, backend_options)
                检查完整校验对易、相位和实际读出逻辑空间
                要求恰好读出 span(L)，不允许多测其他逻辑
                保存完整 deformed code、读出见证和距离状态

    返回有序计划、不同测量的矩阵、frame 规则和后端状态

RESOLVE(plan, raw_outcomes, initial_frame=I):
    按步骤处理真实测量结果
    结合当前 frame 计算每个有效结果
    用相应门的分支公式更新 frame
    保留所有其他数据逻辑的信息和原逻辑标签
    返回最终 correction 和逐步 trace
```

每个步骤都指一次完整的**合并—测量—分离，回到原 memory code**的非破坏性 Pauli 测量。下一步重新使用原码及同一逻辑基。不能把上一张 deformed code 当作下一次调用的输入，也不能把所有步骤的校验合并成一张同时测量的矩阵：H 等协议刻意包含不对易的先后测量。

## 命令行与输出

在项目根目录运行：

```bash
PYTHONPATH=src python -m codesurgery.pauli_gates \
  --example eight --gate H:1 --gate S:1 --gate CNOT:1:2 \
  --backend gauging --no-ensure-distance -o pauli-gates-output
```

仅分解 BB90：

```bash
PYTHONPATH=src python -m codesurgery.pauli_gates \
  --example bb90 --gate H:1 --gate S:1 --gate CNOT:1:2 \
  --decompose-only -o bb90-gate-plan
```

输入 JSON 可以包含 `example` 或 `hx/hz`、`distance`、`logical_basis`、`gates`、`ancilla`、`backend`、`backend_options`、`build`。例如：

```json
{
  "example": "eight",
  "gates": [["H", 1], ["S", 1], ["CNOT", 1, 2]],
  "ancilla": 0,
  "backend": "gauging",
  "backend_options": {"require_distance": false}
}
```

使用 `--backend-options '{"require_distance": false}'` 可覆盖后端参数，或使用 `@options.json`。`--outcomes @outcomes.json` 可以同时计算 correction；`--outcome-format eigenvalues` 表示输入本征值。命令行显式 gate 列表替换 JSON 中的列表。`--overwrite` 才允许覆盖旧导出。

输出目录包含 `plan.json`、每个测量步骤对应的 `.npz` 或 `.json` 矩阵；相同 observable 在内存中复用构造，导出时仍按步骤命名。给出 outcomes 时还包含 `corrections.json`。`plan.json` 内有步骤到矩阵文件的映射。门分解使用本文的 Python API 或命令行；单次 surgery 的完整运行示例见 [BB90](../examples/bb90_compare.py)。

## 认证范围

门恒等式是理想逻辑测量仪器下的逐结果分支恒等式。每张 gadget 还检查校验对易、逻辑读出和符号；静态码距证书继承所选后端。

本模块输出逻辑测量计划、各步骤的静态矩阵和 frame 规则；物理执行层仍需提供非破坏测量、恢复原码的分离过程和相应解码。本模块没有生成完整物理 syndrome-extraction circuit，也没有证明包括准备、分离、测量记录错误和 feedforward 在内的整体时空故障距离。输出中的 `measurement_circuit_distance_certified` 保持为 `False`。不能把某一张 deformed code 的静态 `d=10` 当作整套 H/S/CNOT 的电路级距离已经达到 10。
