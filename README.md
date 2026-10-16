# CodeSurgery

输入 CSS 量子纠错码的校验矩阵和逻辑 Pauli 算符，构造 code surgery 的变形码（deformed code）。支持用户指定物理代表、标准逻辑基、稳定子减重，以及减重前后实际辅助比特开销的比较。

本包输出静态校验矩阵与验证报告。它不生成完整测量电路、解码器或电路级故障距离证明。矩阵采用 dense NumPy，运算在 GF(2) 上进行。

## 安装

需要 Python 3.10 或更新版本。在下载的源码目录中运行：

```sh
python -m pip install .
```

开发时可使用 `python -m pip install -e .`。基础依赖为 NumPy；可选依赖按需安装：

```sh
python -m pip install '.[distance]'  # SciPy/HiGHS 距离与减重后端
python -m pip install '.[gurobi]'    # Gurobi 距离后端，需有效许可
python -m pip install '.[test]'      # pytest
```

以上命令安装当前目录的源码，不表示这个版本已经发布到 PyPI。发布流程见 [RELEASING.md](docs/RELEASING.md)。

## 快速开始：BB90 的两个逻辑测量

下面分别测量 IBM BB `[[90,8,10]]` 的 `Z₁`、`Z₂`，使用 devised sticking，同时构造原始代表与减重代表的两份变形码。

```python
from codesurgery import compare_deformations
from codesurgery.examples import ibm_bb90_code

code = ibm_bb90_code()
comparison = compare_deformations(
    code,
    scheme="devised_sticking",
    logical_paulis=["ZIIIIIII", "IZIIIIII"],
    optimization_options={"method": "heuristic", "seed": 0},
)

print("原始重量：", comparison.summary()["before_weights"])
print("减重后重量：", comparison.summary()["after_weights"])
print("开销比较：", comparison.overhead())

raw_code = comparison.original
reduced_code = comparison.optimized
comparison.save("bb90_Z1_Z2_comparison")
```

保存目录包含成功分支的 `original.npz`、`optimized.npz`，以及 `comparison.json`。报告记录算符符号、物理支撑、稳定子减重见证、构造状态和开销。为避免覆盖已有结果，保存时应选择新的输出目录。

列表的每一项表示一个独立测量：

| 输入 | 含义 |
| --- | --- |
| `["ZIIIIIII", "IZIIIIII"]` | 分别测量 `Z₁` 和 `Z₂`，得到两个结果 |
| `["ZZIIIIII"]` | 只测量乘积 `Z₁Z₂`，得到一个结果 |
| `["YIIIIIII"]` | 测量 `Y₁`；可使用 `scheme="gauging"` |

未提供逻辑基时使用 Zhang–Li simultaneous standard form。生成标准基本身不会优化代表重量。

## 使用自己的码与逻辑算符

```python
from codesurgery import CSSCode, compare_deformations

code = CSSCode(hx=HX, hz=HZ, distance=d)

comparison = compare_deformations(
    code,
    scheme="devised_sticking",
    logicals=[z1, z2],  # 每个向量长度为 n，表示用户指定的物理代表
    pauli="Z",
    optimization_options={"method": "heuristic", "seed": 0},
)
```

`HX`、`HZ` 是二进制校验矩阵，满足 `HX @ HZ.T = 0 (mod 2)`。未知原码距离时省略 `distance`；该参数是使用者提供的可信距离声明，不是包已计算得到的证明。

比较接口的三种目标输入互斥：

| 参数 | 输入格式 |
| --- | --- |
| `logical=...` | 一个物理 Pauli 字符串、二进制向量或 `{"x": ..., "z": ..., "sign": 0}` |
| `logicals=[...]` | 多个物理目标，每个长度为 `n`；二进制向量由 `pauli` 指定类型 |
| `logical_paulis=[...]` | 逻辑基上的 Pauli 字符串，每个长度为 `k` |

物理字符串支持前置 `+`、`-`；字典中 `sign=1` 表示负号。数组下标及输出 `support` 均从 0 开始。可用 `logical_basis={"jx": JX, "jz": JZ}` 指定一组配对的纯 CSS 逻辑基；未提供目标时默认选择首个标准 Z 逻辑生成元。

输入目标须为非平凡逻辑算符。程序检查对易、符号及实际读出的逻辑空间；不会因目标形式合法就接受额外逻辑读出。多个独立测量还须满足所选构造的独立性和兼容条件。

## 可选方案

`compare_deformations(..., scheme=...)` 支持下列六个模块。原生 API 可用于不需要双版本比较的调用。

| `scheme` | 原生 API | 当前输入范围与用途 |
| --- | --- | --- |
| `ckbb` | `codesurgery.ckbb_deform` | CKBB 分层构造，支持纯 Pauli、Y/XZ 及受限的显式分支连接 |
| `devised_sticking` | `codesurgery.devised_sticking.devised_deform` | 同型纯 X 或同型纯 Z 的独立目标集，允许物理支撑重叠 |
| `explicit` | `codesurgery.explicit_gadgets.gadget_deform` | Explicit 单目标 gadget；双因子 adapter 另用 `gadget_pair_deform` |
| `gauging` | `codesurgery.gauging_measurement.gauging_deform` | 辅助图测量，支持单个一般 Pauli；双因子另用 `gauging_pair_deform` |
| `high_rate` | `codesurgery.high_rate_surgery.high_rate_deform` | 共享辅助系统的多目标测量，混合目标须满足物理逐比特对易条件 |
| `branching` | `codesurgery.parallel_branching.branching_deform` | 将兼容目标路由到分支；本身保留逻辑信息，不完成测量 |

比较接口中的 `ckbb`、`explicit`、`gauging` 每次接受一个完整观测量；乘积应写为一个目标。`*_pair_deform(left, right)` 表示只测量 `left × right`，不表示分别测量两个因子；其支撑和 adapter 条件见详细文档。

Devised sticking 当前不接受 Y、XZ 或混合 X/Z 目标集合。多目标 branching/high-rate 要求逐物理比特上作用相同的非 I Pauli，或至少一方为 I；仅整体对易不足够。顶层 `deform` 当前仍只分派 CKBB。

## 减重与实际 overhead

比较接口严格保留原始分支的输入物理代表。优化分支仅在相同带符号稳定子陪集中搜索更轻代表，再重新构造并核验变形码。

- 默认仅在 `wt(L) > 2d` 时尝试减重；未知 `d` 时跳过，等于 `2d` 时也不触发。
- `optimization_options={"threshold": None}` 可要求始终尝试；整数阈值也可指定。
- 启发式搜索没有改进时保留原代表。找到更轻代表不等于证明重量最优。
- 原生 devised/explicit/gauging 的 `optimize_logical` 默认关闭；branching/high-rate 原生接口默认开启，各自规则见详细文档。比较接口统一控制两份代表，禁用构造器的再次减重。

```python
overhead = comparison.overhead()
print(overhead["original"]["ancilla_qubits"])
print(overhead["optimized"]["ancilla_qubits"])
print(overhead["savings"])
```

主要指标 `ancilla_qubits = n_deformed - n_memory`，从最终矩阵计算，包含已完成的重复层、expansion 和稀疏化。另记录总比特数、校验行数、最大行重量和列参与次数；Y 在一个物理位置只计一次。

`savings` 为原始开销减优化开销，负值表示开销增加。校验行数包含冗余行。统计不包含校验提取电路额外辅助比特、轮数或时空开销；branching 的开销仅对应分支码。**逻辑代表变轻不保证变形码更小，也不保证保距。**

某分支构造失败时，其开销为 `None`。默认 `strict=True` 抛出 `ComparisonConstructionError`，异常的 `.comparison` 保留成功结果与失败原因；`strict=False` 可直接取回部分结果：

```python
comparison = compare_deformations(
    code, logical_paulis="ZIIIIIII", scheme="gauging", strict=False,
    optimization_options={"method": "heuristic", "seed": 0},
    construction_options={"require_distance": False},
)
print(comparison.errors)
print(comparison.overhead())
```

这里显式关闭距离保护，适合查看基础构造开销；不能把该结果解释为保距构造。

## 距离与验证边界

所有成功测量构造核验完整校验对易、相位及实际逻辑读出。距离是另一个问题：

| 方案 | 默认距离处理 |
| --- | --- |
| CKBB | 默认层数取声明的 `d`，未知时取 `n`；还须满足不可约支撑等证书条件。构造成功可能仍未认证保距 |
| Devised sticking | 默认 `d_R=d`，未知时取 `n`；当前连接映射范数为 1，已知 `d` 时下界为 `min(d,d_R)` |
| Explicit / gauging | 默认 `require_distance=True`；尝试认证并按需修复，最终无法认证或超限时报错 |
| High-rate | 默认 `ensure_distance=True`，但 `strict=False` 可返回未认证结果；原生 `strict=True` 才要求成功认证 |
| Branching | 分支构造有静态保距证书；后续测量需另行认证 |

这些构造证书可能依赖使用者声明的原码距离。比较“同等保护下的 overhead”时，应分别检查两份结果的距离证据；`.overhead()` 本身不实施保距筛选。`compare_deformations(strict=True)` 也不等同于 high-rate 原生距离选项；后者通过 `construction_options={"strict": True}` 传入。

可直接对实际矩阵计算静态距离或检查阈值：

```python
from codesurgery.distance_preservation import code_distance, verify_distance

report = code_distance(code, method="auto", time_limit=60.0, seed=0)
proof = verify_distance(code, 10, method="auto", time_limit=60.0, seed=0)
print(report["distance_statement"])
print(proof["target_met"])  # True / False / None：已证明 / 已找到反例 / 未确定
```

随机搜索只提供经过验证的上界；超时不等于保距。一般稳定子距离接口会把辅助 gauge 自由度也视为逻辑比特，不能无条件作为 protected/dressed subsystem 距离。求解器时限不包含全部矩阵前处理与重建时间。详见 [距离与稀疏化说明](DISTANCE_PRESERVATION_ZH.md)。

## BB90 示例与命令行

源码仓库的 `examples/` 仅保留 BB90 相关示例，运行前先安装包：

```sh
python examples/bb90_compare.py --scheme devised_sticking --logical ZIIIIIII IZIIIIII
python examples/bb90_compare.py --scheme gauging --logical YIIIIIII --always
python examples/bb90_compare.py --scheme gauging --logical ZIIIIIII --distance
```

`bb90_compare.py` 默认以启发式减重比较两份构造；gauging/explicit/high-rate 的距离后处理由 `--distance` 开启。CKBB/devised 的重复层使用各自原生默认值。支持多个目标的方案才可传多个 `--logical` 值。

其他保留示例：[gauging](examples/bb90_gauging.py)、[explicit](examples/bb90_explicit.py)、[减重后 explicit](examples/bb90_optimized_explicit.py)、[ZX adapter](examples/bb90_zx_adapter.py)、[ZX 审计](examples/bb90_zx_audit.py)。Explicit 示例使用同目录的 `bb90-explicit-data/` 输入数据；运行生成的结果不属于发行包。

六个命令行入口均可用 `--help` 查看 JSON 输入和构造选项：

```sh
python -m codesurgery --help
python -m codesurgery.devised_sticking --help
python -m codesurgery.explicit_gadgets --help
python -m codesurgery.gauging_measurement --help
python -m codesurgery.high_rate_surgery --help
python -m codesurgery.parallel_branching --help
```

## 文档与开发

- [CKBB 矩阵与伪代码](ALGORITHM.md)
- [Devised sticking](DEVISED_STICKING.md)
- [Explicit、gauging 与 Universal adapters](EXPLICIT_GADGETS_ZH.md)
- [Parallel branching](docs/PARALLEL_BRANCHING.md)
- [High-rate surgery](docs/HIGH_RATE_SURGERY.md)
- [通过 Pauli 测量实现逻辑 H、S、CNOT](docs/PAULI_GATES.md)
- [打包与发布](docs/RELEASING.md)

安装 `test` 可选依赖后，可自行运行 `python -m pytest -q`。本轮发行整理未运行算法测试，不将整理完成视为测试通过。

代码按 [MIT License](LICENSE) 发布。可选 Gurobi 后端受其自身许可约束。
