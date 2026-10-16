"""Compare BB90 surgery overhead before and after logical weight reduction.

Run as a normal Python program in the environment where codesurgery is installed:

    python bb90_compare.py
    python bb90_compare.py --logical ZIIIIIII IZIIIIII --always
    python bb90_compare.py --scheme gauging --logical XZIIIIII

PyCharm: use a Python run configuration, not pytest. This file imports the
installed package directly; it does not alter the Python module search path.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

import codesurgery
from codesurgery import ComparisonConstructionError, compare_deformations
from codesurgery.examples import ibm_bb90_code


SCHEMES = ("devised_sticking", "ckbb", "explicit", "gauging", "high_rate", "branching")
DISTANCE_SCHEMES = ("explicit", "gauging", "high_rate")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, default="devised_sticking")
    parser.add_argument(
        "--logical", nargs="+", action="extend", default=None, metavar="PAULI",
        help="每个目标用 8 个 I/X/Y/Z 字符表示；默认 ZIIIIIII。负号可写 --logical=-ZIIIIIII",
    )
    parser.add_argument("--always", action="store_true", help="不受 wt > 2d 阈值限制，始终尝试减重")
    parser.add_argument(
        "--distance", action="store_true",
        help="开启 explicit/gauging/high_rate 的原生码距后处理；两份构造分别处理",
    )
    parser.add_argument("--seed", type=int, default=0, help="减重启发式随机种子，默认 0")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results") / "bb90-comparison",
        help="结果根目录，每次运行创建带时间戳的子目录",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")
    if args.distance and args.scheme not in DISTANCE_SCHEMES:
        parser.error("--distance 适用于 explicit、gauging、high_rate；其他方案保留原生构造参数")
    return parser, args


def construction_options(args):
    if args.scheme in ("explicit", "gauging"):
        return {
            "require_distance": args.distance,
            "distance_options": {"distance_method": "auto", "distance_time_limit": 60.0},
        }
    if args.scheme == "high_rate":
        return {"ensure_distance": args.distance, "distance_method": "auto", "distance_time_limit": 60.0}
    # CKBB layers and devised-sticking repetition length retain their native
    # defaults. Branching constructs a routing code, not a measurement.
    return {}


def print_target(title, target):
    print(f"  {title}物理重量：{target['weight']}")
    print(f"  {title}带符号物理 Pauli：{target['pauli']}")
    print(f"  {title}物理支撑（1-based）：{[q + 1 for q in target['support']]}")


def print_code(title, result, error):
    print(f"\n{title}：")
    if error is not None:
        print(f"  构造失败：{type(error).__name__}: {error}")
        return
    if result is None:
        print("  未构造")
        return
    incidence = np.bitwise_or(result.check_x, result.check_z)
    row_weights = incidence.sum(axis=1)
    column_weights = incidence.sum(axis=0)
    print(f"  物理比特数：{result.n}")
    print(f"  校验行数：{incidence.shape[0]}")
    print(f"  最大 check 重量：{int(row_weights.max()) if row_weights.size else 0}")
    print(f"  最大物理 qubit 度数：{int(column_weights.max()) if column_weights.size else 0}")
    print(f"  protected logical qubits：{result.protected_k}")
    print("  已核验：校验对易、带符号的输入代表及实际逻辑读出")


def print_overhead(comparison):
    report = comparison.overhead()
    before, after = report["original"], report["optimized"]
    status_labels = {"constructed": "已构造", "failed": "失败", "not_built": "未构造"}
    label = "分支阶段" if report["construction_stage"] == "branching_only" else "Code surgery"
    print(f"\n{label}开销对比（最终输出矩阵）：")
    print("  指标 | 不减重 | 减重后")
    print(f"  状态 | {status_labels[before['status']]} | {status_labels[after['status']]}")
    rows = (
        ("新增辅助比特（主要 overhead）", "ancilla_qubits", "count"),
        ("总物理比特", "total_qubits", "count"),
        ("辅助比特 / 原码比特", "ancilla_fraction", "percent"),
        ("总物理比特 / 原码比特", "total_qubit_factor", "factor"),
        ("校验行数", "check_rows", "count"),
        ("校验行数相对原码的变化", "check_row_delta", "count"),
        ("最大 check 重量", "max_check_weight", "count"),
        ("最大物理 qubit 度数", "max_qubit_degree", "count"),
    )

    def display(value, kind):
        if value is None:
            return "不可用"
        if kind == "percent":
            return f"{value:.2%}"
        if kind == "factor":
            return f"{value:.3f}"
        return str(value)

    for label, key, kind in rows:
        print(f"  {label} | {display(before[key], kind)} | {display(after[key], kind)}")
    savings = report["savings"]
    if savings is None:
        print("  辅助比特节省量／比例：不可用（需要两份构造均成功）")
    else:
        percent = savings["ancilla_percent"]
        percent_text = "不可用（原始辅助比特为 0）" if percent is None else f"{percent:.2f}%"
        print(f"  辅助比特节省：{savings['ancilla_qubits']}；节省比例：{percent_text}")
        print("  节省量 = 不减重 − 减重后；负值表示减重后的构造开销更大。")
    print("  计入已执行的 expansion、稀疏化和重复层；不含校验提取电路辅助比特及测量轮次。")
    print("  这是当前构造参数下的开销，不代表最小开销。")
    print("  是否保持码距需分别查看两份构造的距离证据。")


def main():
    parser, args = parse_args()
    code = ibm_bb90_code()
    logicals = [logical.strip().upper() for logical in (args.logical or ["ZIIIIIII"])]
    for logical in logicals:
        label = logical[1:] if logical[:1] in ("+", "-") else logical
        if len(label) != code.k or any(p not in "IXYZ" for p in label):
            parser.error(f"每个 --logical 目标需要 {code.k} 个 I/X/Y/Z 字符，前面可加一个 +/- 符号")
        if not any(p != "I" for p in label):
            parser.error("逻辑目标不能是恒等算符")
    if len(logicals) > 1 and args.scheme in ("ckbb", "explicit", "gauging"):
        parser.error(f"{args.scheme} 比较接口接受一个完整观测量；多个独立目标可选择 devised_sticking")
    print(f"实际导入路径：{codesurgery.__file__}")
    print(f"输入码：{code.name}；n={code.n}, k={code.k}, 声明的 d={code.distance}")
    print(f"方案：{args.scheme}；逻辑目标：{', '.join(logicals)}")
    print("逻辑基：包的 standard form；每个字符串是一个观测量，多个字符串请求各自的结果")
    print("减重阈值：" + ("无限制尝试" if args.always else "严格 wt > 2d，未知 d 时跳过"))
    print("码距后处理：" + ("开启，两份构造分别处理" if args.distance else "关闭（保留方案原生重复层数）"))
    if args.scheme == "branching":
        print("注意：branching 只分支/路由逻辑代表，保留全部逻辑比特，本身不执行测量。")

    options = {"method": "heuristic", "seed": args.seed}
    if args.always:
        options["threshold"] = None
    failed = False
    try:
        comparison = compare_deformations(
            code,
            logical_paulis=logicals,
            scheme=args.scheme,
            optimization_options=options,
            construction_options=construction_options(args),
        )
    except ComparisonConstructionError as error:
        # Preserve successful branches even when reduction or construction fails.
        comparison = error.comparison
        failed = True
    except (ValueError, TypeError, ImportError) as error:
        print(f"输入或预处理失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    after_targets = comparison.optimized_targets
    for index, before in enumerate(comparison.original_targets):
        print(f"\n目标 {index + 1}：")
        print_target("不减重的", before)
        if index >= len(after_targets):
            print("  减重失败：未生成有效的减重代表；原因见下方及保存的报告。")
            continue
        print_target("减重后的", after_targets[index])
        report = comparison.optimization[index]
        print(f"  减重状态：{report['status']}")
        print(f"  已证明代表重量全局最优：{report.get('optimal', False)}")
    print_code("不减重的构造", comparison.original, comparison.errors.get("original"))
    print_code("减重后的构造", comparison.optimized, comparison.errors.get("optimized"))
    print_overhead(comparison)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory = args.output_dir.expanduser().resolve() / f"{stamp}-{args.scheme}"
    report_path = comparison.save(directory, matrix_format="npz")
    print(f"\n比较报告：{report_path}")
    for name in ("original", "optimized"):
        if getattr(comparison, name) is not None:
            print(f"变形码矩阵：{directory / (name + '.npz')}")
    print("报告中的 support 索引从 0 开始；上面打印的物理支撑从 1 开始。")
    print("重量不一定降低；重量降低也不保证辅助比特更少。码距证据见各构造的 metadata。")
    if failed:
        print("至少一个构造失败；成功结果与失败原因均已保存。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
