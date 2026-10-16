#!/usr/bin/env python3
"""用户自行运行的 codesurgery 测试脚本（Python >= 3.10）。

    python -m pytest tests/test_package_smoke.py
    python tests/test_package_smoke.py
    python tests/test_package_smoke.py --bb90
    python tests/test_package_smoke.py --bb90 --distance

默认：小码测试各方案、逻辑读出、矩阵导出，以及 H/S/CNOT 测量分解。
--bb90：额外测试 IBM [[90,8,10]] 的 gauging X/Z/Y/XZ/XZXZ。
--distance：额外计算原码与 gauging-Z 输出的静态码距，默认 Gurobi。
            只计算/报告，不自动扩展或稀疏化；超时返回上下界不是精确值。
            每次 MILP 求解默认限时 60 秒，不是整个脚本总限时。

构造用例显式关闭自动码距后处理，方便把代数错误与求解开销分开检查。
不会安装依赖或改变 sys.path；测试的是当前 Python 实际导入的包。
PASS 表示该项断言通过，不代表带噪测量电路的容错码距已经证明。
每次运行创建独立结果目录，保存 report.json、deformed 矩阵及 gate plan。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

try:
    import numpy as np
    import codesurgery
except ImportError as error:
    raise SystemExit(
        f"无法导入包：{error}\n请先在运行本脚本的 Python 环境安装 codesurgery。\n"
        '在项目根目录运行：python -m pip install -e ".[test]"'
    ) from error


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能写入 JSON：{type(value).__name__}")


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2,
                               default=json_value) + "\n", encoding="utf-8")


def logical_signature(label):
    """如 XZ = X_0 Z_1；先放 X 系数，再放 Z 系数。"""
    return np.array([p in "XY" for p in label] + [p in "ZY" for p in label],
                    dtype=np.uint8)


def logical_input(label, sign=0):
    signature = logical_signature(label)
    return dict(logical_x=signature[:len(label)],
                logical_z=signature[len(label):], sign=sign)


def check_basis(code):
    basis = code.standard_logicals()
    require(not np.any((code.hx @ code.hz.T) % 2), "原码 HX/HZ 不对易")
    require(basis.jx.shape == basis.jz.shape == (code.k, code.n), "逻辑基维度错误")
    require(not np.any((basis.jx @ code.hz.T) % 2), "JX 不在 normalizer 中")
    require(not np.any((basis.jz @ code.hx.T) % 2), "JZ 不在 normalizer 中")
    require(np.array_equal((basis.jx @ basis.jz.T) % 2,
                           np.eye(code.k, dtype=np.uint8)), "逻辑基未正确配对")
    return dict(n=code.n, k=code.k, declared_distance=code.distance,
                note="declared_distance 是输入参数，不是本项重新计算的结果")


def check_deformation(code, result, target_labels):
    """独立重算对易矩阵，并从最终 check 消去辅助比特检查实际读出空间。"""
    from codesurgery.gf2 import nullspace, rank

    verdict = result.validate()  # 各方案内部的相位、witness 和结构检查。
    require(verdict is not False, "validate() 返回 False")
    if isinstance(verdict, dict):
        require(verdict.get("valid", True), "validate() 报告构造无效")
    x, z = np.asarray(result.check_x), np.asarray(result.check_z)
    signs = np.asarray(result.check_signs)
    require(x.ndim == 2 and x.shape == z.shape, "check_x/check_z 形状不匹配")
    require(signs.shape == (len(x),), "check_signs 形状不匹配")
    for value in (x, z, signs):
        require(np.all((value == 0) | (value == 1)), "矩阵/符号包含非二元值")
    require(result.original_n == code.n and x.shape[1] == result.n, "比特数错误")
    require(result.n >= code.n, "输出未保留完整 memory 坐标")
    # Y 计一个物理作用；不可直接把 X/Z 两块的重量相加。
    occupied = np.bitwise_or(x, z)
    commutator = (x.astype(np.int64) @ z.T + z.astype(np.int64) @ x.T) % 2
    require(not np.any(commutator), "存在反对易的输出 check")

    # u A = 0：仅保留消去所有辅助 X/Z 支撑的 check 乘积。
    auxiliary = np.hstack((x[:, code.n:], z[:, code.n:]))
    kernel = nullspace(auxiliary.T)
    memory_x, memory_z = (kernel @ x[:, :code.n]) % 2, (kernel @ z[:, :code.n]) % 2
    require(not np.any((memory_x @ code.hz.T) % 2), "读出了非逻辑 memory X 信息")
    require(not np.any((memory_z @ code.hx.T) % 2), "读出了非逻辑 memory Z 信息")
    basis = code.standard_logicals()
    actual = np.hstack(((memory_x @ basis.jz.T) % 2,
                        (memory_z @ basis.jx.T) % 2))
    expected = np.array([logical_signature(s) for s in target_labels],
                        dtype=np.uint8).reshape(len(target_labels), 2 * code.k)
    expected_rank = rank(expected)
    require(rank(actual) == expected_rank, "实际读出的独立逻辑数与目标不同")
    require(rank(np.vstack((actual, expected))) == expected_rank,
            "实际读出的逻辑空间与目标不同")
    require(result.protected_k == code.k - expected_rank, "剩余 protected_k 错误")
    return dict(deformed_n=result.n, checks=len(x), protected_k=result.protected_k,
                gauge_qubits=result.gauge_qubits, logical_readout_rank=expected_rank,
                max_check_weight=int(occupied.sum(axis=1).max(initial=0)),
                max_qubit_degree=int(occupied.sum(axis=0).max(initial=0)),
                target_labels=target_labels, summary=result.summary())


def export_deformation(result, destination):
    destination.mkdir()
    result.save(destination / "deformed.npz")
    result.save(destination / "deformed.json")
    # 读取保存文件，防止导出的矩阵与内存中的结果不同。
    with np.load(destination / "deformed.npz", allow_pickle=False) as saved:
        for name in ("check_x", "check_z", "check_signs"):
            require(np.array_equal(saved[name], getattr(result, name)),
                    f"NPZ 导出后的 {name} 不一致")
    saved_json = json.loads((destination / "deformed.json").read_text(encoding="utf-8"))
    for name in ("check_x", "check_z", "check_signs"):
        require(np.array_equal(saved_json[name], getattr(result, name)),
                f"JSON 导出后的 {name} 不一致")


def check_gate_protocols():
    """逻辑 Hilbert 空间中枚举每个门的测量分支，独立验证 Pauli frame。"""
    from codesurgery.pauli_gates import decompose_gates

    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    paulis = dict(I=identity, X=x, Y=1j * x @ z, Z=z)

    def tensor(factors):
        out = np.ones((1, 1), dtype=complex)
        for factor in factors:
            out = np.kron(out, factor)
        return out

    def pauli_matrix(label):
        return tensor([paulis[p] for p in label])

    details = {}
    for gate, k, operands in (("H", 2, (1,)), ("S", 2, (1,)), ("CNOT", 3, (1, 2))):
        plan = decompose_gates(k, [(gate, *operands)], ancilla=0)
        # 辅助 logical 0 初态 |0>，其余列覆盖任意数据输入态。
        initial = np.eye(2 ** k, dtype=complex)[:, :2 ** (k - 1)]
        if gate in ("H", "S"):
            one_gate = (x + z) / np.sqrt(2) if gate == "H" else np.diag([1, 1j])
            unitary = tensor([identity, one_gate])
        else:
            unitary = np.zeros((8, 8), dtype=complex)
            for column in range(8):
                unitary[column ^ ((column >> 1) & 1), column] = 1
        desired = unitary @ initial
        full_identity = np.eye(2 ** k, dtype=complex)
        count, probability_sum = 0, 0.0
        for bits in product((0, 1), repeat=len(plan.measurements)):
            branch = initial.copy()
            for measurement, bit in zip(plan.measurements, bits):
                branch = (full_identity + (-1) ** bit * pauli_matrix(measurement.logical_pauli)) @ branch / 2
            if np.linalg.norm(branch) < 1e-12:
                continue
            resolved = plan.resolve(list(bits))  # 枚举的模拟分支，不是真实实验结果。
            expected = pauli_matrix(resolved["correction_pauli"]) @ desired
            amplitude = np.vdot(expected, branch) / np.vdot(expected, expected)
            require(np.allclose(branch, amplitude * expected, atol=1e-10, rtol=1e-10),
                    f"{gate} 分支 {bits} 的测量分解/Pauli frame 错误")
            probability_sum += float(abs(amplitude) ** 2)
            count += 1
        require(count > 0 and np.isclose(probability_sum, 1), f"{gate} 分支不完备")
        details[gate] = dict(nonzero_branches=count, probability_sum=probability_sum)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bb90", action="store_true")
    parser.add_argument("--distance", action="store_true")
    parser.add_argument("--distance-method", choices=("gurobi", "auto", "enumeration", "milp", "cluster", "ris"),
                        default="gurobi")
    parser.add_argument("--time-limit", type=float, default=60.0, help="每次 MILP 求解秒数")
    parser.add_argument("--output-dir", type=Path, default=Path("codesurgery-test-results"))
    parser.add_argument("--verbose", action="store_true", help="终端显示失败堆栈；报告始终保存堆栈")
    args = parser.parse_args(argv)
    if not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit 必须是正数")

    from codesurgery import CSSCode, ckbb_deform
    from codesurgery.examples import four_qubit_code, direct_sum, ibm_bb90_code
    from codesurgery.devised_sticking import devised_deform
    from codesurgery.explicit_gadgets import gadget_deform
    from codesurgery.gauging_measurement import gauging_deform
    from codesurgery.high_rate_surgery import high_rate_deform
    from codesurgery.parallel_branching import branching_deform
    from codesurgery.pauli_gates import compile_gates, decompose_gates

    destination = args.output_dir.expanduser().resolve() / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        package_version = version("codesurgery")
    except PackageNotFoundError:
        package_version = "未找到安装元数据"
    report = dict(python=sys.executable, package_path=str(codesurgery.__file__),
                  package_version=package_version, options=vars(args), cases=[])
    print(f"Python：{sys.executable}\n包路径：{codesurgery.__file__}\n版本：{package_version}", flush=True)
    print(f"结果目录：{destination}\n逻辑编号从 0 开始；门测试保留 logical 0 为辅助比特。", flush=True)
    print("默认构造关闭自动码距后处理；PASS 不等于距离保证。", flush=True)

    def run(name, action, status="PASS"):
        print(f"\n[RUN] {name}", flush=True)
        start = perf_counter()
        record = dict(name=name)
        try:
            record.update(status=status, details=action())
        except Exception as error:
            record.update(status="FAIL", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
            print(record["error"], flush=True)
            if args.verbose:
                print(record["traceback"], flush=True)
        record["seconds"] = round(perf_counter() - start, 3)
        report["cases"].append(record)
        write_json(destination / "report.json", report)
        print(f"[{record['status']}] {name} ({record['seconds']:.3f}s)", flush=True)

    built = {}

    def surgery_case(name, code, targets, build):
        def action():
            result = build()
            detail = check_deformation(code, result, targets)
            export_deformation(result, destination / name)
            built[name] = result
            print(f"  n={result.n}, protected_k={result.protected_k}, "
                  f"最大行重量={detail['max_check_weight']}, 最大列度数={detail['max_qubit_degree']}", flush=True)
            return detail
        run(name, action)

    four = four_qubit_code()
    run("standard_basis_four", lambda: check_basis(four))
    surgery_case("ckbb_Z", four, ["ZI"], lambda: ckbb_deform(four, pauli="Z", layers=2))
    surgery_case("devised_Z", four, ["ZI"], lambda: devised_deform(four, pauli="Z", d_R=2))
    surgery_case("explicit_Z", four, ["ZI"], lambda: gadget_deform(four, pauli="Z", require_distance=False))
    for axis in "XYZ":
        surgery_case(f"gauging_{axis}", four, [axis + "I"],
                     lambda axis=axis: gauging_deform(four, pauli=axis, require_distance=False))
    surgery_case("gauging_XZ", four, ["XZ"],
                 lambda: gauging_deform(four, logical_input("XZ"), require_distance=False))
    # 两个独立读出 vs 一个乘积读出，必须保留不同数量的逻辑比特。
    surgery_case("high_rate_Z0_Z1", four, ["ZI", "IZ"],
                 lambda: high_rate_deform(four, logical_paulis=["ZI", "IZ"], ensure_distance=False))
    surgery_case("high_rate_Z0Z1", four, ["ZZ"],
                 lambda: high_rate_deform(four, logical_paulis=["ZZ"], ensure_distance=False))
    # Branching core 只分枝，不测量；期望实际读出空间为零。
    surgery_case("branching_Z0_Z1", four, [], lambda: branching_deform(
        four, logical_paulis=["ZI", "IZ"], branching_mode="always"))

    run("H_S_CNOT_all_outcome_branches", check_gate_protocols)

    def gate_backend():
        code = direct_sum(four_qubit_code(), four_qubit_code())
        plan = compile_gates(code, [("H", 1), ("S", 1), ("CNOT", 1, 2)], ancilla=0,
                             backend="gauging", backend_options={"require_distance": False})
        plan.validate()
        require(len(plan.measurements) == 11, "H/S/CNOT 测量步数错误（含辅助态准备）")
        require(set(plan.gadgets) == {m.id for m in plan.measurements}, "有测量步骤未构造 gadget")
        for measurement in plan.measurements:
            check_deformation(code, plan.gadgets[measurement.id], [measurement.logical_pauli])
        saved = plan.save(destination / "gate_H_S_CNOT", matrix_format="npz")
        require(Path(saved).is_file(), "gate plan 未保存")
        return plan.summary()

    run("H_S_CNOT_gauging_backend", gate_backend)

    def invalid_inputs():
        operations = {
            "invalid_CSS": lambda: CSSCode([[1]], [[1]]),
            "ancilla_as_data": lambda: decompose_gates(2, [("H", 0)], ancilla=0),
            "CNOT_missing_data_qubit": lambda: decompose_gates(2, [("CNOT", 1, 2)]),
        }
        messages = {}
        for name, operation in operations.items():
            try:
                operation()
            except ValueError as error:
                messages[name] = str(error)
            else:
                raise AssertionError(f"{name} 应拒绝非法输入，却没有报错")
        return messages

    run("invalid_inputs_rejected", invalid_inputs)

    if args.bb90:
        bb90 = ibm_bb90_code()
        run("standard_basis_bb90", lambda: check_basis(bb90))
        for label in ("XIIIIIII", "ZIIIIIII", "YIIIIIII", "XZIIIIII", "XZXZIIII"):
            surgery_case(f"bb90_gauging_{label}", bb90, [label],
                         lambda label=label: gauging_deform(bb90, logical_input(label),
                             require_distance=False, exact_limit=1, optimize_logical=True))

    if args.distance:
        from codesurgery.distance_preservation import code_distance
        distance_source = bb90 if args.bb90 else four
        deformed_name = "bb90_gauging_ZIIIIIII" if args.bb90 else "gauging_Z"

        def distance_report(code):
            result = code_distance(code, method=args.distance_method, time_limit=args.time_limit, seed=0)
            result["interpretation"] = "静态 stabilizer 码距；若有 gauge qubits，会把它们也计入逻辑空间"
            print(f"  {result.get('distance_statement', result.get('status'))}", flush=True)
            print("  REPORT 只记录求解结果/上下界，不表示已达到输入码距。", flush=True)
            return result

        run("distance_source", lambda: distance_report(distance_source), status="REPORT")
        if deformed_name in built:
            run("distance_gauging_Z", lambda: distance_report(built[deformed_name]), status="REPORT")
        else:
            print("[SKIP] gauging-Z 构造未通过，无法计算其码距。", flush=True)

    failures = sum(item["status"] == "FAIL" for item in report["cases"])
    report["failure_count"] = failures
    write_json(destination / "report.json", report)
    print(f"\n完成：{len(report['cases'])} 项，失败 {failures} 项。\n报告：{destination / 'report.json'}", flush=True)
    return 1 if failures else 0


def test_codesurgery_package_with_bb90(tmp_path):
    """PyCharm/pytest 入口：测试全部小码案例及 BB90 的代数构造。"""
    exit_code = main([
        "--bb90",
        "--output-dir", str(tmp_path / "codesurgery-results"),
    ])
    require(exit_code == 0, "至少一个 codesurgery 测试案例失败；查看终端和 report.json")


def test_codesurgery_bb90_distance_when_enabled(tmp_path, monkeypatch):
    """可选慢测试：CODESURGERY_TEST_DISTANCE=1 时使用 Gurobi，单次限时 60 秒。"""
    import os
    import pytest

    if os.environ.get("CODESURGERY_TEST_DISTANCE") != "1":
        pytest.skip("设置 CODESURGERY_TEST_DISTANCE=1 后运行 BB90/Gurobi 码距测试")
    exit_code = main([
        "--bb90", "--distance", "--distance-method", "gurobi",
        "--time-limit", "60",
        "--output-dir", str(tmp_path / "codesurgery-distance-results"),
    ])
    require(exit_code == 0, "BB90 码距测试失败；查看终端和 report.json")


if __name__ == "__main__":
    raise SystemExit(main())
