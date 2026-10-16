"""User-facing raw/reduced surgery comparisons on deterministic small codes."""
import json

import numpy as np
import pytest

import codesurgery.measurement_comparison as comparison_module
from codesurgery import CSSCode
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import nullspace, rank
from codesurgery.measurement_comparison import (
    ComparisonConstructionError,
    compare_deformations,
)
from codesurgery.symplectic import pauli_product


HEURISTIC = {"method": "heuristic", "max_trials": 1, "seed": 0}
BASIC_GAUGING = {"require_distance": False}


def _steane_comparison(**kwargs):
    return compare_deformations(
        steane_code(), "-XXXXXXX", scheme="gauging",
        optimization_options=HEURISTIC,
        construction_options=BASIC_GAUGING, **kwargs)


def test_comparison_builds_both_exact_requested_representatives():
    comparison = _steane_comparison()
    assert comparison.errors == {}
    assert comparison.original is not None
    assert comparison.optimized is not None
    assert comparison.original_targets[0]["weight"] == 7
    assert comparison.optimized_targets[0]["weight"] == 3
    assert comparison.original.measurement_sign == 1
    assert comparison.optimized.measurement_sign == 1
    assert np.array_equal(comparison.original.measurement_x,
                          comparison.original_targets[0]["x"])
    assert np.array_equal(comparison.optimized.measurement_x,
                          comparison.optimized_targets[0]["x"])
    assert comparison.original.protected_k == comparison.optimized.protected_k == 0
    assert comparison.summary()["before_weights"] == [7]
    assert comparison.summary()["after_weights"] == [3]
    assert comparison.validate()


@pytest.mark.parametrize("logical_label, signature", [
    ("-YI", [1, 0, 1, 0]),
    ("-XZ", [1, 0, 0, 1]),
])
def test_y_and_xz_reduction_preserves_signed_logical_class(logical_label, signature):
    code = four_qubit_code()
    comparison = compare_deformations(
        code, logical_paulis=logical_label, scheme="gauging",
        optimization_options={**HEURISTIC, "threshold": None},
        construction_options=BASIC_GAUGING)
    before, after = comparison.original_targets[0], comparison.optimized_targets[0]
    report = comparison.optimization[0]
    assert before["logical_signature"] == after["logical_signature"] == signature
    assert after["weight"] <= before["weight"]
    # Independently multiply the signed input with the recorded positive CSS
    # generators, rather than equating literal signs (mixed Paulis can flip).
    x = np.vstack((before["x"], code.hx, np.zeros_like(code.hz)))
    z = np.vstack((before["z"], np.zeros_like(code.hx), code.hz))
    signs = np.zeros(len(x), dtype=np.uint8)
    signs[0] = before["sign"]
    coefficients = [1] + report["hx_coefficients"] + report["hz_coefficients"]
    out_x, out_z, phase = pauli_product(x, z, signs,
                                      selected=np.flatnonzero(coefficients))
    assert np.array_equal(out_x, after["x"])
    assert np.array_equal(out_z, after["z"])
    assert phase == 2 * after["sign"]
    assert comparison.original.protected_k == comparison.optimized.protected_k == 1
    assert comparison.validate()


def test_mixed_y_reduction_carries_required_literal_sign_change():
    checks = np.ones((1, 6), dtype=np.uint8)
    code = CSSCode(checks, checks, distance=2)
    comparison = compare_deformations(
        code, "-YYYYII", scheme="gauging",
        optimization_options={**HEURISTIC, "threshold": None},
        construction_options=BASIC_GAUGING)
    # X^6 Z^6 = -Y^6, so (-YYYYII) * X^6 * Z^6 = +IIIIYY.
    before, after = comparison.original_targets[0], comparison.optimized_targets[0]
    assert before["weight"] == 4 and before["sign"] == 1
    assert after["weight"] == 2 and after["sign"] == 0
    assert after["x"] == after["z"] == [0, 0, 0, 0, 1, 1]
    assert comparison.optimization[0]["hx_coefficients"] == [1]
    assert comparison.optimization[0]["hz_coefficients"] == [1]
    assert comparison.validate()


def test_one_product_differs_from_two_independent_measurements():
    code = four_qubit_code()
    common = dict(scheme="devised_sticking", optimization_options=HEURISTIC,
                  construction_options={"d_R": 2, "ldpc": False})
    product = compare_deformations(code, logical_paulis=["ZZ"], **common)
    separate = compare_deformations(code, logical_paulis=["ZI", "IZ"], **common)
    for name in ("original", "optimized"):
        one, two = getattr(product, name), getattr(separate, name)
        assert len(one.measurement_x) == 1
        assert len(two.measurement_x) == 2
        assert one.metadata["logical_measurement_rank"] == 1
        assert two.metadata["logical_measurement_rank"] == 2
        assert one.protected_k == 1
        assert two.protected_k == 0
    assert product.validate() and separate.validate()


def test_original_failure_still_attempts_optimized_and_retains_success(monkeypatch, tmp_path):
    native_build = comparison_module._build
    attempted_weights = []

    def reject_weight_seven(code, targets, basis, scheme, options):
        weight = int(np.count_nonzero(targets[0][0] | targets[0][1]))
        attempted_weights.append(weight)
        if weight == 7:
            raise ValueError("Deliberate original-construction failure")
        return native_build(code, targets, basis, scheme, options)

    monkeypatch.setattr(comparison_module, "_build", reject_weight_seven)
    with pytest.raises(ComparisonConstructionError) as caught:
        _steane_comparison()
    comparison = caught.value.comparison
    assert attempted_weights == [7, 3]
    assert comparison.original is None
    assert comparison.optimized is not None
    assert set(comparison.errors) == {"original"}
    assert comparison.summary()["original"]["status"] == "failed"
    assert comparison.summary()["optimized"]["status"] == "constructed"
    assert comparison.validate()
    report_path = comparison.save(tmp_path / "partial", matrix_format="json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["code_files"] == {"optimized": "optimized.json"}
    assert not (report_path.parent / "original.json").exists()
    assert (report_path.parent / "optimized.json").is_file()


@pytest.mark.parametrize("strict", [False, True])
def test_optimizer_failure_retains_the_successful_original_build(monkeypatch, strict):
    def failed_optimizer(*args, **kwargs):
        raise RuntimeError("Deliberate weight-reduction failure")

    monkeypatch.setattr(comparison_module, "optimize_prepared_logical", failed_optimizer)
    if strict:
        with pytest.raises(ComparisonConstructionError) as caught:
            _steane_comparison(strict=True)
        comparison = caught.value.comparison
    else:
        comparison = _steane_comparison(strict=False)
    assert comparison.original is not None
    assert comparison.original_targets[0]["weight"] == 7
    assert comparison.optimized is None
    assert comparison.optimized_targets == []
    assert set(comparison.errors) == {"optimized"}
    assert isinstance(comparison.errors["optimized"], RuntimeError)
    assert comparison.summary()["original"]["status"] == "constructed"
    assert comparison.summary()["optimized"]["status"] == "failed"
    assert comparison.summary()["after_weights"] is None
    assert comparison.validate()


def test_branching_preserves_logicals_without_measuring_them():
    code = four_qubit_code()
    comparison = compare_deformations(
        code, logical_paulis=["ZI", "IZ"], scheme="branching",
        optimization_options=HEURISTIC,
        construction_options={"branching_mode": "always"})
    basis = code.standard_logicals()
    for result in (comparison.original, comparison.optimized):
        assert result.protected_k == code.k
        cancel = nullspace(np.hstack((result.check_x[:, code.n:],
                                      result.check_z[:, code.n:])).T)
        memory_x = (cancel @ result.check_x[:, :code.n]) % 2
        memory_z = (cancel @ result.check_z[:, :code.n]) % 2
        logical_readout = np.hstack(((memory_x @ basis.jz.T) % 2,
                                    (memory_z @ basis.jx.T) % 2))
        assert rank(logical_readout) == 0
    assert comparison.validate()


@pytest.mark.parametrize("matrix_format", ["json", "npz"])
def test_save_persists_two_codes_and_refuses_overwrite(matrix_format, tmp_path):
    comparison = _steane_comparison()
    output = tmp_path / matrix_format
    report_path = comparison.save(output, matrix_format=matrix_format)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["before_weights"] == [7]
    assert report["summary"]["after_weights"] == [3]
    assert len(report["optimization"]) == 1
    for name in ("original", "optimized"):
        path = output / report["code_files"][name]
        native = getattr(comparison, name)
        if matrix_format == "json":
            matrix = json.loads(path.read_text(encoding="utf-8"))
            assert np.array_equal(matrix["check_x"], native.check_x)
            assert np.array_equal(matrix["check_z"], native.check_z)
        else:
            with np.load(path, allow_pickle=False) as matrix:
                assert np.array_equal(matrix["check_x"], native.check_x)
                assert np.array_equal(matrix["check_z"], native.check_z)
    original_report = report_path.read_bytes()
    with pytest.raises(FileExistsError):
        comparison.save(output, matrix_format=matrix_format)
    assert report_path.read_bytes() == original_report


def test_build_false_returns_representatives_without_builders(monkeypatch):
    def unexpected_build(*args, **kwargs):
        pytest.fail("build=False must not invoke a surgery builder")

    monkeypatch.setattr(comparison_module, "_build", unexpected_build)
    comparison = _steane_comparison(build=False)
    assert comparison.original is None and comparison.optimized is None
    assert comparison.errors == {}
    assert comparison.summary()["before_weights"] == [7]
    assert comparison.summary()["after_weights"] == [3]
    assert comparison.summary()["original"]["status"] == "not_built"
    assert comparison.validate()
