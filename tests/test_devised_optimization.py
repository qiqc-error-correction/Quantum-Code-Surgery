"""Regression tests for signed representative selection before devised sticking.

All searches are deterministic tiny-code heuristics; no distance solver is used.
"""
import copy

import numpy as np
import pytest

from codesurgery import CSSCode
from codesurgery.devised_sticking import devised_deform
from codesurgery.examples import direct_sum, steane_code


HEURISTIC = {"method": "heuristic", "max_trials": 1, "seed": 0}


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_disabled_preserves_exact_negative_input_and_identity_witness(pauli, monkeypatch):
    import codesurgery.logical_optimization as optimization

    def unexpected_search(*args, **kwargs):
        pytest.fail("Disabled devised optimization must not invoke its search engine")

    monkeypatch.setattr(optimization, "optimize_prepared_logical", unexpected_search)
    code = steane_code()
    original_hx, original_hz = code.hx.copy(), code.hz.copy()
    result = devised_deform(code, "-" + pauli * 7, pauli=pauli,
                            d_R=2, ldpc=False, optimize_logical=False)
    selected = result.measurement_x if pauli == "X" else result.measurement_z
    assert np.array_equal(selected, np.ones((1, 7), dtype=np.uint8))
    assert result.measurement_signs.tolist() == [1]
    assert np.array_equal(code.hx, original_hx)
    assert np.array_equal(code.hz, original_hz)
    report = result.metadata["logical_optimization"][0]
    assert report["status"] == "disabled"
    assert report["before_weight"] == report["after_weight"] == 7
    assert not any(report["hx_coefficients"] + report["hz_coefficients"])
    assert report["original_sign"] == report["sign"] == 1
    assert result.summary()["input_measurement_weights"] == [7]
    assert result.summary()["selected_measurement_weights"] == [7]
    assert result.validate()


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_weight_seven_steane_target_reduces_to_three_preserving_sign(pauli):
    code = steane_code()
    result = devised_deform(code, "-" + pauli * 7, pauli=pauli, d_R=2,
                            ldpc=False, optimize_logical=True,
                            logical_optimization_options=HEURISTIC)
    report = result.metadata["logical_optimization"][0]
    assert report["threshold_rule"] == "2d"
    assert report["threshold"] == 6
    assert report["before_weight"] == 7
    assert report["after_weight"] == 3
    assert report["original_signature"] == report["optimized_signature"]
    assert result.measurement_signs.tolist() == [1]
    chosen = result.measurement_x if pauli == "X" else result.measurement_z
    # First heuristic move multiplies the first weight-four Steane stabilizer.
    checks = code.hx if pauli == "X" else code.hz
    assert np.array_equal(chosen[0], np.ones(7, dtype=np.uint8) ^ checks[0])
    assert result.validate()


def test_default_two_d_threshold_is_strict_and_none_forces_attempt():
    # Qubit 1 is fixed by +X_1; qubit 2 carries the one distance-one logical.
    # X_1 X_2 has weight exactly 2d and can become X_2 by multiplying X_1.
    code = CSSCode([[1, 0]], np.zeros((0, 2), dtype=np.uint8), distance=1)
    skipped = devised_deform(code, "XX", pauli="X", d_R=2, ldpc=False,
                             optimize_logical=True, logical_optimization_options=HEURISTIC)
    forced = devised_deform(code, "XX", pauli="X", d_R=2, ldpc=False,
                            optimize_logical=True,
                            logical_optimization_options={**HEURISTIC, "threshold": None})
    skip_report = skipped.metadata["logical_optimization"][0]
    force_report = forced.metadata["logical_optimization"][0]
    assert skip_report["status"] == "skipped_threshold"
    assert skip_report["before_weight"] == skip_report["after_weight"] == 2
    assert force_report["threshold_rule"] == "always"
    assert force_report["after_weight"] == 1
    assert forced.measurement_x.tolist() == [[0, 1]]
    assert skipped.validate() and forced.validate()


def test_multiple_independent_targets_keep_order_and_signed_classes():
    code = direct_sum(steane_code(), steane_code())
    targets = ["-" + "X" * 7 + "I" * 7, "I" * 7 + "X" * 7]
    result = devised_deform(code, targets, pauli="X", d_R=2, ldpc=False,
                            optimize_logical=True,
                            logical_optimization_options=HEURISTIC)
    reports = result.metadata["logical_optimization"]
    assert len(reports) == 2
    assert [r["before_weight"] for r in reports] == [7, 7]
    assert [r["after_weight"] for r in reports] == [3, 3]
    assert result.measurement_signs.tolist() == [1, 0]
    assert not np.any(result.measurement_x[0, 7:])
    assert not np.any(result.measurement_x[1, :7])
    before = result.metadata["input_measurement_x"]
    logical_z = code.standard_logicals().jz
    assert np.array_equal((before @ logical_z.T) % 2,
                          (result.measurement_x @ logical_z.T) % 2)
    assert result.metadata["logical_measurement_rank"] == 2
    assert result.protected_k == 0
    assert result.validate()


@pytest.mark.parametrize("tampering", ["coefficient", "sign", "input"])
def test_validate_rejects_tampered_equivalence_records(tampering):
    result = devised_deform(steane_code(), "-XXXXXXX", pauli="X", d_R=2,
                            ldpc=False, optimize_logical=True,
                            logical_optimization_options=HEURISTIC)
    bad = copy.deepcopy(result)
    report = bad.metadata["logical_optimization"][0]
    if tampering == "coefficient":
        report["hx_coefficients"][0] ^= 1
    elif tampering == "sign":
        report["sign"] ^= 1
    else:
        bad.metadata["input_measurement_x"][0, 0] ^= 1
    # Cached assertions must not suppress the actual signed-witness checks.
    report["equivalence_verified"] = True
    report["signature_verified"] = True
    with pytest.raises(ValueError):
        bad.validate()
