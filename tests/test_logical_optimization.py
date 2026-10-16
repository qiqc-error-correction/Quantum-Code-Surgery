"""Fixed logical cosets, signed Y products, and optimized gadget readout."""
from itertools import product
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, ibm_bb90_code, steane_code
from codesurgery.explicit_gadgets import gadget_deform
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.logical_optimization import reduce_logical_weight, verify_logical_optimization
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli, pauli_product


def css_checks(code):
    return (np.vstack([code.hx, np.zeros_like(code.hz)]),
            np.vstack([np.zeros_like(code.hx), code.hz]))


def brute_coset_weight(code, word):
    """Exhaust the fixed affine coset independently of the optimizer."""
    x, z, _ = parse_pauli(word, code.n)
    cx, cz = css_checks(code)
    best = code.n
    for bits in product((0, 1), repeat=len(cx)):
        bits = np.asarray(bits, dtype=np.uint8)
        rx, rz = x ^ (bits @ cx % 2), z ^ (bits @ cz % 2)
        best = min(best, int(np.count_nonzero(rx | rz)))
    return best


def assert_equivalence(code, word, report):
    x, z, sign = parse_pauli(word, code.n)
    rx, rz = np.asarray(report["x"]), np.asarray(report["z"])
    ax = np.asarray(report["hx_coefficients"], dtype=np.uint8)
    az = np.asarray(report["hz_coefficients"], dtype=np.uint8)
    np.testing.assert_array_equal(rx, x ^ (ax @ code.hx % 2))
    np.testing.assert_array_equal(rz, z ^ (az @ code.hz % 2))
    cx, cz = css_checks(code)
    chosen = np.flatnonzero(np.r_[ax, az]) + 1
    px, pz, phase = pauli_product(np.vstack([x, cx]), np.vstack([z, cz]),
                                 np.r_[sign, np.zeros(len(cx), dtype=np.uint8)],
                                 np.r_[0, chosen])
    np.testing.assert_array_equal(px, rx)
    np.testing.assert_array_equal(pz, rz)
    assert phase == 2 * report["sign"]
    basis = code.standard_logicals()
    original_signature = np.r_[x @ basis.jz.T % 2, z @ basis.jx.T % 2]
    actual_signature = np.r_[rx @ basis.jz.T % 2, rz @ basis.jx.T % 2]
    np.testing.assert_array_equal(actual_signature, original_signature)
    assert report["before_weight"] == int(np.count_nonzero(x | z))
    assert report["after_weight"] == int(np.count_nonzero(rx | rz))
    assert report["after_weight"] <= report["before_weight"]
    assert report["equivalence_verified"] and report["signature_verified"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_optimizer_keeps_requested_coset_even_when_other_logical_is_lighter(pauli):
    # The six-qubit code has weight-one logicals, but this requested class
    # has minimum weight three: 111011 + span(111100).
    stabilizer = np.array([[1, 1, 1, 1, 0, 0]], dtype=np.uint8)
    empty = np.zeros((0, 6), dtype=np.uint8)
    code = CSSCode(stabilizer if pauli == "X" else empty,
                   stabilizer if pauli == "Z" else empty)
    word = pauli * 3 + "I" + pauli * 2
    report = reduce_logical_weight(code, word, threshold=None)
    assert report["before_weight"] == 5
    assert report["after_weight"] == 3 == brute_coset_weight(code, word)
    assert report["optimal"]
    assert not np.any(report["z"] if pauli == "X" else report["x"])
    assert_equivalence(code, word, report)


@pytest.mark.parametrize("sign,expected_sign", [("", 1), ("-", 0)])
@pytest.mark.parametrize("method", ["auto", "milp"])
def test_y_counts_once_and_stabilizer_product_can_change_literal_sign(sign, expected_sign, method):
    if method == "milp":
        pytest.importorskip("scipy.optimize")
    # +XXI and +ZZI imply -YYI. Thus YYY = -IIY on the codespace.
    # The only weight-one representative in this fixed coset is +/-IIY.
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    word = sign + "YYY"
    report = reduce_logical_weight(code, word, method=method, threshold=None)
    np.testing.assert_array_equal(report["x"], [0, 0, 1])
    np.testing.assert_array_equal(report["z"], [0, 0, 1])
    assert report["sign"] == expected_sign
    assert report["after_weight"] == 1
    assert report["optimal"]
    assert_equivalence(code, word, report)


@pytest.mark.parametrize("word", ["XXXXXXX", "ZZZZZZZ", "-YYYYYYY", "-XZYIIII"])
@pytest.mark.parametrize("method", ["auto", "milp"])
def test_small_affine_optimum_matches_complete_stabilizer_enumeration(word, method):
    if method == "milp":
        pytest.importorskip("scipy.optimize")
    if word == "-XZYIIII":
        # A known mixed logical of the four-qubit code, extended by three
        # fixed Z ancillas, tests asymmetric stabilizer dimensions.
        code = CSSCode([[1, 1, 1, 1, 0, 0, 0]],
                       [[1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]])
    else:
        code = steane_code()
    report = reduce_logical_weight(code, word, method=method, threshold=None)
    assert report["after_weight"] == brute_coset_weight(code, word)
    assert report["optimal"]
    assert_equivalence(code, word, report)


def test_threshold_skip_keeps_original_physical_operator_and_phase():
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    report = reduce_logical_weight(code, "-YYY", threshold=3)
    assert report["before_weight"] == report["after_weight"] == 3
    assert report["sign"] == 1
    assert not report["optimal"]
    assert not np.any(report["hx_coefficients"])
    assert not np.any(report["hz_coefficients"])
    assert_equivalence(code, "-YYY", report)


def test_zero_trials_heuristic_never_worsens_input():
    code = steane_code()
    report = reduce_logical_weight(code, "-YYYYYYY", method="heuristic", max_trials=0)
    assert_equivalence(code, "-YYYYYYY", report)


def test_custom_basis_selects_its_own_logical_not_the_standard_first_one():
    code = four_qubit_code()
    standard = code.standard_logicals()
    custom = dict(jx=standard.jx[::-1], jz=standard.jz[::-1])
    report = reduce_logical_weight(code, pauli="Z", logical_basis=custom, threshold=None)
    np.testing.assert_array_equal(report["original_z"], custom["jz"][0])
    actual = np.asarray(report["z"])
    assert in_rowspace(actual ^ custom["jz"][0], code.hz)
    assert not in_rowspace(actual ^ standard.jz[0], code.hz)


@pytest.mark.parametrize("kwargs", [
    dict(method="distance"), dict(max_trials=-1), dict(max_trials=True),
    dict(max_trials=1.5), dict(seed=-1), dict(seed=True),
    dict(threshold=-1), dict(threshold=True), dict(threshold=1.5),
    dict(threshold="2*d"), dict(threshold="20"),
    dict(time_limit=-1), dict(time_limit=np.inf), dict(time_limit=True),
])
def test_invalid_options_are_rejected_even_if_threshold_would_skip(kwargs):
    options = dict(threshold=7)
    options.update(kwargs)
    with pytest.raises(ValueError):
        reduce_logical_weight(steane_code(), "ZZZZZZZ", **options)


@pytest.mark.parametrize("word", ["IIII", "XXXX", "XIII"])
def test_identity_stabilizer_and_noncommuting_target_are_rejected(word):
    with pytest.raises(ValueError):
        reduce_logical_weight(four_qubit_code(), word)


def assert_only_equivalent_target(code, result, original):
    assert result.validate()
    x, z = result.check_x, result.check_z
    assert not np.any((x @ z.T + z @ x.T) % 2)
    cancel = nullspace(np.hstack([x[:, code.n:], z[:, code.n:]]).T)
    image = cancel @ np.hstack([x[:, :code.n], z[:, :code.n]]) % 2
    ox, oz, _ = parse_pauli(original, code.n)
    cx, cz = css_checks(code)
    stabilizers = np.hstack([cx, cz])
    assert rank(np.vstack([stabilizers, image])) == rank(stabilizers) + 1
    assert in_rowspace(image, np.vstack([stabilizers, np.r_[ox, oz]]))
    assert result.k_stabilizer == code.k - 1
    px, pz, phase = pauli_product(x, z, result.check_signs,
                                 np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(px[:code.n], result.measurement_x)
    np.testing.assert_array_equal(pz[:code.n], result.measurement_z)
    assert not np.any(px[code.n:] | pz[code.n:])
    assert phase == 2 * result.measurement_sign
    assert_equivalence(code, original, result.metadata["logical_optimization"])


@pytest.mark.parametrize("builder", [gadget_deform, gauging_deform])
@pytest.mark.parametrize("word", ["ZZZZZZZ", "-YYYYYYY"])
def test_optimized_gadget_is_commuting_and_reads_only_same_logical(builder, word):
    code = steane_code()
    result = builder(code, word, optimize_logical=True, require_distance=False)
    assert int(np.count_nonzero(result.measurement_x | result.measurement_z)) == 3
    assert_only_equivalent_target(code, result, word)


@pytest.mark.parametrize("builder", [gadget_deform, gauging_deform])
def test_optimizer_is_opt_in_and_supplied_options_require_enablement(builder):
    code = steane_code()
    result = builder(code, "ZZZZZZZ", require_distance=False)
    assert int(np.count_nonzero(result.measurement_z)) == 7
    assert "logical_optimization" not in result.metadata
    with pytest.raises(ValueError, match="optimize_logical"):
        builder(code, "ZZZZZZZ", logical_optimization_options={}, require_distance=False)


@pytest.mark.parametrize("builder", [gadget_deform, gauging_deform])
def test_optimized_negative_y_gadget_preserves_original_outcome_convention(builder):
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    result = builder(code, "-YYY", optimize_logical=True,
                     logical_optimization_options=dict(threshold=None), require_distance=False)
    np.testing.assert_array_equal(result.measurement_x, [0, 0, 1])
    np.testing.assert_array_equal(result.measurement_z, [0, 0, 1])
    assert result.measurement_sign == 0
    assert_only_equivalent_target(code, result, "-YYY")


def test_scipy_optional_for_auto_but_required_for_explicit_milp(monkeypatch):
    from codesurgery import logical_optimization

    def unavailable():
        raise ImportError("SciPy intentionally absent")

    monkeypatch.setattr(logical_optimization, "_scipy", unavailable)
    # Rank 19 exceeds auto's small-coset enumeration limit. The last two
    # free qubits remain in this logical product, so the lower bound 1
    # does not match its weight 2 incumbent and the optional backend is tried.
    code = CSSCode(np.zeros((0, 21), dtype=np.uint8), np.eye(21, dtype=np.uint8)[:19])
    word = "Z" * 21
    report = reduce_logical_weight(code, word, max_trials=2, threshold=None)
    assert report["after_weight"] == 2 and not report["optimal"]
    assert report["milp"]["status"] == "unavailable"
    assert_equivalence(code, word, report)
    with pytest.raises(ImportError, match="SciPy"):
        reduce_logical_weight(code, word, method="milp", threshold=21)


def test_milp_timeout_keeps_feasible_coset_witness_without_false_optimum(monkeypatch):
    scipy = pytest.importorskip("scipy.optimize")
    solver = scipy.milp

    def timed_out(**kwargs):
        result = solver(**kwargs)
        assert result.status == 0
        result.status = 1
        result.mip_dual_bound = 999.0
        return result

    monkeypatch.setattr(scipy, "milp", timed_out)
    code = CSSCode(np.zeros((0, 6), dtype=np.uint8), [[1, 1, 1, 1, 0, 0]])
    report = reduce_logical_weight(code, "ZZZIZZ", method="milp", threshold=None)
    assert report["after_weight"] == 3
    assert report["proven_lower_bound"] == 1 and not report["optimal"]
    assert report["milp"]["witness_validated"]
    assert report["milp"]["proof"] is None
    assert_equivalence(code, "ZZZIZZ", report)


@pytest.mark.parametrize("mode", ["zero", "fractional", "nan"])
def test_invalid_solver_incumbent_cannot_replace_original_or_prove_optimality(monkeypatch, mode):
    scipy = pytest.importorskip("scipy.optimize")

    def invalid(**kwargs):
        value = dict(zero=0.0, fractional=0.5, nan=np.nan)[mode]
        return SimpleNamespace(status=0, x=np.full_like(kwargs["c"], value),
                               fun=0.0, message="invalid fixture")

    monkeypatch.setattr(scipy, "milp", invalid)
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    report = reduce_logical_weight(code, "-YYY", method="milp", threshold=None)
    assert report["after_weight"] == 3 and not report["optimal"]
    assert not report["milp"]["witness_validated"]
    assert report["milp"]["proof"] is None
    assert_equivalence(code, "-YYY", report)


@pytest.mark.parametrize("field", ["sign", "hx_coefficients", "original_signature", "after_weight"])
def test_report_verifier_rejects_tampering_despite_cached_true_flags(field):
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    report = reduce_logical_weight(code, "-YYY", threshold=None)
    forged = deepcopy(report)
    if isinstance(forged[field], list):
        forged[field][0] ^= 1
    else:
        forged[field] ^= 1
    basis = code.standard_logicals()
    with pytest.raises(ValueError):
        verify_logical_optimization(code, (report["x"], report["z"], report["sign"]),
                                    dict(jx=basis.jx, jz=basis.jz), forged)


def test_report_survives_full_distance_sparsification_rebuild():
    code = four_qubit_code()
    edges = [(v, (v + 1) % 8) for v in range(8)] + [(0, 4)]
    graph = np.zeros((9, 8), dtype=np.uint8)
    for e, endpoints in enumerate(edges):
        graph[e, list(endpoints)] = 1
    matching = np.zeros((2, 9), dtype=np.uint8)
    matching[0, 8] = 1
    matching[1, [0, 1]] = 1
    # All representatives of this mixed logical have weight three. Keep its
    # specified physical ports while forcing a later cellulation rebuild.
    result = gauging_deform(code, "-YXZI", graph=graph, ports=[0, 2, 4],
                           matching=matching, optimize_logical=True,
                           logical_optimization_options=dict(threshold=3),
                           require_distance=True, distance_options=dict(max_cycle_weight=4))
    assert result.metadata["distance_postprocessing"]["changed"]
    assert result.metadata["distance_postprocessing"]["sparsification"]["added_chord_count"] > 0
    assert result.metadata["logical_optimization"]["status"] == "skipped_threshold"
    assert_only_equivalent_target(code, result, "-YXZI")


@pytest.mark.parametrize("scheme", ["explicit", "gauging"])
def test_module_cli_accepts_optimization_and_weight_threshold(tmp_path, scheme):
    code = steane_code()
    config = dict(hx=code.hx.tolist(), hz=code.hz.tolist(), distance=3,
                  logical="ZZZZZZZ", scheme=scheme, optimize_logical=True,
                  logical_optimization_options=dict(threshold=2), require_distance=False)
    source, output = tmp_path / "input.json", tmp_path / "deformed.json"
    source.write_text(json.dumps(config))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    run = subprocess.run([sys.executable, "-m", "codesurgery.explicit_gadgets",
                          str(source), "-o", str(output)], cwd=root, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    payload = json.loads(output.read_text())
    report = payload["metadata"]["logical_optimization"]
    assert report["before_weight"] == 7 and report["after_weight"] == 3
    assert report["threshold"] == 2
    assert_equivalence(code, "ZZZZZZZ", report)


def padded_four_qubit_code(*, distance=2):
    """Three fixed Z ancillas give equivalent target weights on both sides of 2d."""
    base = four_qubit_code()
    hx = np.pad(base.hx, ((0, 0), (0, 3)))
    hz = np.vstack([np.pad(base.hz, ((0, 0), (0, 3))),
                    np.pad(np.eye(3, dtype=np.uint8), ((0, 0), (4, 0)))])
    return CSSCode(hx, hz, distance=distance)


@pytest.mark.parametrize("weight", [3, 4, 5])
def test_default_threshold_uses_strictly_greater_than_twice_distance(weight):
    code = padded_four_qubit_code()
    word = "ZZII" + "Z" * (weight - 2) + "I" * (5 - weight)
    report = reduce_logical_weight(code, word)
    assert report["threshold"] == 4
    assert report["threshold_rule"] == "2d"
    assert report["threshold_distance"] == 2
    assert report["before_weight"] == weight
    assert report["after_weight"] == (2 if weight > 4 else weight)
    assert (report["status"] == "skipped_threshold") == (weight <= 4)
    assert_equivalence(code, word, report)


@pytest.mark.parametrize("builder", [gadget_deform, gauging_deform])
@pytest.mark.parametrize("weight", [3, 4, 5])
def test_enabled_gadget_default_applies_2d_threshold_and_rechecks_readout(builder, weight):
    code = padded_four_qubit_code()
    word = "ZZII" + "Z" * (weight - 2) + "I" * (5 - weight)
    result = builder(code, word, optimize_logical=True, require_distance=False)
    report = result.metadata["logical_optimization"]
    assert report["threshold"] == 4 and report["threshold_rule"] == "2d"
    assert report["after_weight"] == (2 if weight > 4 else weight)
    assert_only_equivalent_target(code, result, word)


def test_unknown_distance_skips_2d_reduction_and_preserves_signed_target():
    code = padded_four_qubit_code(distance=None)
    word = "-ZZIIZZZ"
    report = reduce_logical_weight(code, word)
    assert report["status"] == "skipped_unknown_distance"
    assert report["threshold"] is None
    assert report["threshold_rule"] == "2d"
    assert report["threshold_distance"] is None
    assert report["before_weight"] == report["after_weight"] == 5
    assert report["sign"] == 1
    assert not np.any(report["hx_coefficients"])
    assert not np.any(report["hz_coefficients"])
    assert_equivalence(code, word, report)


@pytest.mark.parametrize("distance", [None, 2])
@pytest.mark.parametrize("threshold,rule", [(0, "explicit"), (None, "always")])
def test_explicit_threshold_can_optimize_without_distance_or_below_2d(distance, threshold, rule):
    code = padded_four_qubit_code(distance=distance)
    word = "-ZZIIZII"  # Weight 3, below twice the known distance 2.
    report = reduce_logical_weight(code, word, threshold=threshold)
    assert report["threshold"] == threshold and report["threshold_rule"] == rule
    assert report["threshold_distance"] is None
    assert report["before_weight"] == 3 and report["after_weight"] == 2
    assert_equivalence(code, word, report)


def test_bb90_default_threshold_reduces_standard_z1_in_its_fixed_coset():
    code = ibm_bb90_code()
    report = reduce_logical_weight(code, pauli="Z", method="heuristic")
    assert report["threshold"] == 20
    assert report["threshold_rule"] == "2d" and report["threshold_distance"] == 10
    assert report["before_weight"] == 24 and report["after_weight"] == 12
    word = "".join("Z" if bit else "I" for bit in code.standard_logicals().jz[0])
    assert_equivalence(code, word, report)


@pytest.mark.parametrize("scheme", ["explicit", "gauging"])
@pytest.mark.parametrize("threshold_option", [[], ["--logical-weight-threshold", "2d"]])
def test_cli_2d_default_and_named_option(tmp_path, scheme, threshold_option):
    code = padded_four_qubit_code()
    word = "-ZZIIZZI"  # Exactly 2d: CLI must keep this representative.
    config = dict(hx=code.hx.tolist(), hz=code.hz.tolist(), distance=2,
                  logical=word, scheme=scheme, require_distance=False)
    # A CLI 2d argument must override a numeric configuration threshold.
    if threshold_option:
        config["logical_optimization_options"] = dict(threshold=0)
    source, output = tmp_path / "input.json", tmp_path / "deformed.json"
    source.write_text(json.dumps(config))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    run = subprocess.run([sys.executable, "-m", "codesurgery.explicit_gadgets",
                          str(source), "-o", str(output), "--optimize-logical", *threshold_option],
                         cwd=root, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())["metadata"]["logical_optimization"]
    assert report["status"] == "skipped_threshold"
    assert report["threshold"] == 4 and report["threshold_rule"] == "2d"
    assert report["before_weight"] == report["after_weight"] == 4
    assert_equivalence(code, word, report)
