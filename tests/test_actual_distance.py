"""Actual-distance searches, independent witnesses and resource bounds."""
import itertools
import json

import numpy as np
import pytest

from codesurgery.distance_preservation.distance import (code_distance as _code_distance,
    stabilizer_distance as _stabilizer_distance)
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.gf2 import in_rowspace
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli


def code_distance(code, **kwargs):
    # Keep these tests as the original exhaustive-reference regression suite.
    return _code_distance(code, method="enumeration", **kwargs)


def stabilizer_distance(x, z, **kwargs):
    return _stabilizer_distance(x, z, method="enumeration", **kwargs)


def checks(words):
    parsed = [parse_pauli(word, len(words[0])) for word in words]
    return np.array([row[0] for row in parsed]), np.array([row[1] for row in parsed])


def css_checks(code):
    return (np.vstack([code.hx, np.zeros_like(code.hz)]),
            np.vstack([np.zeros_like(code.hx), code.hz]))


def verify_witness(x, z, report):
    witness = report["witness"]
    wx, wz = np.array(witness["x"]), np.array(witness["z"])
    assert not np.any((x @ wz + z @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack([x, z]))
    assert int((wx | wz).sum()) == witness["weight"] == report["upper_bound"]
    px, pz, _ = parse_pauli(witness["pauli"], len(wx))
    assert np.array_equal(wx, px) and np.array_equal(wz, pz)
    json.dumps(report, allow_nan=False)


def independent_distance(x, z):
    """Exhaustively build the stabilizer group, then all Pauli operators."""
    group = set()
    rows = np.hstack([x, z])
    for coeff in itertools.product((0, 1), repeat=len(rows)):
        group.add(tuple((np.array(coeff) @ rows) % 2))
    best = None
    n = x.shape[1]
    for letters in itertools.product((0, 1, 2, 3), repeat=n):
        wx = np.array([letter & 1 for letter in letters])
        wz = np.array([letter >> 1 for letter in letters])
        if tuple(np.r_[wx, wz]) in group or np.any((x @ wz + z @ wx) % 2):
            continue
        weight = int((wx | wz).sum())
        best = weight if best is None else min(weight, best)
    return best


@pytest.mark.parametrize("code,distance", [(four_qubit_code(), 2), (steane_code(), 3)])
def test_known_css_distances(code, distance):
    result = code_distance(code)
    assert result["status"] == "exact" and result["distance"] == distance
    assert result["proven_lower_bound"] == distance
    assert result["method"] == "css_pure_pauli_weight_enumeration"
    verify_witness(*css_checks(code), result)


def test_hamming_fifteen_distance_three_ignores_untrusted_declaration():
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)] for bit in (3, 2, 1, 0)])
    code = CSSCode(h, h, distance=8)
    result = code_distance(code)
    assert result["distance"] == 3
    assert result["k"] == 7
    verify_witness(*css_checks(code), result)


def test_five_qubit_non_css_code():
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    result = stabilizer_distance(x, z)
    assert result["distance"] == 3 == independent_distance(x, z)
    assert result["method"] == "symplectic_pauli_weight_enumeration"
    verify_witness(x, z, result)


@pytest.mark.parametrize("seed", range(5))
def test_local_clifford_mixed_checks_preserve_distance(seed):
    rng = np.random.default_rng(seed)
    x, z = css_checks(steane_code())
    for q in range(x.shape[1]):
        if rng.integers(2):
            x[:, q], z[:, q] = z[:, q].copy(), x[:, q].copy()
        if rng.integers(2):
            z[:, q] ^= x[:, q]
    result = stabilizer_distance(x, z)
    assert result["distance"] == 3
    verify_witness(x, z, result)


def test_signed_y_deformed_code_distance_is_computed_from_actual_checks():
    source = four_qubit_code()
    result = gauging_deform(source, "-YXZI")
    report = code_distance(result)
    assert report["distance"] == 2
    assert report["gauge_qubits"] == 0
    verify_witness(result.check_x, result.check_z, report)


@pytest.mark.parametrize("seed", range(4))
def test_css_and_mixed_match_independent_full_enumeration(seed):
    x, z = css_checks(four_qubit_code())
    rng = np.random.default_rng(seed)
    for _ in range(5):
        a, b = rng.choice(4, size=2, replace=False)
        # Clifford CNOT can change distance; use exhaustive reference.
        x[:, b] ^= x[:, a]
        z[:, a] ^= z[:, b]
        z[:, a] ^= x[:, a]
    report = stabilizer_distance(x, z)
    assert report["distance"] == independent_distance(x, z)
    verify_witness(x, z, report)


def test_full_weight_shells_prove_threshold():
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    result = stabilizer_distance(x, z, max_weight=2)
    assert result["proven_lower_bound"] == 3
    assert result["completed_weight"] == 2
    assert result["checked_candidates"] == 5 * 3 + 10 * 9
    # A linear-algebra witness may additionally make this exact.
    assert result["status"] in ("exact", "bounded")
    verify_witness(x, z, result)


def test_max_weight_limits_without_guessing_distance():
    result = code_distance(steane_code(), max_weight=1)
    assert result["status"] == "bounded"
    assert result["distance"] is None
    assert result["proven_lower_bound"] == 2
    assert result["upper_bound"] == 3
    assert result["completed_weight"] == 1
    assert result["checked_candidates"] == 14
    assert result["stop_reason"] == "weight_limit"


@pytest.mark.parametrize("budget,completed,lower,status", [(0, 0, 1, "unknown"),
    (3, 0, 1, "unknown"), (14, 1, 2, "bounded"), (15, 1, 2, "bounded")])
def test_partial_css_shell_budget_does_not_inflate_lower_bound(budget, completed, lower, status):
    result = code_distance(steane_code(), max_candidates=budget)
    assert result["status"] == status and result["distance"] is None
    assert result["completed_weight"] == completed
    assert result["proven_lower_bound"] == lower
    assert result["checked_candidates"] == budget
    assert result["stop_reason"] == "candidate_budget"
    verify_witness(*css_checks(steane_code()), result)


def test_partial_mixed_shell_budget_does_not_inflate_lower_bound():
    x, z = checks(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    result = stabilizer_distance(x, z, max_candidates=17)
    assert result["proven_lower_bound"] == 2
    assert result["completed_weight"] == 1
    assert result["checked_candidates"] == 17
    assert result["distance"] is None


def test_exact_budget_at_threshold_finishes_successfully():
    result = code_distance(steane_code(), max_weight=1, max_candidates=14)
    assert result["stop_reason"] == "weight_limit"
    assert result["proven_lower_bound"] == 2


def test_zero_max_weight_has_only_trivial_bound_and_witness():
    result = code_distance(steane_code(), max_weight=0)
    assert result["status"] == "bounded"
    assert result["proven_lower_bound"] == 1
    assert result["checked_candidates"] == 0
    verify_witness(*css_checks(steane_code()), result)


def test_stabilizer_state_has_json_safe_infinite_convention():
    x, z = checks(["XX", "ZZ"])
    result = stabilizer_distance(x, z)
    assert result["status"] == "no_logicals"
    assert result["distance"] is None and result["distance_is_infinite"]
    assert result["proven_lower_bound"] is None and result["upper_bound"] is None
    assert result["witness"] is None and result["k"] == 0
    json.dumps(result, allow_nan=False)


def test_no_checks_and_empty_zero_qubit_code():
    result = stabilizer_distance([], np.zeros((0, 3)))
    assert result["distance"] == 1 and result["k"] == 3
    assert result["checked_candidates"] == 0
    assert stabilizer_distance([], [])["status"] == "no_logicals"


def test_redundant_and_zero_generators_do_not_change_distance():
    x, z = css_checks(four_qubit_code())
    x = np.vstack([x, x, [0, 0, 0, 0]])
    z = np.vstack([z, z, [0, 0, 0, 0]])
    result = stabilizer_distance(x, z)
    assert result["distance"] == 2 and result["stabilizer_rank"] == 2
    verify_witness(x, z, result)


@pytest.mark.parametrize("ancilla", ["Z", "Y"])
def test_low_weight_stabilizer_is_not_counted_as_a_logical(ancilla):
    x, z = checks(["XXXXI", "ZZZZI", "IIII" + ancilla])
    result = stabilizer_distance(x, z)
    assert result["distance"] == 2
    assert result["completed_weight"] == 1
    verify_witness(x, z, result)


def test_search_improves_the_initial_linear_algebra_upper_bound():
    x = np.array([[0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1],
                  [1, 1, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0], [0, 1, 0, 1, 1, 1, 0]])
    z = np.array([[0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0]])
    assert stabilizer_distance(x, z, max_weight=0)["upper_bound"] == 3
    result = stabilizer_distance(x, z)
    assert result["distance"] == 2
    assert result["stop_reason"] == "logical_witness_found"
    assert result["completed_weight"] == 1
    assert result["proven_lower_bound"] == 2
    verify_witness(x, z, result)


@pytest.mark.parametrize("x,z,message", [([[2]], [[0]], "0 and 1"),
    ([[0]], [[0, 1]], "same shape"), ([[1], [0]], [[0], [1]], "do not commute"),
    ([1], [0], "two-dimensional")])
def test_invalid_checks_rejected(x, z, message):
    with pytest.raises(ValueError, match=message):
        stabilizer_distance(x, z)


@pytest.mark.parametrize("keyword,value", [("max_weight", -1), ("max_weight", True),
    ("max_weight", 1.5), ("max_candidates", -1), ("max_candidates", False)])
def test_invalid_resource_parameters(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        code_distance(four_qubit_code(), **{keyword: value})


def test_unsupported_wrapper_input():
    with pytest.raises(TypeError, match="expose"):
        code_distance(object())
