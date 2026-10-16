"""Pre-branch reductions preserve signed classes and never increase tree size."""
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.models import CSSCode
from codesurgery.parallel_branching import branching_deform
from codesurgery.parallel_branching.targets import prepare_targets, qubitwise_commuting
from codesurgery.symplectic import parse_pauli, pauli_product

# These historical tests exercise the complete binary-tree policy.
branching_deform = partial(branching_deform, branching_mode="always")


def even_code(n=8, *, distance=2):
    checks = np.ones((1, n), dtype=np.uint8)
    return CSSCode(checks, checks, distance=distance)


def assert_signed_source_witnesses(result):
    """Reproduce actual Pauli multiplication; binary signatures alone do not suffice."""
    source = result.source
    for i, witness in enumerate(source["canonicalization_witnesses"]):
        x = np.vstack((source["target_x"][i], source["check_x"]))
        z = np.vstack((source["target_z"][i], source["check_z"]))
        signs = np.r_[source["target_signs"][i], source["check_signs"]]
        rx, rz, phase = pauli_product(x, z, signs, [0, *(np.flatnonzero(witness) + 1)])
        np.testing.assert_array_equal(rx, source["prepared_x"][i])
        np.testing.assert_array_equal(rz, source["prepared_z"][i])
        assert phase == 2 * int(source["prepared_signs"][i])


def test_default_reduces_large_z_support_before_building_both_branches():
    code = even_code()
    targets = ["-ZZZZZZII", "ZZZZZIZI"]
    original_hx, original_hz = code.hx.copy(), code.hz.copy()
    optimized = branching_deform(code, targets=targets, representatives="preserve")
    baseline = branching_deform(code, targets=targets, representatives="preserve",
                                optimize_logical=False)
    assert baseline.n == 22
    assert optimized.n == 14
    assert optimized.k == baseline.k == code.k == 6
    assert optimized.summary()["leaf_weights"] == [2, 2]
    assert len(optimized.logical_basis_x) == 12
    assert not np.any(optimized.leaf_x)
    np.testing.assert_array_equal(optimized.source["target_signs"], [1, 0])
    np.testing.assert_array_equal(optimized.leaf_signs, [1, 0])
    np.testing.assert_array_equal(code.hx, original_hx)
    np.testing.assert_array_equal(code.hz, original_hz)
    assert_signed_source_witnesses(optimized)
    report = optimized.metadata["support_optimization"]
    assert report["before_weights"] == [6, 6]
    assert report["after_weights"] == [2, 2]
    assert report["before_ancillas"] == 14 and report["after_ancillas"] == 6
    assert optimized.validate()["unintended_logical_readout_rank"] == 0


def test_opt_out_preserves_exact_physical_inputs_and_signs():
    targets = ["-ZZZZZZII", "ZZZZZIZI"]
    result = branching_deform(even_code(), targets=targets, representatives="preserve",
                              optimize_logical=False)
    for i, word in enumerate(targets):
        x, z, sign = parse_pauli(word, 8)
        np.testing.assert_array_equal(result.source["prepared_x"][i], x)
        np.testing.assert_array_equal(result.source["prepared_z"][i], z)
        assert result.source["prepared_signs"][i] == sign
    assert not np.any(result.source["canonicalization_witnesses"])
    assert result.metadata["support_optimization"]["enabled"] is False


@pytest.mark.parametrize("sign,expected", [("", 1), ("-", 0)])
def test_mixed_y_reduction_carries_literal_sign_change(sign, expected):
    # In this n=6 code XXXX.. ZZZZ.. = -YYYYYY, hence YYY YII = -IIIIYY.
    result = branching_deform(even_code(6), targets=sign + "YYYYII", force_copy=True,
        logical_optimization_options={"threshold": None})
    assert result.summary()["leaf_weights"] == [2]
    np.testing.assert_array_equal(result.source["prepared_x"], [[0, 0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(result.source["prepared_z"], [[0, 0, 0, 0, 1, 1]])
    assert result.source["prepared_signs"][0] == expected
    assert result.leaf_signs[0] == expected
    assert result.k == 4 and result.n == 10
    assert_signed_source_witnesses(result)
    assert result.validate()["valid"]


def test_independent_reduction_that_breaks_qwc_is_rejected_safely():
    code = even_code()
    # The weight-two representative of XXXXXXII overlaps both sites of Z.
    # Their global commutation does not justify simultaneous branching there.
    result = branching_deform(code, targets=["XXXXXXII", "IIIIIIZZ"],
                              logical_optimization_options={"threshold": None})
    assert result.summary()["leaf_weights"] == [6, 2]
    assert result.n == 18 and result.k == 6
    assert qubitwise_commuting(result.source["prepared_x"], result.source["prepared_z"])
    report = result.metadata["support_optimization"]
    assert report["candidate_weights"] == [2, 2]
    assert report["candidate_ancillas"] is None
    assert report["selection"] == "sequential"
    assert not report["accepted"][0]
    assert_signed_source_witnesses(result)
    assert result.validate()["valid"]


def test_sequential_fallback_retains_independent_safe_improvements():
    # A failed X/Z replacement in the first block must not discard the useful
    # Z reduction on the second, independent block of the same CSS code.
    checks = np.asarray([[1] * 8 + [0] * 8, [0] * 8 + [1] * 8], dtype=np.uint8)
    code = CSSCode(checks, checks, distance=2)
    result = branching_deform(code, targets=["XXXXXXII" + "I" * 8,
        "IIIIIIZZ" + "I" * 8, "I" * 8 + "ZZZZZZII"])
    report = result.metadata["support_optimization"]
    assert report["selection"] == "sequential"
    assert report["after_weights"] == [6, 2, 2]
    assert report["accepted"] == [False, False, True]
    assert report["before_ancillas"] == 27 and report["after_ancillas"] == 19
    assert result.n == 35 and result.k == 12
    assert_signed_source_witnesses(result)
    assert result.validate()["valid"]


def test_lighter_support_with_more_incident_checks_does_not_increase_ancillas():
    # Redundant sparse check rows are retained by branching. The lighter
    # representative IIIZZ touches five rows, whereas ZZZII touches none.
    code = CSSCode(np.tile([0, 0, 0, 1, 1], (5, 1)), [[1, 1, 1, 1, 1]])
    result = branching_deform(code, targets="ZZZII", force_copy=True,
                              logical_optimization_options={"threshold": None})
    report = result.metadata["support_optimization"]
    assert report["before_weights"] == [3]
    assert report["candidate_weights"] == [2]
    assert report["after_weights"] == [3]
    assert report["before_ancillas"] == 3
    assert report["candidate_ancillas"] == 7
    assert report["after_ancillas"] == 3
    assert not report["accepted"][0]
    assert result.n == 8 and result.k == 3
    assert len(result.branches[0]["incident_rows"]) == 0
    assert_signed_source_witnesses(result)


@pytest.mark.parametrize("distance", [None, 2])
def test_2d_threshold_and_unknown_distance_skip_until_explicit_override(distance):
    # Weight four can be reduced to two, but it is exactly 2d, not above it.
    code = CSSCode([[1, 1, 1, 1, 0, 0, 0]],
                  [[1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], distance=distance)
    default = branching_deform(code, targets="-ZZIIZZI", force_copy=True)
    always = branching_deform(code, targets="-ZZIIZZI", force_copy=True,
                              logical_optimization_options={"threshold": None})
    assert default.summary()["leaf_weights"] == [4]
    assert always.summary()["leaf_weights"] == [2]
    assert default.n == 12 and always.n == 10
    assert_signed_source_witnesses(always)


def test_single_target_without_copy_reduces_representative_but_adds_no_qubits():
    result = branching_deform(even_code(), targets="-ZZZZZZII")
    assert result.n == 8 and result.k == 6 and not result.branches
    assert result.summary()["leaf_weights"] == [2]
    assert result.metadata["support_optimization"]["after_ancillas"] == 0
    assert_signed_source_witnesses(result)


@pytest.mark.parametrize("leaf", ["auto", "X", "Z"])
@pytest.mark.parametrize("targets", [
    ["ZI", "IZ", "ZZ", "ZI", "-IZ"],
    ["YI", "IX", "YX", "-YI", "-IX", "YX", "YI", "IX", "YX"],
])
def test_cheap_tree_cost_matches_actual_recursive_construction(leaf, targets):
    from codesurgery.parallel_branching.optimization import branching_ancilla_count

    code = four_qubit_code()
    p = prepare_targets(code, logical_paulis=targets)
    predicted = branching_ancilla_count(p["source_check_x"], p["source_check_z"],
                                       p["target_x"], p["target_z"])
    result = branching_deform(code, logical_paulis=targets, leaf_pauli=leaf,
                              optimize_logical=False)
    assert predicted == result.n - code.n
    assert predicted == sum(len(b["support"]) + len(b["incident_rows"])
                            for b in result.branches)
    assert result.summary()["tree_depth"] >= 3


def test_saved_preparation_witness_is_checked_after_optimization():
    result = branching_deform(even_code(), targets="-ZZZZZZII", force_copy=True)
    damaged = deepcopy(result)
    damaged.source["prepared_signs"][0] ^= 1
    with pytest.raises(ValueError, match="Signed logical-equivalence"):
        damaged.validate()


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_non_boolean_optimization_switch_is_rejected(bad):
    with pytest.raises(ValueError, match="optimize_logical"):
        branching_deform(even_code(), targets="ZZZZZZII", optimize_logical=bad)
