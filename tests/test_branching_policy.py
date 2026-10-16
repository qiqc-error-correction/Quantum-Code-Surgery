"""Adaptive routes budget the actual cumulative original-code connectivity."""
import numpy as np
import pytest

from codesurgery.parallel_branching.construction import append_branch, lift_paulis
from codesurgery.parallel_branching.direct_attachment import preview_direct_attachment
from codesurgery.parallel_branching.optimization import branching_ancilla_count
from codesurgery.parallel_branching.policy import (
    BranchingBudgetError, evaluate_routing, plan_branching,
)
from codesurgery.symplectic import parse_pauli


def words(items, n=None):
    if not items:
        return np.zeros((0, n), np.uint8), np.zeros((0, n), np.uint8)
    n = len(items[0].lstrip("+-"))
    rows = [parse_pauli(item, n) for item in items]
    return tuple(np.asarray([r[j] for r in rows], np.uint8) for j in (0, 1))


def arrays(checks, targets):
    tx, tz = words(targets)
    cx, cz = words(checks, tx.shape[1])
    return cx, cz, tx, tz


def route(checks, targets, **kwargs):
    return plan_branching(*arrays(checks, targets), **kwargs)


def test_two_overlapping_targets_stay_direct_when_aggregate_fits():
    report = route(["XXXX", "ZZZZ"], ["ZZII", "ZIZI"])
    assert report["root_groups"] == []
    assert report["direct_targets"] == [0, 1]
    assert report["ancillas"] == 0
    assert report["projected_original_row_growth"] == [2, 0]
    assert report["projected_original_column_growth"] == [2, 1, 1, 0]


def test_single_weight_six_boundary_branches_without_infinite_aux_recursion():
    report = route(["XXXXXX"], ["ZZZZZZ"])
    assert report["root_groups"] == [[0]]
    assert report["ancillas"] == 7
    assert report["direct_all_row_growth"] == [3]
    assert report["projected_original_row_growth"] == [1]
    assert not report["auxiliary_limits_enforced"]


def test_individually_legal_targets_need_reconsidering_together():
    report = route(["XXXXXXXX"], ["ZZZZIIII", "ZZIIZZII"])
    assert report["direct_all_row_growth"] == [4]
    assert report["branched_targets"] == [0, 1]
    assert report["direct_targets"] == []
    assert report["root_groups"] == [[0], [1]]
    assert report["projected_original_row_growth"] == [2]


def test_shared_root_reduces_memory_connections():
    report = route(["XXXX", "ZZZZ"], ["ZZII"] * 3)
    assert report["original_limits_satisfied"]
    assert len(report["branched_targets"]) == 2
    assert len(report["direct_targets"]) == 1
    assert len(report["root_groups"]) == 1
    assert report["projected_original_column_growth"][:2] == [2, 2]
    assert report["ancillas"] == 9


def test_per_row_baseline_not_global_maximum():
    report = route(["XXIIIIII", "IIXXXXXX"], ["ZZIIIIII"] * 3)
    assert report["original_row_weights"] == [2, 6]
    assert report["direct_all_row_growth"] == [3, 0]
    assert report["root_groups"]
    assert report["projected_original_row_weights"] == [4, 6]


def test_column_budget_triggers_when_all_boundaries_are_empty():
    # No stabilizers: direct row loads vanish, but the same physical qubit
    # would participate in three independent gadget vertex checks.
    report = route([], ["ZI", "ZI", "ZI"])
    assert report["direct_all_row_growth"] == []
    assert report["direct_all_column_growth"] == [3, 0]
    assert report["root_groups"]
    assert report["projected_original_column_growth"] == [2, 0]


def test_mixed_y_support_counts_once_and_normalizer_checked():
    report = route(["XXXX", "ZZZZ"], ["ZXYI", "ZXIY"])
    assert report["ancillas"] == 0
    assert report["projected_original_row_growth"] == [2, 2]
    assert report["projected_original_column_growth"] == [2, 2, 1, 1]


def test_always_mode_preserves_balanced_tree_and_reports_violations():
    cx, cz, tx, tz = arrays(["XXXXXXXX"], ["ZZIIIIII", "ZIZIIIII", "ZIIZIIII", "ZIIIZIII"])
    report = plan_branching(cx, cz, tx, tz, branching_mode="always", max_check_increase=0)
    assert report["root_groups"] == [[0, 1], [2, 3]]
    assert report["ancillas"] == branching_ancilla_count(cx, cz, tx, tz)
    assert not report["original_limits_satisfied"]
    assert report["selection_method"] == "legacy_balanced"


def test_single_legacy_does_not_branch_without_force_even_if_budget_exceeded():
    report = route(["XXXXXX"], ["ZZZZZZ"], branching_mode="always")
    assert report["ancillas"] == 0
    assert not report["original_limits_satisfied"]


@pytest.mark.parametrize("mode", ["adaptive", "always"])
def test_single_force_copy_and_multi_force_has_no_extra_effect(mode):
    forced = route(["XXXX", "ZZZZ"], ["ZZII"], branching_mode=mode, force_copy=True)
    assert forced["root_groups"] == [[0]]
    assert forced["ancillas"] == 3
    normal = route(["XXXX", "ZZZZ"], ["ZZII", "ZIZI"], branching_mode=mode)
    copied = route(["XXXX", "ZZZZ"], ["ZZII", "ZIZI"], branching_mode=mode, force_copy=True)
    assert normal == copied


@pytest.mark.parametrize("key,value", [("max_check_increase", True),
    ("max_qubit_increase", False), ("max_check_increase", -1),
    ("max_qubit_increase", 1.5), ("max_qubit_increase", "2"),
    ("force_copy", 1), ("branching_mode", "sometimes")])
def test_invalid_policy_options_are_not_budget_failures(key, value):
    with pytest.raises(ValueError) as exc:
        route(["XXXX", "ZZZZ"], ["ZZII"], **{key: value})
    assert not isinstance(exc.value, BranchingBudgetError)


@pytest.mark.parametrize("kwargs", [dict(max_check_increase=0), dict(max_qubit_increase=0)])
def test_zero_connectivity_budget_is_explicitly_infeasible(kwargs):
    with pytest.raises(BranchingBudgetError, match="original per-check/per-qubit"):
        route(["XXXX", "ZZZZ"], ["ZZII"], **kwargs)


def test_zero_row_budget_can_work_when_no_check_has_a_boundary():
    report = route([], ["ZI"] * 3, max_check_increase=0)
    assert report["original_limits_satisfied"]
    assert report["projected_original_row_growth"] == []


@pytest.mark.parametrize("groups", [[[0, 0]], [[0], [0]], [[]], [[2]], [[True]], [[-1]], [1]])
def test_fixed_route_rejects_bad_root_groups(groups):
    with pytest.raises(ValueError):
        evaluate_routing(*arrays(["XXXX", "ZZZZ"], ["ZZII", "ZIZI"]), groups)


def test_large_target_set_has_a_bounded_feasible_deterministic_fallback():
    first = route(["XXXX", "ZZZZ"], ["ZZII"] * 13)
    second = route(["XXXX", "ZZZZ"], ["ZZII"] * 13)
    assert first == second
    assert first["selection_method"] == "bounded_greedy"
    assert not first["optimal_within_candidate_family"]
    assert first["original_limits_satisfied"]
    assert first["candidates_evaluated"] < 1000
    assert len(first["direct_targets"]) == 1


def _build_roots(cx, cz, tx, tz, groups):
    cs, ts = np.zeros(len(cx), np.uint8), np.zeros(len(tx), np.uint8)

    def add(group):
        nonlocal cx, cz, cs, tx, tz, ts
        px, pz = np.bitwise_or.reduce(tx[group], axis=0), np.bitwise_or.reduce(tz[group], axis=0)
        branch = append_branch(cx, cz, cs, px, pz, leaf_pauli="Z")
        mask = (tx[group][:, branch["support"]] | tz[group][:, branch["support"]]).copy()
        tx, tz, ts = lift_paulis(tx, tz, ts, branch)
        tx[group], tz[group] = 0, 0
        tz[np.ix_(group, branch["leaf_columns"])] = mask
        cx, cz, cs = branch["check_x"], branch["check_z"], branch["check_signs"]

    def split(group):
        mid = len(group) // 2
        for child in (group[:mid], group[mid:]):
            add(child)
            if len(child) > 1:
                split(child)

    for group in groups:
        add(group)
        if len(group) > 1:
            split(group)
    return cx, cz, cs, tx, tz, ts


@pytest.mark.parametrize("groups", [[], [[0]], [[0], [1]], [[0, 1]], [[0, 1], [2, 3]], [[0, 1, 2, 3]]])
def test_fixed_route_forecast_matches_actual_branching_and_measurement_matrices(groups):
    cx, cz, tx, tz = arrays(["XXXXXXXX"], ["ZZIIIIII", "ZIZIIIII", "ZIIZIIII", "ZIIIZIII"])
    expected = evaluate_routing(cx, cz, tx, tz, groups)
    outx, outz, signs, lx, lz, ls = _build_roots(cx.copy(), cz.copy(), tx.copy(), tz.copy(), groups)
    assert outx.shape[1] - cx.shape[1] == expected["ancillas"]
    attached = preview_direct_attachment(outx, outz, lx, lz, check_signs=signs, target_signs=ls)
    original = cx | cz
    final = attached["check_x"] | attached["check_z"]
    np.testing.assert_array_equal(final.sum(axis=1)[:len(cx)] - original.sum(axis=1),
                                  expected["projected_original_row_growth"])
    np.testing.assert_array_equal(final.sum(axis=0)[:cx.shape[1]] - original.sum(axis=0),
                                  expected["projected_original_column_growth"])


def test_non_qwc_commuting_inputs_are_rejected():
    with pytest.raises(ValueError, match="qubit-wise"):
        route(["XXXXXXXX", "ZZZZZZZZ"], ["XXXXIIII", "ZZIIZZII"])

