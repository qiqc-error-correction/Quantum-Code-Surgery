"""Behavioral checks for seeded, one-degree-at-a-time graph augmentation.

Cut values below use an independent set-boundary oracle.  They do not call
production expansion helpers or trust a reported floating-point h value.
"""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from codesurgery.distance_preservation.expansion import (
    ExpansionConstructionError,
    enhance_expansion,
)


def cycle(n):
    matrix = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        matrix[i, [i, (i + 1) % n]] = 1
    return matrix


def exact_global_h(matrix):
    """Enumerate vertex subsets directly, keeping parallel edges distinct."""
    n = matrix.shape[1]
    edges = [set(np.flatnonzero(row)) for row in matrix]
    return min(
        Fraction(sum(len(edge & set(subset)) == 1 for edge in edges), size)
        for size in range(1, n // 2 + 1)
        for subset in combinations(range(n), size)
    )


def assert_extension(original, result, cap):
    assert np.array_equal(result[:len(original)], original)
    assert np.all(result.sum(axis=1) == 2)
    assert int(result.sum(axis=0).max()) <= cap


def test_default_attempts_degree_three_before_four_and_distance_can_stop_early():
    original = cycle(8)
    before = original.copy()
    seen = []

    def sufficient_distance(matrix, context):
        seen.append((matrix.copy(), dict(context)))
        return {"distance_state": "adequate", "eligible": True,
                "distance_report": {"proven_lower_bound": 10}}

    result, report = enhance_expansion(
        original, max_degree=4, seed=23, candidate_evaluator=sufficient_distance,
    )
    assert len(seen) == 1
    assert seen[0][1]["stage_degree"] == 3
    assert np.all(result.sum(axis=0) == 3)
    assert_extension(original, result, 4)
    assert np.array_equal(original, before)
    assert report["acceptance"] == "distance"
    assert report["expansion_evaluated"] is False
    assert report["condition_met"] is False  # A distance proof is not an h proof.
    assert report["regular"] is True


def test_odd_vertex_count_is_reported_as_near_regular_not_three_regular():
    original = cycle(7)
    result, report = enhance_expansion(
        original, max_degree=3, seed=11,
        candidate_evaluator=lambda matrix, context: {
            "distance_state": "adequate", "eligible": True,
        },
    )
    degrees = result.sum(axis=0)
    assert sorted(degrees.tolist()) == [2, 3, 3, 3, 3, 3, 3]
    assert report["regular"] is False
    assert report["final_degrees"] == degrees.tolist()
    assert_extension(original, result, 3)


@pytest.mark.parametrize("n", [6, 7, 8])
def test_expansion_acceptance_requires_strict_global_h(n):
    original = cycle(n)
    assert exact_global_h(original) <= 1
    result, report = enhance_expansion(original, max_degree=5, attempts=8, seed=19)
    assert_extension(original, result, 5)
    assert report["acceptance"] == "expansion"
    assert report["strict_global_h"] is True
    assert report["expansion_evaluated"] is True
    assert exact_global_h(result) > 1


def test_unknown_distance_does_not_grant_a_distance_certificate():
    seen = []

    def unknown(matrix, context):
        seen.append(dict(context))
        return {"distance_state": "unknown", "eligible": True,
                "distance_report": {"upper_bound": 10, "proven_lower_bound": None}}

    result, report = enhance_expansion(cycle(6), max_degree=5, seed=6,
                                      candidate_evaluator=unknown)
    assert seen
    assert report["acceptance"] == "expansion"
    assert report["expansion_evaluated"] is True
    assert exact_global_h(result) > 1


def test_distance_deficit_can_trigger_h_checks_and_later_candidates():
    seen = []

    def deficit_then_unknown(matrix, context):
        seen.append(matrix.copy())
        return {"distance_state": "insufficient" if len(seen) == 1 else "unknown",
                "eligible": True}

    result, report = enhance_expansion(cycle(6), max_degree=5, seed=7,
                                      candidate_evaluator=deficit_then_unknown)
    assert len(seen) > 1
    assert exact_global_h(seen[0]) <= 1
    assert report["acceptance"] == "expansion"
    assert exact_global_h(result) > 1


def test_distance_counterexample_contradicting_expansion_fails_closed():
    seen = []

    def impossible_counterexample(matrix, context):
        seen.append(matrix.copy())
        return {"distance_state": "insufficient", "eligible": True,
                "distance_report": {"upper_bound": 1}}

    with pytest.raises(ExpansionConstructionError) as failure:
        enhance_expansion(cycle(6), max_degree=5, seed=7,
                          candidate_evaluator=impossible_counterexample)
    assert failure.value.reason == "distance_certificate_conflict"
    assert exact_global_h(seen[-1]) > 1


def test_ineligible_code_candidate_is_retried_even_if_distance_was_adequate():
    seen = []

    def reject_then_accept(matrix, context):
        seen.append(dict(context))
        return {"distance_state": "adequate", "eligible": len(seen) > 1,
                "reason": "full check row exceeds its fixed cap" if len(seen) == 1 else None}

    result, report = enhance_expansion(
        cycle(8), max_degree=4, attempts=3, seed=13,
        candidate_evaluator=reject_then_accept,
    )
    assert len(seen) == 2
    assert [item["stage_degree"] for item in seen] == [3, 3]
    assert report["acceptance"] == "distance"
    assert_extension(cycle(8), result, 4)


def test_every_ineligible_candidate_fails_closed_under_fixed_degree_cap():
    seen = []

    def ineligible(matrix, context):
        seen.append((int(matrix.sum(axis=0).max()), dict(context)))
        return {"distance_state": "adequate", "eligible": False,
                "reason": "full check row exceeds its fixed cap"}

    with pytest.raises(ExpansionConstructionError):
        enhance_expansion(cycle(6), max_degree=4, attempts=2, seed=1,
                          candidate_evaluator=ineligible)
    assert len(seen) >= 2
    assert all(degree <= 4 for degree, context in seen)
    stages = [context["stage_degree"] for degree, context in seen]
    assert stages == sorted(stages)
    assert stages[0] == 3


def test_seed_and_retry_log_reproduce_the_same_search_without_mutating_input():
    original = cycle(8)
    before = original.copy()
    runs = []
    for _ in range(2):
        calls = []

        def retry_once(matrix, context):
            calls.append((matrix.copy(), dict(context)))
            return {"distance_state": "adequate", "eligible": len(calls) >= 2}

        output, report = enhance_expansion(original, max_degree=4, attempts=3,
                                          seed=100, candidate_evaluator=retry_once)
        runs.append((output, report, calls))
    assert np.array_equal(original, before)
    assert np.array_equal(runs[0][0], runs[1][0])
    assert runs[0][1] == runs[1][1]
    assert runs[0][1]["search_history"]
    assert runs[0][1]["seed"] == 100
    assert len(runs[0][2]) == len(runs[1][2]) == 2
    assert runs[0][2][0][1]["candidate_seed"] != runs[0][2][1][1]["candidate_seed"]
    for (first, first_context), (second, second_context) in zip(runs[0][2], runs[1][2]):
        assert np.array_equal(first, second)
        assert first_context == second_context


def test_an_exhausted_degree_budget_never_silently_increases_the_cap():
    original = cycle(8)
    before = original.copy()
    with pytest.raises(ExpansionConstructionError) as failure:
        enhance_expansion(original, max_degree=2, attempts=2, seed=0)
    assert failure.value.reason == "degree_budget"
    assert np.array_equal(original, before)


def test_input_already_over_cap_is_rejected_before_any_evaluation():
    calls = []
    with pytest.raises(ExpansionConstructionError, match="exceeds max_degree"):
        enhance_expansion(cycle(6), max_degree=1,
                          candidate_evaluator=lambda matrix, context: calls.append(context))
    assert not calls


def test_distance_acceptance_does_not_require_an_available_expansion_budget():
    # h cannot be enumerated or spectrally certified at these limits.  A valid
    # code-distance result must still be usable, since it is checked first.
    result, report = enhance_expansion(
        cycle(8), max_degree=3, exact_limit=1, certificate_limit=1,
        candidate_evaluator=lambda matrix, context: {
            "distance_state": "adequate", "eligible": True,
            "distance_report": {"proven_lower_bound": 10},
        },
    )
    assert np.all(result.sum(axis=0) == 3)
    assert report["acceptance"] == "distance"
    assert report["expansion_evaluated"] is False


def test_worst_cut_repair_keeps_a_failed_seed_and_uses_only_needed_degree_slots():
    original = cycle(8)
    options = dict(max_degree=4, attempts=12, seed=0, allow_partial_overlay=True)
    result, report = enhance_expansion(original, **options)
    assert report["augmentation_strategy"] == "degree_incremental_worst_cut_repair"
    assert report["acceptance"] == "expansion"
    assert exact_global_h(result) > 1
    assert_extension(original, result, 4)
    # Local repair succeeds without making every vertex degree four.
    assert result.sum(axis=0).min() == 3
    assert result.sum(axis=0).max() == 4
    assert len(result) - len(original) < original.shape[1]
    history = report["search_history"]
    matching_runs = [row for row in history if row["stage"] == "random_matching"]
    repair_runs = [row for row in history if row["stage"] == "worst_cut_repair"]
    assert len(matching_runs) == 12
    assert all(row["stage_degree"] == 3 for row in matching_runs)
    assert repair_runs
    selected = next(row for row in matching_runs
                    if row["candidate_seed"] == report["candidate_seed"])
    assert all(row["candidate_seed"] == selected["candidate_seed"] for row in repair_runs)
    assert all(row["stage_degree"] == 4 for row in repair_runs)
    assert report["added_edges"][:len(selected["added_edges"])] == selected["added_edges"]
    # Verify the exact violating cut immediately preceding each local edge.
    for repair in repair_runs:
        step = repair["targeted_steps"][-1]
        inside = set(step["worst_cut"])
        assert len(set(step["added_edge"]) & inside) == 1
        before_edges = ([list(np.flatnonzero(row)) for row in original]
                        + repair["added_edges"][:-1])
        boundary = sum(len(set(edge) & inside) == 1 for edge in before_edges)
        assert boundary == step["previous_boundary"]
        assert step["denominator"] == min(len(inside), original.shape[1] - len(inside))
        assert boundary <= step["denominator"]
    repeated, repeated_report = enhance_expansion(original, **options)
    assert np.array_equal(repeated, result)
    assert repeated_report == report
