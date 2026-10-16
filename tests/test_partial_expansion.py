"""Certified individual-edge fallback when full cycle overlays do not fit."""
import itertools

import numpy as np
import pytest

from codesurgery.distance_preservation.expansion import (
    ExpansionConstructionError, certify_expansion, enhance_expansion as _enhance_expansion,
)


def enhance_expansion(*args, **kwargs):
    """These regression cases intentionally cover the retained legacy strategy."""
    return _enhance_expansion(*args, strategy="hamiltonian", **kwargs)


def graph(n, edges):
    result = np.zeros((len(edges), n), dtype=np.uint8)
    for row, endpoints in enumerate(edges):
        result[row, list(endpoints)] = 1
    return result


def thin_y_graph():
    return graph(5, [(0, 2), (0, 1), (0, 3), (3, 4)])


def exact_beta(b, ports, distance):
    best = float("inf")
    for membership in itertools.product((0, 1), repeat=b.shape[1]):
        mask = np.array(membership)
        p = sum(membership[v] for v in ports)
        denominator = min(distance, p, len(ports) - p)
        if denominator:
            boundary = int(((b @ mask) % 2).sum())
            best = min(best, boundary / denominator)
    return best


def assert_verified(b, output, report, max_degree, ports, distance):
    assert np.array_equal(output[:len(b)], b)
    assert np.all(output.sum(axis=1) == 2)
    assert int(output.sum(axis=0).max()) <= max_degree
    assert report["condition_met"] and report["exact"]
    assert exact_beta(output, ports, distance) >= 1
    assert certify_expansion(output, ports=ports, distance=distance)["condition_met"]
    assert report["added_edge_count"] == len(output) - len(b)
    assert report["added_edges"] == [list(np.flatnonzero(row)) for row in output[len(b):]]
    assert report["final_max_degree"] == int(output.sum(axis=0).max())
    assert report["augmentation_strategy"] == "exact_worst_cut_partial_overlay"
    assert "not a prescribed paper algorithm" in report["implementation_choice"]
    assert report["overlay_layers"] == 0


@pytest.mark.parametrize("degree", [3, 4])
def test_five_vertex_y_adapter_repaired_with_individual_spare_degrees(degree):
    original = thin_y_graph()
    assert exact_beta(original, range(5), 2) == 0.5
    output, report = enhance_expansion(original, ports=range(5), distance=2,
                                     max_degree=degree, allow_partial_overlay=True)
    assert_verified(original, output, report, degree, range(5), 2)
    assert report["added_edge_count"] == 1
    assert report["partial_search_attempts"] == 1
    assert len(report["partial_edge_steps"]) == 1
    step = report["partial_edge_steps"][0]
    assert step["previous_boundary"] < step["denominator"]
    assert len(set(step["added_edge"]) & set(step["worst_cut"])) == 1


def test_disabled_option_preserves_old_failure_behavior():
    for options in ({}, {"allow_partial_overlay": False}):
        with pytest.raises(ExpansionConstructionError, match="cycle overlay") as failure:
            enhance_expansion(thin_y_graph(), distance=2, max_degree=4, **options)
        assert failure.value.reason == "degree_budget"


def test_full_hamiltonian_search_failure_can_fall_back_to_individual_edges():
    original = graph(8, [(v, v + 1) for v in range(7)])
    options = dict(max_degree=4, attempts=1, seed=10)
    with pytest.raises(ExpansionConstructionError) as failure:
        enhance_expansion(original, **options)
    assert failure.value.reason == "search_budget"
    output, report = enhance_expansion(original, allow_partial_overlay=True, **options)
    assert_verified(original, output, report, 4, range(8), 8)
    assert report["attempts_used"] > report["added_edge_count"]


def test_saturated_low_expansion_graph_fails_closed():
    cycle = graph(6, [(v, (v + 1) % 6) for v in range(6)])
    assert exact_beta(cycle, range(6), 6) < 1
    with pytest.raises(ExpansionConstructionError, match="degree and attempt budgets") as failure:
        enhance_expansion(cycle, max_degree=2, allow_partial_overlay=True)
    assert failure.value.reason == "degree_budget"
    assert not failure.value.details["last_certificate"]["condition_met"]


def test_parallel_original_edges_preserved_byte_for_byte():
    original = thin_y_graph()
    original = np.vstack([original, original[0]])
    output, report = enhance_expansion(original, distance=2, max_degree=5,
                                     allow_partial_overlay=True)
    assert_verified(original, output, report, 5, range(5), 2)
    assert np.array_equal(output[0], output[4])


def test_seeded_partial_fallback_is_deterministic():
    outputs = set()
    for seed in range(6):
        first, first_report = enhance_expansion(thin_y_graph(), distance=2, max_degree=4,
                                               seed=seed, allow_partial_overlay=True)
        second, second_report = enhance_expansion(thin_y_graph(), distance=2, max_degree=4,
                                                 seed=seed, allow_partial_overlay=True)
        assert np.array_equal(first, second)
        assert first_report == second_report
        outputs.add(first.tobytes())
    assert len(outputs) > 1


@pytest.mark.parametrize("limits", [{"exact_limit": 4}, {"max_entries": 100}])
def test_partial_fallback_requires_an_exact_cut_budget(limits):
    with pytest.raises(ExpansionConstructionError, match="exact worst-cut certificate") as failure:
        enhance_expansion(thin_y_graph(), distance=2, max_degree=4,
                          allow_partial_overlay=True, **limits)
    assert failure.value.reason == "certificate_budget"


def test_enabling_fallback_keeps_successful_full_cycle_choice():
    original = graph(6, [(v, v + 1) for v in range(5)])
    plain, plain_report = enhance_expansion(original, seed=5, max_degree=8)
    enabled, enabled_report = enhance_expansion(original, seed=5, max_degree=8,
                                               allow_partial_overlay=True)
    assert np.array_equal(plain, enabled)
    assert plain_report == enabled_report
    assert enabled_report["augmentation_strategy"] == "hamiltonian_cycle_overlays"


def test_preexisting_certificate_needs_no_augmentation():
    original = thin_y_graph()
    output, report = enhance_expansion(original, ports=[1, 4], distance=2,
                                     max_degree=4, allow_partial_overlay=True)
    assert np.array_equal(output, original)
    assert report["condition_met"] and report["added_edge_count"] == 0


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_partial_overlay_flag_must_be_boolean(value):
    with pytest.raises(ValueError, match="allow_partial_overlay"):
        enhance_expansion(graph(2, [(0, 1)]), allow_partial_overlay=value)
