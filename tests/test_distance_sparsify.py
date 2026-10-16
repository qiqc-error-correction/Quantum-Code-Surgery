"""Cycle-space, resource and sparsity checks for optional postprocessing."""
import itertools

import numpy as np
import pytest

from codesurgery.distance_preservation.sparsify import sparsify_cycles
from codesurgery.gf2 import rank
from codesurgery.universal_adapters.graphs import cycle_basis, relative_expansion


def graph(n, edges):
    result = np.zeros((len(edges), n), dtype=np.uint8)
    for e, pair in enumerate(edges):
        result[e, list(pair)] = 1
    return result


def check_result(base, result, matching=None):
    b, m, cycles, report = result
    assert np.all(b.sum(axis=1) == 2)
    assert not np.any((cycles @ b) & 1)
    assert rank(cycles) == len(b) - b.shape[1] + 1
    np.testing.assert_array_equal(b[:len(base), :base.shape[1]], base)
    assert not np.any(b[:len(base), base.shape[1]:])
    if matching is not None:
        np.testing.assert_array_equal(m[:, :len(base)], matching)
        assert not np.any(m[:, len(base):])
        np.testing.assert_array_equal(((m @ b) & 1)[:, :base.shape[1]], (matching @ base) & 1)
        assert not np.any(((m @ b) & 1)[:, base.shape[1]:])
    if report["applied"]:
        for key, bound in report["theorem_bounds"].items():
            assert report["after"][key] <= bound
        for layer in range(report["layers"]):
            used = set()
            for entry in report["cycle_assignments"]:
                if entry["layer"] == layer:
                    assert not (used & set(entry["base_vertices"]))
                    used.update(entry["base_vertices"])
    return b, m, cycles, report


@pytest.mark.parametrize("n", [5, 6, 7, 10, 15])
def test_long_cycles_use_zigzag_not_high_degree_fan(n):
    base = graph(n, [(i, (i + 1) % n) for i in range(n)])
    matching = np.eye(n, dtype=np.uint8)
    b, m, cycles, report = check_result(base, sparsify_cycles(base, matching=matching, max_cycle_weight=4), matching)
    assert report["layers"] == 1
    assert report["added_chord_count"] == n - 3
    assert report["triangle_count"] == n - 2
    assert int(b.sum(axis=0).max()) <= 4
    assert np.all(cycles.sum(axis=1) == 3)
    assert int(cycles.sum(axis=0).max()) <= 2


def test_vertex_conflicts_decongest_shared_edges_and_keep_parallel_edge_cycles():
    # Three overlapping fundamental cycles, including a parallel-edge digon.
    base = graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                     (0, 5), (1, 5), (0, 1)])
    matching = np.array([[1, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
    b, m, cycles, report = check_result(base, sparsify_cycles(base, matching=matching, max_cycle_weight=4,
                                                            max_cycle_congestion=1), matching)
    assert report["layers"] >= 2
    assert report["interlayer_square_count"] == len(base) * (report["layers"] - 1)
    assert any(len(c["base_edge_ids"]) == 2 for c in report["cycle_assignments"])
    assert report["after"]["max_cycle_congestion"] <= max(3, int(base.sum(axis=0).max()))
    assert not report["requested_cycle_congestion_met"]


def test_redundant_supplied_generators_are_validated_and_preserved_if_sparse():
    base = graph(3, [(0, 1), (1, 2), (2, 0)])
    supplied = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    result = check_result(base, sparsify_cycles(base, cycles=supplied))
    assert not result[3]["applied"]
    np.testing.assert_array_equal(result[2], supplied)
    with pytest.raises(ValueError, match="complete"):
        sparsify_cycles(base, cycles=[[1, 1, 0]])
    with pytest.raises(ValueError, match="complete"):
        sparsify_cycles(base, cycles=[])


def test_legacy_layering_replaces_nonsimple_generators_by_fundamental_basis():
    base = graph(5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    fundamental = cycle_basis(base)
    # A non-simple sum of touching triangles plus redundant generators.
    supplied = np.vstack([fundamental, fundamental[0] ^ fundamental[1], fundamental])
    result = check_result(base, sparsify_cycles(base, cycles=supplied, max_cycle_congestion=1,
                                              optimize_cycle_basis=False))
    assert result[3]["layers"] == 2
    assert len(result[3]["cycle_assignments"]) == 2


def test_tree_and_isolated_vertex_need_no_cycle_changes_and_return_copies():
    for base in [graph(1, []), graph(4, [(0, 1), (1, 2), (2, 3)])]:
        result = check_result(base, sparsify_cycles(base))
        assert result[1] is None and not result[3]["applied"]
        assert result[0] is not base
        assert result[2].shape == (0, len(base))


def test_layer_zero_expansion_is_retained_for_all_cuts():
    # K4 is already expanding. Trigger layering using a strict congestion goal.
    base = graph(4, list(itertools.combinations(range(4), 2)))
    b, _, _, report = check_result(base, sparsify_cycles(base, max_cycle_congestion=1))
    assert report["layers"] > 1
    original = relative_expansion(base, [0, 1, 2, 3], distance=2)
    final = relative_expansion(b, [0, 1, 2, 3], distance=2)
    assert original["condition_met"] and final["condition_met"]
    assert final["beta"] >= original["beta"]


def test_resource_limits_and_invalid_input_fail_before_large_allocation():
    base = graph(8, [(i, (i + 1) % 8) for i in range(8)])
    with pytest.raises(ValueError, match="max_entries"):
        sparsify_cycles(base, max_cycle_weight=4, max_entries=64)
    many = graph(4, list(itertools.combinations(range(4), 2)))
    with pytest.raises(ValueError, match="max_layers"):
        sparsify_cycles(many, max_cycle_congestion=1, max_layers=1)
    for name, value in [("max_cycle_weight", 3), ("max_cycle_weight", True),
                        ("max_cycle_congestion", 0), ("max_layers", 0)]:
        with pytest.raises(ValueError, match=name):
            sparsify_cycles(base, **{name: value})
    with pytest.raises(ValueError, match="matching"):
        sparsify_cycles(base, matching=[[1, 0]])
    with pytest.raises(ValueError, match="weight"):
        sparsify_cycles([[1, 1, 1, 1]])


@pytest.mark.parametrize("seed", range(8))
def test_small_multigraphs_have_complete_bounded_cycle_generators(seed):
    rng = np.random.default_rng(seed)
    n = 7
    edges = [(v, int(rng.integers(v))) for v in range(1, n)]
    edges.extend(tuple(map(int, rng.choice(n, 2, replace=False))) for _ in range(5))
    base = graph(n, edges)
    result = check_result(base, sparsify_cycles(base, max_cycle_weight=4, max_cycle_congestion=1))
    assert result[3]["after"]["max_cycle_weight"] <= 4
