"""Independent cut and spectral checks for bounded-degree augmentation."""
from itertools import combinations

import numpy as np
import pytest

from codesurgery.distance_preservation.expansion import (
    ExpansionConstructionError,
    certify_expansion,
    enhance_expansion as _enhance_expansion,
)


def enhance_expansion(*args, **kwargs):
    """These regression cases intentionally cover the retained legacy strategy."""
    return _enhance_expansion(*args, strategy="hamiltonian", **kwargs)


def incidence(n, edges):
    b = np.zeros((len(edges), n), dtype=np.uint8)
    for e, (u, v) in enumerate(edges):
        b[e, [u, v]] = 1
    return b


def path(n):
    return incidence(n, [(v, v + 1) for v in range(n - 1)])


def independent_beta(b, ports, d):
    # Enumerate all bit strings (including both cut orientations), deliberately
    # not using the production subset-mask or boundary-enumeration helper.
    values = []
    for subset in range(1, 2 ** b.shape[1] - 1):
        inside = {v for v in range(b.shape[1]) if subset & (1 << v)}
        p = len(inside & set(ports))
        den = min(d, p, len(ports) - p)
        if den:
            boundary = sum(len(set(np.flatnonzero(row)) & inside) == 1 for row in b)
            values.append(boundary / den)
    return min(values) if values else float("inf")


@pytest.mark.parametrize("n", [4, 5, 8, 10])
def test_cycle_overlays_certified_preserved_and_degree_bounded(n):
    original = path(n)
    augmented, report = enhance_expansion(original, max_degree=8, seed=5)
    assert report["condition_met"]
    assert np.array_equal(augmented[:len(original)], original)
    assert np.max(augmented.sum(axis=0)) <= 8
    assert report["added_edge_count"] == len(augmented) - len(original)
    assert report["final_max_degree"] == int(augmented.sum(axis=0).max())
    assert independent_beta(augmented, range(n), n) >= 1
    repeat, repeat_report = enhance_expansion(original, max_degree=8, seed=5)
    assert np.array_equal(augmented, repeat)
    assert report == repeat_report
    assert not np.array_equal(augmented, original)


def test_relative_ports_can_avoid_unnecessary_expansion():
    b = path(8)
    augmented, report = enhance_expansion(b, ports=[0, 7], distance=2)
    assert np.array_equal(augmented, b)
    assert report["condition_met"] and report["added_edge_count"] == 0
    assert report["global_h_certified"] is False
    assert independent_beta(b, range(8), 8) < 1
    assert independent_beta(b, [0, 7], 2) == 1


def test_exact_integer_spectral_proof_and_counterexamples():
    complete = incidence(6, list(combinations(range(6), 2)))
    proved = certify_expansion(complete, exact_limit=1)
    assert proved["condition_met"]
    assert proved["spectral_proof"]["proved"]
    assert proved["global_h_lower_bound"] == 1
    assert proved["beta"] is None and not proved["exact"]
    assert independent_beta(complete, range(6), 6) >= 1
    for n in [4, 5, 8]:
        rejected = certify_expansion(path(n), exact_limit=1)
        assert not rejected["condition_met"]
        assert not rejected["spectral_proof"]["proved"]


def test_random_spectral_success_implies_independent_exact_expansion():
    rng = np.random.default_rng(32)
    successes = 0
    for n in range(4, 9):
        for count in range(1, 4):
            edges = [(v, v + 1) for v in range(n - 1)]
            for _ in range(count):
                order = rng.permutation(n)
                edges.extend((int(order[v]), int(order[(v + 1) % n])) for v in range(n))
            b = incidence(n, edges)
            report = certify_expansion(b, exact_limit=1)
            if report["condition_met"]:
                successes += 1
                assert independent_beta(b, range(n), n) >= 1
                # Independent float eigenspectrum is a test oracle only; it
                # is never used to grant a production certificate.
                adjacency = np.zeros((n, n), dtype=int)
                for u, v in edges:
                    adjacency[u, v] += 1
                    adjacency[v, u] += 1
                lap = np.diag(adjacency.sum(axis=0)) - adjacency
                assert np.linalg.eigvalsh(lap)[1] > 2 - 1e-10
    assert successes >= 8


def test_spectral_augmentation_above_exact_enumeration_limit():
    b = path(40)
    out, report = enhance_expansion(b, exact_limit=4, max_degree=12,
                                    attempts=3, seed=15)
    assert report["condition_met"]
    assert report["spectral_proof"]["proved"]
    assert int(out.sum(axis=0).max()) <= 12
    assert certify_expansion(out, exact_limit=4)["condition_met"]


def test_resource_and_degree_failure_are_closed():
    with pytest.raises(ExpansionConstructionError, match="degree") as exc:
        enhance_expansion(path(8), max_degree=2)
    assert exc.value.reason == "degree_budget"
    star = incidence(8, [(0, v) for v in range(1, 8)])
    with pytest.raises(ExpansionConstructionError, match="exceeds max_degree"):
        enhance_expansion(star, max_degree=3)
    with pytest.raises(ExpansionConstructionError, match="resource limits") as exc:
        enhance_expansion(path(40), exact_limit=4, certificate_limit=20)
    assert exc.value.reason == "certificate_budget"
    assert not certify_expansion(path(40), exact_limit=4, certificate_limit=20)["condition_met"]


def test_parallel_edges_count_and_single_vertex_is_vacuous():
    b = np.repeat(path(6), 3, axis=0)
    out, report = enhance_expansion(b, max_degree=6)
    assert report["condition_met"]
    assert report["added_edge_count"] == 0
    assert np.array_equal(b, out)
    assert independent_beta(b, range(6), 6) == 1
    out, report = enhance_expansion(np.zeros((0, 1), dtype=np.uint8))
    assert report["condition_met"] and out.shape == (0, 1)


@pytest.mark.parametrize("kwargs", [
    {"max_degree": True}, {"max_degree": 0}, {"seed": -1}, {"seed": True},
    {"attempts": 0}, {"attempts": False}, {"exact_limit": 25},
    {"certificate_limit": 0}, {"certificate_limit": True},
    {"max_entries": False}, {"ports": [0, 0]}, {"distance": True},
])
def test_reject_invalid_options_even_if_graph_is_already_certified(kwargs):
    with pytest.raises(ValueError):
        enhance_expansion(path(2), **kwargs)


def test_graph_digest_changes_on_parallel_edge_addition():
    b = path(4)
    first = certify_expansion(b)
    second = certify_expansion(np.vstack((b, b[0])))
    assert first["graph_sha256"] != second["graph_sha256"]
