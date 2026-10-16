"""Independent Pauli-group checks of exact irreducible-cluster search."""
from copy import deepcopy
import json

import numpy as np
import pytest

from codesurgery.distance_preservation._distance_cluster import cluster_distance


def checks(words):
    n = len(words[0])
    x = np.array([[int(p in "XY") for p in row] for row in words], dtype=np.uint8)
    z = np.array([[int(p in "ZY") for p in row] for row in words], dtype=np.uint8)
    return x.reshape(-1, n), z.reshape(-1, n)


def independent_reference(x, z, max_weight=None):
    """Build the entire group and all logical Paulis, without GF(2) solvers.

    Deliberately give the backend a *largest*-weight witness so its search
    cannot accidentally pass because preprocessing already knew distance.
    """
    n = x.shape[1]
    mask = (1 << n) - 1
    pack = lambda row: sum(int(value) << q for q, value in enumerate(row))
    generators = [pack(a) | (pack(b) << n) for a, b in zip(x, z)]
    group = {0}
    for generator in generators:
        group |= {value ^ generator for value in tuple(group)}
    logicals = []
    for value in range(1, 1 << (2 * n)):
        if value in group:
            continue
        a, b = value & mask, value >> n
        if all((((a & (row >> n)) ^ (b & (row & mask))).bit_count() % 2 == 0)
               for row in generators):
            logicals.append(((a | b).bit_count(), value))
    rank = len(group).bit_length() - 1
    initial = {
        "status": "unknown" if logicals else "no_logicals",
        "distance": None, "distance_is_infinite": not logicals,
        "proven_lower_bound": 1 if logicals else None,
        "upper_bound": None, "witness": None,
        "n": n, "k": n - rank, "stabilizer_rank": rank,
        "method": "independent_test_fixture", "distance_kind": "static_stabilizer",
        "max_weight": n if max_weight is None else max_weight,
        "max_candidates": 0, "checked_candidates": 0, "completed_weight": 0,
        "stopped_at_weight": None, "stop_reason": "no_logical_qubits" if not logicals else None,
    }
    if logicals:
        weight, value = max(logicals)
        a, b = value & mask, value >> n
        initial["upper_bound"] = weight
        initial["witness"] = {
            "weight": weight, "x": [(a >> q) & 1 for q in range(n)],
            "z": [(b >> q) & 1 for q in range(n)],
            "pauli": "".join("IXZY"[((a >> q) & 1) + 2 * ((b >> q) & 1)] for q in range(n)),
        }
    return initial, min((weight for weight, _ in logicals), default=None), group


def verify(x, z, report, expected, group):
    if expected is None:
        assert report["status"] == "no_logicals"
        assert report["distance_is_infinite"]
        return
    assert report["proven_lower_bound"] <= expected <= report["upper_bound"]
    if report["status"] == "exact":
        assert report["distance"] == expected
        assert report["proven_lower_bound"] == expected == report["upper_bound"]
    witness = report["witness"]
    a, b = np.array(witness["x"]), np.array(witness["z"])
    assert not np.any((x @ b + z @ a) % 2)
    value = sum(int(bit) << q for q, bit in enumerate(np.r_[a, b]))
    assert value not in group
    assert int((a | b).sum()) == witness["weight"] == report["upper_bound"]
    json.dumps(report, allow_nan=False)


def test_every_commuting_one_row_css_pair_on_three_qubits():
    for a in range(8):
        for b in range(8):
            if (a & b).bit_count() % 2:
                continue
            x = np.array([[(a >> q) & 1 for q in range(3)], [0, 0, 0]], dtype=np.uint8)
            z = np.array([[0, 0, 0], [(b >> q) & 1 for q in range(3)]], dtype=np.uint8)
            initial, expected, group = independent_reference(x, z)
            report = cluster_distance(x, z, initial)
            assert report["status"] == "exact"
            verify(x, z, report, expected, group)


@pytest.mark.parametrize("seed", range(12))
def test_random_commuting_mixed_codes_match_full_pauli_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = 5
    r = 1 + seed % n
    x = np.zeros((r, n), dtype=np.uint8)
    z = np.eye(n, dtype=np.uint8)[:r].copy()
    for _ in range(30):
        a, b = rng.choice(n, size=2, replace=False)
        gate = rng.integers(3)
        if gate == 0:
            x[:, a], z[:, a] = z[:, a].copy(), x[:, a].copy()
        elif gate == 1:
            z[:, a] ^= x[:, a]
        else:
            x[:, b] ^= x[:, a]
            z[:, a] ^= z[:, b]
    initial, expected, group = independent_reference(x, z)
    report = cluster_distance(x, z, initial)
    verify(x, z, report, expected, group)
    assert report["status"] in ("exact", "no_logicals")


@pytest.mark.parametrize("words", [
    ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
    ["XXXXXX", "ZZZZZZ"],  # Dense stabilizer generators.
    ["XXXXII", "ZZZZII", "IIIIXX", "IIIIZZ"],  # Disconnected Bell component.
    ["XXXXII", "ZZZZII", "IIIIYI", "IIIIIZ"],  # Local degenerate stabilizers.
    ["XZZXII", "IXZZXI", "XIXZZI", "ZXIXZI", "IIIIIY"],
])
def test_dense_disconnected_and_degenerate_checks(words):
    x, z = checks(words)
    # Redundant/zero checks must not change the row-space exclusion.
    x = np.vstack([x, x[0] ^ x[-1], np.zeros(x.shape[1], dtype=np.uint8)])
    z = np.vstack([z, z[0] ^ z[-1], np.zeros(z.shape[1], dtype=np.uint8)])
    initial, expected, group = independent_reference(x, z)
    report = cluster_distance(x, z, initial)
    assert report["status"] == "exact"
    verify(x, z, report, expected, group)


def steane_checks():
    return checks(["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"])


@pytest.mark.parametrize("seed", range(4))
def test_local_clifford_steane_has_mixed_distance_three(seed):
    x, z = steane_checks()
    rng = np.random.default_rng(seed)
    for q in range(7):
        if rng.integers(2):
            x[:, q], z[:, q] = z[:, q].copy(), x[:, q].copy()
        if rng.integers(2):
            z[:, q] ^= x[:, q]
    initial, expected, group = independent_reference(x, z)
    report = cluster_distance(x, z, initial)
    assert report["distance"] == 3
    verify(x, z, report, expected, group)


@pytest.mark.parametrize("budget,completed,lower", [(0, 0, 1), (1, 0, 1), (13, 0, 1),
    (14, 1, 2), (15, 1, 2)])
def test_interrupted_shell_never_inflates_lower_bound(budget, completed, lower):
    x, z = steane_checks()
    initial, expected, group = independent_reference(x, z)
    report = cluster_distance(x, z, initial, max_candidates=budget)
    assert report["checked_candidates"] == budget
    assert report["completed_weight"] == completed
    assert report["proven_lower_bound"] == lower
    assert report["stop_reason"] == "candidate_budget"
    assert report["distance"] is None
    verify(x, z, report, expected, group)


def test_exact_budget_at_threshold_and_one_node_before():
    x, z = steane_checks()
    initial, expected, group = independent_reference(x, z, max_weight=2)
    full = cluster_distance(x, z, initial)
    used = full["checked_candidates"]
    assert full["proven_lower_bound"] == 3
    assert full["distance"] is None  # Deliberately loose starting upper bound.
    assert full["stop_reason"] == "weight_limit"
    exact_budget = cluster_distance(x, z, initial, max_candidates=used)
    short_budget = cluster_distance(x, z, initial, max_candidates=used - 1)
    assert exact_budget["proven_lower_bound"] == 3
    assert short_budget["proven_lower_bound"] == 2
    assert short_budget["stop_reason"] == "candidate_budget"
    verify(x, z, exact_budget, expected, group)
    verify(x, z, short_budget, expected, group)


def test_upper_bound_meeting_proven_threshold_is_exact():
    x, z = steane_checks()
    initial, _, _ = independent_reference(x, z, max_weight=2)
    # This independently checked logical witness is pure X on sites 0,1,2.
    initial["upper_bound"] = 3
    initial["witness"] = {"x": [1, 1, 1, 0, 0, 0, 0], "z": [0] * 7,
                          "pauli": "XXXIIII", "weight": 3}
    report = cluster_distance(x, z, initial)
    assert report["status"] == "exact" and report["distance"] == 3
    assert report["stop_reason"] == "bounds_meet"


def test_input_is_unchanged_and_zero_weight_search_returns_bounds_only():
    x, z = steane_checks()
    initial, expected, group = independent_reference(x, z, max_weight=0)
    original = deepcopy(initial)
    report = cluster_distance(x, z, initial)
    assert report["checked_candidates"] == 0
    assert report["proven_lower_bound"] == 1
    assert report["distance"] is None
    assert initial == original
    report["witness"]["x"][0] ^= 1
    assert initial == original
    verify(x, z, initial, expected, group)


def test_existing_bound_and_other_backend_metadata_are_preserved():
    x, z = steane_checks()
    initial, expected, group = independent_reference(x, z)
    initial.update(proven_lower_bound=2, completed_weight=0,
                   ris={"trials": 12}, milp={"optimal": False})
    report = cluster_distance(x, z, initial, max_candidates=0)
    assert report["status"] == "bounded"
    assert report["proven_lower_bound"] == 2
    assert report["stopped_at_weight"] == 2
    assert report["ris"] == {"trials": 12}
    assert report["milp"] == {"optimal": False}
    verify(x, z, report, expected, group)


@pytest.mark.parametrize("budget", [-1, True, 0.5])
def test_invalid_budget_is_rejected(budget):
    x, z = steane_checks()
    initial, _, _ = independent_reference(x, z)
    with pytest.raises(ValueError, match="max_candidates"):
        cluster_distance(x, z, initial, max_candidates=budget)
