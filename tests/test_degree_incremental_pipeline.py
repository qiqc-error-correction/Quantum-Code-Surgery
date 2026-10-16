"""Actual signed stabilizer distance, deferred expansion and fixed row limits."""
from itertools import combinations, product

import numpy as np
import pytest

from codesurgery.distance_preservation import ensure_distance
from codesurgery.examples import direct_sum, steane_code, four_qubit_code
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank


def mixed_cycle():
    code = direct_sum(steane_code(), steane_code())
    basis = code.standard_logicals()
    target = dict(x=basis.jx[0], z=basis.jz[1], sign=1)
    graph = np.zeros((6, 6), dtype=np.uint8)
    for vertex in range(6):
        graph[vertex, [vertex, (vertex + 1) % 6]] = 1
    basic = gauging_deform(code, target, graph=graph, require_distance=False)
    return code, basic


def independent_distance_and_readout(code, result):
    cx, cz = result.check_x.astype(int), result.check_z.astype(int)
    checks = np.hstack((cx, cz))
    assert not np.any((cx @ cz.T + cz @ cx.T) % 2)
    # Independently exclude every weight-one and weight-two logical Pauli.
    candidates = []
    for weight in (1, 2):
        for support in combinations(range(result.n), weight):
            for assignment in product(((1, 0), (0, 1), (1, 1)), repeat=weight):
                error = np.zeros(2 * result.n, dtype=np.uint8)
                for q, (x, z) in zip(support, assignment):
                    error[q], error[result.n + q] = x, z
                candidates.append(error)
    errors = np.asarray(candidates)
    syndrome = (errors[:, :result.n] @ cz.T + errors[:, result.n:] @ cx.T) % 2
    assert in_rowspace(errors[~syndrome.any(axis=1)], checks)
    # The surviving logical witness really has weight three.
    witness = result.metadata["distance_calculation"]["witness"]
    x, z = np.asarray(witness["x"]), np.asarray(witness["z"])
    assert np.count_nonzero(x | z) == 3
    assert not np.any((cx @ z + cz @ x) % 2)
    assert not in_rowspace(np.r_[x, z], checks)
    # Auxiliary cancellation yields exactly the requested logical class.
    cancel = nullspace(np.hstack((cx[:, code.n:], cz[:, code.n:])).T)
    image = cancel @ np.hstack((cx[:, :code.n], cz[:, :code.n])) % 2
    old = np.block([[code.hx, np.zeros_like(code.hx)],
                    [np.zeros_like(code.hz), code.hz]])
    target = np.r_[result.measurement_x, result.measurement_z]
    assert rank(np.vstack((old, image))) == rank(old) + 1
    assert in_rowspace(image, np.vstack((old, target)))
    assert result.measurement_sign == 1 and result.validate()


def test_cubic_candidate_restores_real_distance_before_any_candidate_h(monkeypatch):
    from codesurgery.distance_preservation import expansion
    from codesurgery.gauging_measurement import api as graph_api
    code, basic = mixed_cycle()

    def forbidden(*args, **kwargs):
        pytest.fail("A distance-certified candidate must not calculate graph h")

    monkeypatch.setattr(expansion, "_strict_global_certificate", forbidden)
    monkeypatch.setattr(graph_api, "relative_expansion", forbidden)
    result = ensure_distance(basic, max_matching_congestion=3, distance_method="enumeration")
    post = result.metadata["distance_postprocessing"]
    search = post["expansion"]
    assert post["initial_distance_calculation"]["upper_bound"] == 2
    assert result.metadata["distance_calculation"]["distance"] == 3
    assert result.n == 23 and result.k_stabilizer == 1
    assert search["acceptance"] == "distance" and not search["expansion_evaluated"]
    assert search["final_degrees"] == [3] * 6 and search["added_edge_count"] == 3
    assert result.metadata["relative_expansion"]["deferred"]
    assert not result.metadata["relative_expansion"]["condition_met"]
    assert result.metadata["max_check_weight"] <= post["effective_row_threshold"] == 6
    independent_distance_and_readout(code, result)


def test_candidate_sparsification_budget_failure_retries_without_lifting_limits(monkeypatch):
    from codesurgery.distance_preservation import sparsify
    code, basic = mixed_cycle()
    real = sparsify.sparsify_cycles
    calls = []

    def first_too_large(*args, **kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            raise ValueError("Dense graph construction exceeds max_entries")
        return real(*args, **kwargs)

    monkeypatch.setattr(sparsify, "sparsify_cycles", first_too_large)
    result = ensure_distance(basic, max_matching_congestion=3, distance_method="enumeration")
    history = result.metadata["distance_postprocessing"]["expansion"]["search_history"]
    assert len(history) >= 2
    assert history[0]["evaluation"]["eligible"] is False
    assert history[0]["certificate"]["expansion_evaluated"] is False
    assert history[0]["stage_degree"] == history[1]["stage_degree"] == 3
    assert history[0]["candidate_seed"] != history[1]["candidate_seed"]
    assert all(item["max_cycle_weight"] == 6 for item in calls)
    independent_distance_and_readout(code, result)


def test_degree_cap_exhaustion_never_silently_increases_weight():
    from codesurgery.distance_preservation.expansion import ExpansionConstructionError
    _, basic = mixed_cycle()
    with pytest.raises(ExpansionConstructionError, match="fixed degree"):
        ensure_distance(basic, max_matching_congestion=3, max_base_degree=2,
                        distance_method="enumeration")
    assert basic.metadata["graph_incidence"].sum(axis=0).tolist() == [2] * 6


def test_full_memory_row_limit_rejects_all_seeds_without_raising_the_cap():
    from codesurgery.distance_preservation.expansion import ExpansionConstructionError
    _, basic = mixed_cycle()
    with pytest.raises(ExpansionConstructionError) as failure:
        ensure_distance(basic, max_matching_congestion=3, max_check_weight=5,
                        attempts=2, distance_method="enumeration")
    history = failure.value.details["search_history"]
    assert history
    for entry in history:
        assert max(entry["final_degrees"]) <= 4
        evaluation = entry["evaluation"]
        assert evaluation["eligible"] is False
        assert evaluation["sparsity_failures"]["max_check_weight"]["limit"] == 5
        assert not entry["certificate"]["expansion_evaluated"]


def test_legacy_hamiltonian_can_still_be_selected_explicitly():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_pair_deform(code, "XXIIIIII", "IIIIZZII", adapter_size=1,
        distance_options={"expansion_strategy": "hamiltonian"})
    search = result.metadata["distance_postprocessing"]["expansion"]
    assert search["augmentation_strategy"] == "hamiltonian_cycle_overlays"
    assert result.n == 15 and result.metadata["distance_calculation"]["distance"] == 2


def test_unknown_strategy_rejected_even_when_original_distance_passes():
    with pytest.raises(ValueError, match="expansion_strategy"):
        gauging_deform(four_qubit_code(), "XXII",
            distance_options={"expansion_strategy": "typo"})
