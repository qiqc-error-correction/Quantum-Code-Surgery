"""Chain maps, readout kernels and degree constraints of high-rate ancillas."""

import numpy as np
import pytest

from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.high_rate_surgery.ancilla import (
    ConstructionError,
    append_copy,
    append_intralayer,
    append_layer,
    hgp_ancilla,
    initialize,
    propose_growth,
)


def _initialize(f, *, targets=None, **options):
    f = np.asarray(f, dtype=np.uint8)
    s = f.shape[1]
    px = np.eye(s, dtype=np.uint8)
    defaults = {"max_check_weight": 8, "max_qubit_degree": 8, "trials": 4}
    defaults.update(options)
    return initialize(f, np.ones((1, s), dtype=np.uint8) if targets is None else targets,
                      px, np.zeros_like(px), **defaults)


def _same_space(a, b):
    assert rank(a) == rank(b)
    assert in_rowspace(a, b)
    assert in_rowspace(b, a)


def _chain_map(state, f):
    padded = np.pad(f, ((0, 0), (0, state.n_checks - f.shape[1])))
    np.testing.assert_array_equal(state.gamma0 @ state.boundary % 2, padded)
    assert not np.any(state.cycles @ state.boundary % 2)


def test_initialization_preserves_redundant_source_rows_and_fills_cycle_space():
    f = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    state = _initialize(f)
    _chain_map(state, f)
    np.testing.assert_array_equal(state.boundary, f[[0, 2, 3]])
    assert state.metadata["initial_source_rows"] == [0, 2, 3]
    assert rank(state.cycles) == 1
    _same_space(nullspace(state.boundary), np.ones((1, 3), dtype=np.uint8))
    _same_space(state.cycles, nullspace(state.boundary.T))


def test_separating_rows_remove_component_readout_but_can_preserve_safe_kernel():
    f = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    target = np.ones((1, 4), dtype=np.uint8)
    separated = _initialize(f)
    _chain_map(separated, f)
    _same_space(nullspace(separated.boundary), target)
    assert separated.metadata["separation_rows"] == 1
    # The caller certifies that the extra component is a memory stabilizer.
    allowed = nullspace(f)
    retained = _initialize(f, allowed_kernel=allowed)
    _chain_map(retained, f)
    _same_space(nullspace(retained.boundary), allowed)
    assert retained.metadata["separation_rows"] == 0


def test_safe_kernel_can_fall_back_to_sparse_target_only_boundary():
    f = np.zeros((0, 4), dtype=np.uint8)
    allowed = nullspace(np.ones((1, 4), dtype=np.uint8))
    # Preserving this entire safe kernel needs the weight-four row 1111.
    # Retaining just target 1111 instead permits a connected weight-two path.
    state = _initialize(f, allowed_kernel=allowed,
                        max_check_weight=3, max_qubit_degree=3, trials=12)
    assert state.metadata["kernel_fallback_to_targets"] is True
    assert state.metadata["kernel_dimension"] == 1
    assert np.all(state.boundary.sum(axis=1) == 2)
    assert np.all(state.boundary.sum(axis=0) + 1 <= 3)
    _same_space(nullspace(state.boundary), np.ones((1, 4), dtype=np.uint8))
    with pytest.raises(ConstructionError, match="no bounded-degree separating"):
        _initialize(f, allowed_kernel=allowed, fallback_to_targets=False,
                    max_check_weight=3, max_qubit_degree=3)


def test_initialization_can_leave_an_unfilled_cycle_as_a_gauge_degree():
    f = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    state = _initialize(f, admissible=lambda candidate: len(candidate.cycles) == 0)
    _chain_map(state, f)
    assert state.cycles.shape == (0, 3)
    assert state.metadata["unfilled_cycle_dimension"] == 1


def test_base_degree_violation_is_not_disguised_as_a_distance_certificate():
    with pytest.raises(ConstructionError, match="initial boundary already exceeds"):
        _initialize([[1, 1, 0], [0, 1, 1]], max_check_weight=2)
    with pytest.raises(ValueError, match="commute"):
        _initialize([[1, 0, 0]])


def test_layer_copy_and_intralayer_moves_preserve_all_chain_relations():
    f = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    initial = _initialize(f)
    before = tuple(a.copy() for a in (initial.boundary, initial.cycles, initial.gamma0))
    state = append_layer(initial, max_check_weight=4, max_qubit_degree=5)
    assert state.bridges[(0, 1)] == (2, 3, 4)
    assert state.layers[1].internal_qubits == ()
    _same_space(nullspace(state.boundary), np.ones((1, 6), dtype=np.uint8))
    _chain_map(state, f)
    state = append_copy(state, 0, 0, 1, max_check_weight=4, max_qubit_degree=5)
    np.testing.assert_array_equal(state.cycles[-1], [1, 0, 1, 1, 0, 1])
    state = append_copy(state, 0, 1, 1, max_check_weight=4, max_qubit_degree=5)
    state = append_intralayer(state, 1, [1, 1], max_check_weight=4, max_qubit_degree=5)
    _chain_map(state, f)
    _same_space(nullspace(state.boundary), np.ones((1, 6), dtype=np.uint8))
    _same_space(state.cycles, nullspace(state.boundary.T))
    assert state.layers[1].internal_qubits == (5, 6, 7)
    for unchanged, snapshot in zip((initial.boundary, initial.cycles, initial.gamma0), before):
        np.testing.assert_array_equal(unchanged, snapshot)
    assert len(initial.layers) == 1
    assert initial.bridges == {}


def test_global_intralayer_vector_and_reverse_copy_use_global_qubit_indices():
    f = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    state = append_copy(append_layer(_initialize(f)), 0, 0, 1)
    v = np.zeros(state.n_qubits, dtype=np.uint8)
    v[5] = 1
    grown = append_intralayer(state, 1, v)
    _chain_map(grown, f)
    reverse = append_copy(grown, 1, 5, 0)
    _chain_map(reverse, f)
    with pytest.raises(ValueError, match="outside"):
        append_intralayer(state, 1, np.ones(state.n_qubits, dtype=np.uint8))
    with pytest.raises(ValueError, match="internal"):
        append_copy(state, 0, 2, 1)  # bridge, not a source-layer qubit


def test_direct_updates_apply_full_callback_and_do_not_mutate_on_rejection():
    state = _initialize([[1, 1, 0], [0, 1, 1]])
    before = state.boundary.copy()
    with pytest.raises(ConstructionError):
        append_intralayer(state, 0, [1, 0], admissible=lambda _: False)
    with pytest.raises(ConstructionError):
        append_layer(state, max_check_weight=3)
    np.testing.assert_array_equal(state.boundary, before)


def test_growth_scores_the_added_z_check_against_auxiliary_x_witnesses():
    f = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    state = _initialize(f)
    witnesses = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    grown, record = propose_growth(state, witnesses, max_check_weight=5,
                                  max_qubit_degree=5, trials=8)
    assert record["move"] == "intralayer"
    assert record["score"] == 2
    extended_witnesses = np.pad(witnesses, ((0, 0), (0, 1)))
    assert np.count_nonzero(extended_witnesses @ grown.cycles[-1] % 2) == record["score"]
    _chain_map(grown, f)


def test_growth_can_select_copy_after_full_code_callback_rejects_intralayer():
    f = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    state = append_layer(_initialize(f))
    witnesses = np.array([[1, 0, 0, 0, 0]], dtype=np.uint8)
    grown, record = propose_growth(
        state, witnesses, max_check_weight=4, max_qubit_degree=5,
        admissible=lambda candidate: bool(candidate.layers[1].internal_qubits))
    assert record["move"] == "copy"
    assert record["from_qubit"] == 0 and record["to_layer"] == 1
    assert np.count_nonzero(np.pad(witnesses, ((0, 0), (0, 1))) @ grown.cycles[-1] % 2) == 1
    _chain_map(grown, f)


def test_growth_layer_fallback_and_explicit_no_candidate_report():
    state = _initialize([[1, 1, 0], [0, 1, 1]])
    witnesses = np.array([[1, 0]], dtype=np.uint8)
    layer, record = propose_growth(
        state, witnesses, max_check_weight=4, max_qubit_degree=5,
        admissible=lambda candidate: len(candidate.cycles) == 0)
    assert record["move"] == "layer"
    assert len(layer.layers) == 2
    assert len(layer.cycles) == 0
    failed, record = propose_growth(
        state, witnesses, max_check_weight=4, max_qubit_degree=5,
        admissible=lambda _: False, allow_layer=False)
    assert failed is None
    assert record["reason"] == "no_degree_admissible_growth_found"
    failed, record = propose_growth(state, np.zeros((1, 2), dtype=np.uint8),
                                    max_check_weight=4, max_qubit_degree=5)
    assert failed is None
    assert record["reason"] == "no_witness_with_auxiliary_X_support"


def test_seeded_initialization_and_growth_are_reproducible():
    f = np.zeros((0, 5), dtype=np.uint8)
    a = _initialize(f, seed=17)
    b = _initialize(f, seed=17)
    np.testing.assert_array_equal(a.boundary, b.boundary)
    np.testing.assert_array_equal(a.cycles, b.cycles)
    witnesses = np.ones((1, a.n_qubits), dtype=np.uint8)
    a_next, a_record = propose_growth(a, witnesses, seed=9, max_check_weight=8, max_qubit_degree=8)
    b_next, b_record = propose_growth(b, witnesses, seed=9, max_check_weight=8, max_qubit_degree=8)
    assert a_record == b_record
    np.testing.assert_array_equal(a_next.boundary, b_next.boundary)


def test_default_growth_reserves_one_last_layer_check_slot_for_the_next_bridge():
    state = _initialize([[1, 1, 0], [0, 1, 1]])
    witness = [[1, 0]]
    grown, record = propose_growth(state, witness, max_check_weight=4,
                                  max_qubit_degree=5, trials=4)
    assert record["move"] == "intralayer"
    assert record["reserved_layer_capacity"] is True
    # Copying just the first row would saturate check 1.  The sum 101 avoids it.
    np.testing.assert_array_equal(grown.boundary[-1], [1, 0, 1])
    assert np.all(grown.boundary.sum(axis=0) + 1 <= 3)
    extended = append_layer(grown, max_check_weight=4, max_qubit_degree=5)
    assert len(extended.layers) == 2
    # Advanced callers can explicitly spend this reserved capacity.
    unreserved, record = propose_growth(
        state, witness, max_check_weight=4, max_qubit_degree=5,
        reserve_layer_capacity=False)
    np.testing.assert_array_equal(unreserved.boundary[-1], [1, 1, 0])
    assert record["reserved_layer_capacity"] is False
    with pytest.raises(ConstructionError):
        append_layer(unreserved, max_check_weight=4, max_qubit_degree=5)


def test_disallowing_more_layers_releases_capacity_and_reports_initial_exhaustion():
    state = _initialize([[1, 1, 0], [0, 1, 1]])
    grown, record = propose_growth(state, [[1, 0]], max_check_weight=4,
                                  max_qubit_degree=5, allow_layer=False)
    assert record["reserved_layer_capacity"] is False
    np.testing.assert_array_equal(grown.boundary[-1], [1, 1, 0])
    failed, record = propose_growth(state, [[1, 0]], max_check_weight=3,
                                   max_qubit_degree=5)
    assert failed is None
    assert record["reason"] == "last_layer_capacity_already_exhausted"


def test_e7_completes_partial_layer_with_rank_increasing_zero_score_copy():
    f = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    partial = append_copy(append_layer(_initialize(f)), 0, 0, 1)
    witness = np.array([[0, 0, 0, 0, 0, 1]], dtype=np.uint8)
    grown, record = propose_growth(partial, witness, max_check_weight=4,
                                  max_qubit_degree=5)
    assert record["move"] == "copy"
    assert record["from_qubit"] == 1
    assert record["score"] == 0
    assert record["reason"] == "complete_last_layer_kernel"
    assert grown.metadata["kernel_complete_layers"] == [True, True]
    assert grown.metadata["convergence_guarantee"] is False
    assert grown.metadata["hgp_fallback_implemented"] is False
    _chain_map(grown, f)
    # Disabling the E7 policy permits an otherwise valid but incomplete-layer
    # intralayer move.  The public primitive itself remains unrestricted too.
    relaxed, record = propose_growth(
        partial, witness, max_check_weight=4, max_qubit_degree=5,
        require_complete_layer_kernel=False)
    assert record["move"] == "intralayer"
    assert record["layer_index"] == 1
    assert relaxed.metadata["kernel_complete_layers"] == [True, False]
    assert relaxed.metadata["e7_kernel_complete_intralayer_only"] is False


def test_e7_does_not_append_another_layer_when_completion_is_blocked():
    partial = append_copy(append_layer(_initialize([[1, 1, 0], [0, 1, 1]])), 0, 0, 1)
    grown, record = propose_growth(
        partial, [[0, 0, 0, 0, 0, 1]], max_check_weight=4, max_qubit_degree=5,
        admissible=lambda candidate: len(candidate.cycles) == len(partial.cycles))
    assert grown is None
    assert record["reason"] == "last_layer_kernel_incomplete_and_no_admissible_completion"
    assert record["kernel_complete_layers"] == [True, False]


@pytest.mark.parametrize("hb,hd", [
    ([[1, 1, 0], [0, 1, 1]], [[1, 1]]),
    ([[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0], [0, 1, 1]]),
    (np.zeros((0, 2), dtype=np.uint8), [[1, 1]]),
])
def test_hgp_block_dimensions_chain_complex_and_tensor_kernel(hb, hd):
    hb, hd = np.asarray(hb, dtype=np.uint8), np.asarray(hd, dtype=np.uint8)
    b, c = hgp_ancilla(hb, hd)
    r, s = hb.shape
    t, u = hd.shape
    assert b.shape == (s * t + r * u, s * u)
    assert c.shape == (r * t, s * t + r * u)
    assert not np.any(c @ b % 2)
    _same_space(nullspace(b), np.kron(nullspace(hb), nullspace(hd)))
