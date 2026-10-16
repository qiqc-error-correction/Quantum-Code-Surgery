"""Independent matrix/phase audits of conditional branching and its preview."""
from copy import deepcopy
import json

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.parallel_branching import branching_deform
from codesurgery.symplectic import commutation_matrix, pauli_product


def audit_actual_readout_and_growth(result):
    """Eliminate every ancilla independently of planner/replay certificates."""
    p = result.measurement_preview
    px, pz, ps = p['check_x'], p['check_z'], p['check_signs']
    sx, sz = result.source['check_x'], result.source['check_z']
    bx, bz = result.source['basis_x'], result.source['basis_z']
    tx, tz = result.source['target_x'], result.source['target_z']
    m, n = sx.shape
    assert not np.any(commutation_matrix(px, pz))
    # Check every dependency with literal Pauli multiplication, including Y.
    dependencies = nullspace(np.hstack((px, pz)).T)
    for relation in dependencies:
        xx, zz, phase = pauli_product(px, pz, ps, np.flatnonzero(relation))
        assert not np.any(xx | zz) and phase == 0
    eliminate = nullspace(np.hstack((px[:, n:], pz[:, n:])).T)
    memory_x, memory_z = (eliminate @ px[:, :n]) % 2, (eliminate @ pz[:, :n]) % 2
    assert not np.any((memory_x @ sz.T + memory_z @ sx.T) % 2)
    observed_signatures = (memory_x @ bz.T + memory_z @ bx.T) % 2
    requested_signatures = (tx @ bz.T + tz @ bx.T) % 2
    assert in_rowspace(observed_signatures, requested_signatures)
    assert in_rowspace(requested_signatures, observed_signatures)
    assert rank(observed_signatures) == rank(requested_signatures)
    assert p['n'] - rank(np.hstack((px, pz))) == result.k - rank(requested_signatures)
    # The branching core itself must reveal zero original logical information.
    cx, cz = result.check_x, result.check_z
    remove_core_aux = nullspace(np.hstack((cx[:, n:], cz[:, n:])).T)
    core_x, core_z = (remove_core_aux @ cx[:, :n]) % 2, (remove_core_aux @ cz[:, :n]) % 2
    assert not np.any((core_x @ bz.T + core_z @ bx.T) % 2)
    assert result.k == n - rank(np.hstack((sx, sz)))
    weights, source_weights = px | pz, sx | sz
    rows = weights[:m].sum(axis=1, dtype=np.int64) - source_weights.sum(axis=1)
    columns = weights[:, :n].sum(axis=0, dtype=np.int64) - source_weights.sum(axis=0)
    np.testing.assert_array_equal(rows, result.routing['projected_original_row_growth'])
    np.testing.assert_array_equal(columns, result.routing['projected_original_column_growth'])
    if result.routing['mode'] == 'adaptive':
        assert np.all(rows <= result.routing['max_check_increase'])
        assert np.all(columns <= result.routing['max_qubit_increase'])
    assert p['distance_certificate'] is None and p['expansion_certificate'] is None
    return rows, columns


def test_default_overlapping_z_targets_stay_on_memory_at_equality():
    result = branching_deform(four_qubit_code(), targets=['ZZII', 'ZIZI'])
    assert result.routing['mode'] == 'adaptive'
    assert result.routing['max_check_increase'] == result.routing['max_qubit_increase'] == 2
    assert result.n == 4 and result.k == 2 and result.branches == []
    assert result.routing['direct_targets'] == [0, 1]
    assert not result.summary()['fully_disjoint']
    rows, columns = audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(rows, [2, 0])
    np.testing.assert_array_equal(columns, [2, 1, 1, 0])
    assert result.measurement_preview['n'] == 6 and result.measurement_preview['k'] == 0
    assert result.validate()['measurement_attachment_limits_satisfied']
    assert result.summary()['logical_measurement_performed'] is False


def test_single_weight_six_boundary_branches_even_without_force_copy():
    code = CSSCode(np.ones((1, 6), np.uint8), np.zeros((0, 6), np.uint8))
    result = branching_deform(code, targets=['ZZZZZZ'], optimize_logical=False)
    assert result.routing['direct_all_row_growth'] == [3]
    assert result.routing['direct_targets'] == []
    assert result.routing['root_groups'] == [[0]]
    assert len(result.branches) == 1 and result.n == 13 and result.k == 5
    assert result.branches[0]['F'].sum(axis=1).tolist() == [6]
    rows, columns = audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(columns, [1] * 6)
    assert result.measurement_preview['n'] == 18
    # Limits use the input code's individual baselines, not a fictitious
    # baseline for the newly created cap checks.
    assert result.summary()['projected_max_new_check_weight'] == 10
    assert result.routing['auxiliary_limits_enforced'] is False


def test_three_repeated_targets_use_one_shared_root_and_one_direct_target():
    result = branching_deform(four_qubit_code(), targets=['ZZII'] * 3,
                              optimize_logical=False)
    assert result.routing['direct_all_row_growth'] == [3, 0]
    assert result.n == 13 and result.k == 2
    assert len(result.routing['direct_targets']) == 1
    assert len(result.routing['root_groups']) == 1
    assert len(result.routing['root_groups'][0]) == 2
    assert len(result.branches) == 3
    assert result.measurement_preview['target_logical_rank'] == 1
    assert result.measurement_preview['k'] == 1
    assert result.summary()['fully_disjoint']
    rows, columns = audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(rows, [2, 0])
    np.testing.assert_array_equal(columns, [2, 2, 0, 0])


def test_row_threshold_uses_each_checks_own_baseline_not_global_maximum():
    code = CSSCode([[1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1]], [])
    result = branching_deform(code, targets=['IIIIIIZZ'] * 3,
                              optimize_logical=False, max_qubit_increase=3)
    # Direct attachment makes the light check 2+3=5, below the unrelated
    # global threshold 6+2=8. Its individual +2 budget still requires a branch.
    assert result.routing['direct_all_row_growth'] == [0, 3]
    assert result.routing['root_groups']
    rows, _ = audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(rows, [0, 2])


def test_column_threshold_uses_each_qubits_own_baseline_not_global_maximum():
    code = CSSCode([[1, 1, 0, 0, 0, 0]] * 3 + [[0, 0, 0, 0, 1, 1]], [])
    result = branching_deform(code, targets=['IIIIZZ'] * 3,
                              optimize_logical=False, max_check_increase=3)
    # Qubits 4,5 start at degree 1: direct degree 4 is below the global 3+2,
    # but exceeds their own 1+2 and must trigger conditional branching.
    assert result.routing['direct_all_column_growth'] == [0, 0, 0, 0, 3, 3]
    assert result.routing['root_groups']
    _, columns = audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(columns, [0, 0, 0, 0, 2, 2])


@pytest.mark.parametrize('logical,expected_outcomes', [
    (['YI'], [0]),
    (['YI', 'IY'], [0, 0]),
    (['YI', 'IY', 'YY'], [0, 0, 0]),
    (['YI', 'IX', '-YX'], [0, 0, 1]),
    (['-YI', 'YI', '-YI'], [0, 1, 0]),
])
def test_y_negative_and_dependent_targets_preserve_the_measured_span(logical, expected_outcomes):
    result = branching_deform(four_qubit_code(), logical_paulis=logical,
                              optimize_logical=False)
    audit_actual_readout_and_growth(result)
    np.testing.assert_array_equal(result.measurement_preview['target_outcome_signs'], expected_outcomes)
    assert result.validate()['signed_relations_valid']
    # Signed vertex products are exactly the routed targets in the chosen
    # consistent preview sector, including dependent/opposite observables.
    p = result.measurement_preview
    for i, graph in enumerate(p['graphs']):
        x, z, phase = pauli_product(p['check_x'], p['check_z'], p['check_signs'], graph['vertex_rows'])
        np.testing.assert_array_equal(x[:result.n], result.leaf_x[i])
        np.testing.assert_array_equal(z[:result.n], result.leaf_z[i])
        assert not np.any(x[result.n:] | z[result.n:])
        assert phase == 2 * int(result.leaf_signs[i] ^ p['target_outcome_signs'][i])


def test_explicit_y_support_and_sign_remain_when_preservation_is_requested():
    result = branching_deform(four_qubit_code(), targets=['-YYII', 'YYII'],
                              representatives='preserve', optimize_logical=False)
    assert result.branches == []
    np.testing.assert_array_equal(result.leaf_signs, [1, 0])
    np.testing.assert_array_equal(result.leaf_x[:, :4], [[1, 1, 0, 0]] * 2)
    np.testing.assert_array_equal(result.leaf_z[:, :4], [[1, 1, 0, 0]] * 2)
    audit_actual_readout_and_growth(result)


def test_explicit_always_mode_keeps_legacy_two_leaf_construction():
    result = branching_deform(four_qubit_code(), targets=['ZZII', 'ZIZI'],
                              branching_mode='always', optimize_logical=False)
    assert result.n == 10 and len(result.branches) == 2
    assert result.routing['direct_targets'] == [] and result.summary()['fully_disjoint']
    audit_actual_readout_and_growth(result)


@pytest.fixture(scope='module')
def routed_result():
    return branching_deform(four_qubit_code(), targets=['ZZII'] * 3,
                             optimize_logical=False)


@pytest.mark.parametrize('damage', [
    'routing_missing', 'routing_roots', 'routing_direct', 'routing_growth',
    'routing_ancillas', 'routing_limit', 'routing_mode', 'branch_parent',
    'preview_missing', 'preview_check', 'preview_sign', 'preview_outcome',
    'preview_matching', 'preview_owner', 'preview_growth', 'preview_k',
    'preview_distance_claim',
])
def test_material_route_or_preview_corruption_is_detected(routed_result, damage):
    result = deepcopy(routed_result)
    if damage == 'routing_missing':
        result.routing = None
    elif damage == 'routing_roots':
        result.routing['root_groups'] = []
    elif damage == 'routing_direct':
        result.routing['direct_targets'] = []
    elif damage == 'routing_growth':
        result.routing['projected_original_column_growth'][0] += 1
    elif damage == 'routing_ancillas':
        result.routing['ancillas'] += 1
    elif damage == 'routing_limit':
        result.routing['max_check_increase'] = 0
    elif damage == 'routing_mode':
        result.routing['mode'] = 'invented'
    elif damage == 'branch_parent':
        result.branches[0]['parent'] = 0
    elif damage == 'preview_missing':
        result.measurement_preview = None
    elif damage == 'preview_check':
        result.measurement_preview['check_x'][0, -1] ^= 1
    elif damage == 'preview_sign':
        result.measurement_preview['check_signs'][0] ^= 1
    elif damage == 'preview_outcome':
        result.measurement_preview['target_outcome_signs'][0] ^= 1
    elif damage == 'preview_matching':
        result.measurement_preview['graphs'][0]['M'][0, 0] ^= 1
    elif damage == 'preview_owner':
        result.measurement_preview['graphs'][0]['edge_owners'][0] = 999
    elif damage == 'preview_growth':
        result.measurement_preview['memory_column_growth'][0] += 1
    elif damage == 'preview_k':
        result.measurement_preview['k'] += 1
    else:
        result.measurement_preview['distance_certificate'] = {'certified': True, 'd': 10}
    with pytest.raises(ValueError):
        result.validate()


@pytest.mark.parametrize('extension', ['.json', '.npz'])
def test_failed_preview_validation_does_not_overwrite_saved_file(routed_result, tmp_path, extension):
    result = deepcopy(routed_result)
    path = tmp_path / ('already_saved' + extension)
    sentinel = b'previous verified artifact, preserve these bytes\x00\xff'
    path.write_bytes(sentinel)
    result.measurement_preview['check_signs'][0] ^= 1
    with pytest.raises(ValueError):
        result.save(path)
    assert path.read_bytes() == sentinel


@pytest.mark.parametrize('extension', ['.json', '.npz'])
def test_saved_output_includes_separate_preview_and_original_limits(routed_result, tmp_path, extension):
    path = tmp_path / ('routing' + extension)
    routed_result.save(path)
    if extension == '.json':
        record = json.loads(path.read_text())
    else:
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(saved['record_json'].item())
    assert record['routing']['mode'] == 'adaptive'
    assert record['routing']['max_check_increase'] == record['routing']['max_qubit_increase'] == 2
    assert record['metadata']['logical_measurement_performed'] is False
    assert record['measurement_preview']['kind'] == 'direct_paired_gauging_preview'
    assert record['measurement_preview']['distance_certificate'] is None
    assert record['measurement_preview']['outcome_sector'] == 'arbitrary_consistent_preview_assignment'
    assert record['summary']['n'] == routed_result.n
    assert record['summary']['projected_measurement_n'] == routed_result.measurement_preview['n']
