"""Independent tree-level checks of the full signed branching map."""
from copy import deepcopy
from functools import partial

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace, nullspace, rank, solve
from codesurgery.parallel_branching import append_branch, branching_deform, retract_paulis
from codesurgery.symplectic import parse_pauli, pauli_product, pauli_string

# These historical tests exercise the complete binary-tree policy.
branching_deform = partial(branching_deform, branching_mode="always")


MIXED_TARGETS = ['YI', 'IX', 'YX', '-YI', '-IX', '-YX', 'YI', 'IX', 'YX']


def signed_equivalence_check(result):
    """Check witnesses using physical Pauli multiplication, independently of validate."""
    for i, w in enumerate(result.equivalence_witnesses):
        cx = np.vstack((result.lifted_target_x[i], result.check_x))
        cz = np.vstack((result.lifted_target_z[i], result.check_z))
        cs = np.r_[result.lifted_target_signs[i], result.check_signs]
        sx, sz, phase = pauli_product(cx, cz, cs, [0, *(np.flatnonzero(w) + 1)])
        np.testing.assert_array_equal(sx, result.leaf_x[i])
        np.testing.assert_array_equal(sz, result.leaf_z[i])
        assert phase == 2 * int(result.leaf_signs[i])


@pytest.mark.parametrize('leaf', ['auto', 'X', 'Z'])
def test_balanced_four_level_mixed_tree_preserves_every_logical(leaf):
    result = branching_deform(four_qubit_code(), logical_paulis=MIXED_TARGETS, leaf_pauli=leaf)
    summary = result.summary()
    assert summary['tree_depth'] == 4
    assert summary['stickers'] == 2 * (len(MIXED_TARGETS) - 1)
    assert summary['k'] == 2 and not summary['logical_measurement_performed']
    assert not result.is_css
    assert len(result.logical_basis_x) == 4
    assert result.metadata['target_logical_rank'] == 2
    assert len(result.leaf_x) == 9
    assert np.all((result.leaf_x | result.leaf_z).sum(axis=0) <= 1)
    assert not np.any((result.check_x @ result.check_z.T + result.check_z @ result.check_x.T) % 2)
    assert result.metadata['validation']['unintended_logical_readout_rank'] == 0
    signed_equivalence_check(result)
    # Inspect actual binary split structure, not merely the reported depth.
    for branch in result.branches:
        children = [b for b in result.branches if b['parent'] == branch['id']]
        if len(branch['targets']) == 1:
            assert not children
        else:
            assert len(children) == 2
            assert abs(len(children[0]['targets']) - len(children[1]['targets'])) <= 1
            assert sorted(children[0]['targets'] + children[1]['targets']) == sorted(branch['targets'])
            assert all(b['depth'] == branch['depth'] + 1 for b in children)


def test_composed_retraction_normalizer_quotient_and_static_distance():
    result = branching_deform(four_qubit_code(), logical_paulis=MIXED_TARGETS)
    normalizer = nullspace(np.hstack((result.check_z, result.check_x)))
    nx, nz = normalizer[:, :result.n], normalizer[:, result.n:]
    rx, rz = nx, nz
    for desc in reversed(result.branches):
        # Retraction only needs the width; no check replay in this independent test.
        output_n = desc['old_n'] + len(desc['dual_columns']) + len(desc['leaf_columns'])
        branch = dict(desc, check_x=np.empty((0, output_n), np.uint8))
        old_weight = np.count_nonzero(rx | rz, axis=1)
        rx, rz = retract_paulis(rx, rz, branch)
        assert np.all(np.count_nonzero(rx | rz, axis=1) <= old_weight)
    sx, sz = result.source['check_x'], result.source['check_z']
    assert not np.any((rx @ sz.T + rz @ sx.T) % 2)
    old_checks = np.hstack((sx, sz))
    new_checks = np.hstack((result.check_x, result.check_z))
    for nr, rr in zip(normalizer, np.hstack((rx, rz))):
        assert in_rowspace(nr, new_checks) == in_rowspace(rr, old_checks)
    # Independently rule out every weight-one logical in this d=2 example.
    eye = np.eye(result.n, dtype=np.uint8)
    for ex, ez in ((eye, np.zeros_like(eye)), (np.zeros_like(eye), eye), (eye, eye)):
        commutes = ~np.any((ex @ result.check_z.T + ez @ result.check_x.T) % 2, axis=1)
        assert all(in_rowspace(np.r_[xx, zz], new_checks)
                   for xx, zz in zip(ex[commutes], ez[commutes]))
    assert min(np.count_nonzero(result.leaf_x | result.leaf_z, axis=1)) == 2
    assert result.metadata['static_distance_certificate']['relation'] == 'd_output >= d_input'


@pytest.mark.parametrize('leaf', ['X', 'Z'])
def test_retraction_proof_does_not_require_local_union_to_be_a_logical(leaf):
    code = four_qubit_code()
    x = np.vstack((code.hx, np.zeros_like(code.hz)))
    z = np.vstack((np.zeros_like(code.hx), code.hz))
    px, pz, _ = parse_pauli('XYZX', 4)
    assert np.any((px @ z.T + pz @ x.T) % 2)
    branch = append_branch(x, z, [0, 0], px, pz, leaf_pauli=leaf)
    bx, bz = branch['check_x'], branch['check_z']
    normalizer = nullspace(np.hstack((bz, bx)))
    nn = bx.shape[1]
    nx, nz = normalizer[:, :nn], normalizer[:, nn:]
    rx, rz = retract_paulis(nx, nz, branch)
    assert not np.any((rx @ z.T + rz @ x.T) % 2)
    assert np.all(np.count_nonzero(rx | rz, axis=1) <= np.count_nonzero(nx | nz, axis=1))
    for nr, rr in zip(normalizer, np.hstack((rx, rz))):
        assert in_rowspace(nr, np.hstack((bx, bz))) == in_rowspace(rr, np.hstack((x, z)))


def test_signed_canonicalization_survives_tree_transport():
    result = branching_deform(four_qubit_code(), targets=['YZZY', 'ZIIZ'])
    prepared = [pauli_string(x, z, s) for x, z, s in zip(
        result.source['prepared_x'], result.source['prepared_z'], result.source['prepared_signs'])]
    assert prepared == ['-IXXI', 'ZIIZ']
    np.testing.assert_array_equal(result.leaf_signs, [1, 0])
    assert result.metadata['representation_action'] == 'canonicalized'
    signed_equivalence_check(result)
    assert result.validate()['valid']


def test_opposite_signed_duplicates_copy_one_class_without_new_readout():
    result = branching_deform(steane_code(), logical_paulis=['Y', '-Y', 'Y', '-Y'])
    assert result.k == 1 and result.metadata['target_logical_rank'] == 1
    assert len(result.leaf_x) == 4 and len(result.branches) == 6
    assert np.all((result.leaf_x | result.leaf_z).sum(axis=0) <= 1)
    signed_equivalence_check(result)
    # Opposite signed logical copies multiply to minus a stabilizer, not +I.
    px, pz, phase = pauli_product(result.leaf_x, result.leaf_z, result.leaf_signs, [0, 1])
    selected = solve(np.hstack((result.check_x, result.check_z)).T, np.r_[px, pz])
    sx, sz, stab_phase = pauli_product(result.check_x, result.check_z, result.check_signs,
                                      np.flatnonzero(selected))
    np.testing.assert_array_equal(px, sx)
    np.testing.assert_array_equal(pz, sz)
    assert (phase - stab_phase) % 4 == 2


@pytest.fixture(scope='module')
def valid_result():
    return branching_deform(four_qubit_code(), targets=['YZZY', 'ZIIZ'])


@pytest.mark.parametrize('damage', ['check', 'check_sign', 'basis_sign', 'leaf_sign',
                                   'leaf_vector', 'witness', 'descriptor', 'source_witness',
                                   'prepared_x_shape', 'prepared_z_shape'])
def test_validation_detects_material_certificate_corruption(valid_result, damage):
    result = deepcopy(valid_result)
    if damage == 'check':
        result.check_x[-1, 0] ^= 1
    elif damage == 'check_sign':
        result.check_signs[-1] ^= 1
    elif damage == 'basis_sign':
        result.logical_basis_signs[0] ^= 1
    elif damage == 'leaf_sign':
        result.leaf_signs[0] ^= 1
    elif damage == 'leaf_vector':
        result.leaf_x[0, 0] ^= 1
    elif damage == 'witness':
        result.equivalence_witnesses[0, 0] ^= 1
    elif damage == 'descriptor':
        result.branches[0]['F'][0, 0] ^= 1
    elif damage.startswith('prepared_'):
        key = damage.removesuffix('_shape')
        result.source[key] = np.vstack((result.source[key], result.source[key][0]))
    else:
        result.source['canonicalization_witnesses'][0, 0] ^= 1
    with pytest.raises(ValueError):
        result.validate()


def test_descriptive_metadata_is_not_a_validation_authority(valid_result):
    result = deepcopy(valid_result)
    result.metadata['validation'] = {'valid': False}
    result.metadata['static_distance_certificate'] = {'relation': 'unsupported'}
    assert result.validate()['valid']
    result.check_signs[-1] ^= 1
    result.metadata['validation'] = {'valid': True}
    with pytest.raises(ValueError):
        result.validate()


def test_budget_guard_and_bad_options():
    code = four_qubit_code()
    with pytest.raises(ValueError, match='max_entries|dense entries'):
        branching_deform(code, logical_paulis=['YI', 'IX'], max_entries=10)
    for kwargs in ({'force_copy': 1}, {'max_entries': True}, {'max_entries': 1.5},
                   {'max_entries': 0}, {'leaf_pauli': 'Y'}):
        with pytest.raises(ValueError):
            branching_deform(code, **kwargs)


@pytest.mark.parametrize('force_copy', [False, True])
def test_single_target_y_keeps_full_information(force_copy):
    result = branching_deform(steane_code(), logical_paulis='-Y', force_copy=force_copy)
    assert result.k == 1 and len(result.branches) == int(force_copy)
    assert result.summary()['tree_depth'] == int(force_copy)
    assert (result.n > result.original_n) == force_copy
    signed_equivalence_check(result)
    assert result.validate()['unintended_logical_readout_rank'] == 0
