"""Branching preserves signed logical information, including mixed axes."""
import numpy as np
import pytest
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.models import CSSCode
from codesurgery.gf2 import rank, nullspace, in_rowspace
from codesurgery.symplectic import parse_pauli, pauli_product, commutation_matrix
from codesurgery.parallel_branching import construction as core


def checks(words):
    rows = [parse_pauli(word, len(words[0].lstrip('+-'))) for word in words]
    return np.asarray([r[0] for r in rows]), np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])


def css_checks(code):
    x = np.vstack((code.hx, np.zeros_like(code.hz)))
    z = np.vstack((np.zeros_like(code.hx), code.hz))
    return x, z, np.zeros(len(x), np.uint8)


def hamming():
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)] for bit in (3, 2, 1, 0)], np.uint8)
    return CSSCode(h, h, distance=3)


def audit_branch(branch, x, z, signs):
    bx, bz, bs = branch['check_x'], branch['check_z'], branch['check_signs']
    old_rank = rank(np.hstack((x, z)))
    assert not np.any(commutation_matrix(bx, bz))
    assert core._signed_check_rank(bx, bz, bs) == old_rank + branch['rank_increment']
    assert bx.shape[1] - rank(np.hstack((bx, bz))) == x.shape[1] - old_rank
    np.testing.assert_array_equal(bs[:len(signs)], signs)
    full_f = ((x[:, branch['support']] * branch['local_z'][branch['support']])
              ^ (z[:, branch['support']] * branch['local_x'][branch['support']]))
    np.testing.assert_array_equal((branch['T'] @ branch['F']) % 2, full_f)
    rx, rz = core.retract_paulis(bx, bz, branch)
    assert all(in_rowspace(row, np.hstack((x, z))) for row in np.hstack((rx, rz)))


@pytest.mark.parametrize('local', ['ZZZZ', 'XXXX', 'YYII', 'XYZI'])
@pytest.mark.parametrize('leaf', ['X', 'Z'])
def test_signed_mixed_sticker_commutes_and_preserves_k(local, leaf):
    x, z, signs = checks(['-XXXX', 'ZZZZ'])
    px, pz, _ = parse_pauli(local, 4)
    saved = [a.copy() for a in (x, z, signs, px, pz)]
    b = core.append_branch(x, z, signs, px, pz, leaf_pauli=leaf)
    audit_branch(b, x, z, signs)
    for original, copy in zip((x, z, signs, px, pz), saved):
        np.testing.assert_array_equal(original, copy)
    lx, lz, ls = core.lift_paulis(x, z, signs, b)
    for i in range(len(x)):
        selected = [i] + [int(b['cap_rows'][j]) for j in np.flatnonzero(b['T'][i])]
        xx, zz, phase = pauli_product(b['check_x'], b['check_z'], b['check_signs'], selected)
        np.testing.assert_array_equal(xx, lx[i]); np.testing.assert_array_equal(zz, lz[i])
        assert phase == 2 * int(ls[i])


@pytest.mark.parametrize('code', [four_qubit_code(), steane_code(), hamming()])
@pytest.mark.parametrize('leaf', ['X', 'Z'])
def test_full_logical_mapping_and_weight_nonincreasing_retraction(code, leaf):
    x, z, signs = css_checks(code)
    basis = code.standard_logicals()
    px, pz = basis.jx[0], basis.jz[0]
    b = core.append_branch(x, z, signs, px, pz, leaf_pauli=leaf)
    audit_branch(b, x, z, signs)
    ox = np.vstack((basis.jx, np.zeros_like(basis.jz)))
    oz = np.vstack((np.zeros_like(basis.jx), basis.jz))
    os = np.arange(2 * code.k, dtype=np.uint8) % 2
    lx, lz, ls = core.lift_paulis(ox, oz, os, b)
    assert not np.any((lx @ b['check_z'].T + lz @ b['check_x'].T) % 2)
    np.testing.assert_array_equal(commutation_matrix(lx, lz), commutation_matrix(ox, oz))
    np.testing.assert_array_equal(ls, os)
    assert rank(np.vstack((np.hstack((b['check_x'], b['check_z'])), np.hstack((lx, lz))))) == b['check_x'].shape[1] + code.k
    rx, rz = core.retract_paulis(lx, lz, b)
    np.testing.assert_array_equal(rx, ox); np.testing.assert_array_equal(rz, oz)
    normalizer = nullspace(np.hstack((b['check_z'], b['check_x'])))
    nn = b['check_x'].shape[1]
    nx, nz = normalizer[:, :nn], normalizer[:, nn:]
    rx, rz = core.retract_paulis(nx, nz, b)
    assert not np.any((rx @ z.T + rz @ x.T) % 2)
    assert np.all(np.count_nonzero(rx | rz, axis=1) <= np.count_nonzero(nx | nz, axis=1))
    old_checks, new_checks = np.hstack((x, z)), np.hstack((b['check_x'], b['check_z']))
    for nr, rr in zip(normalizer, np.hstack((rx, rz))):
        assert in_rowspace(nr, new_checks) == in_rowspace(rr, old_checks)


@pytest.mark.parametrize('leaf', ['X', 'Z'])
def test_signed_y_target_cleans_to_leaf(leaf):
    x, z, signs = checks(['XXXX', 'ZZZZ'])
    tx, tz, target_sign = parse_pauli('-YYII', 4)
    b = core.append_branch(x, z, signs, tx, tz, leaf_pauli=leaf)
    lx, lz, ls = core.lift_paulis(tx, tz, target_sign, b)
    assert not np.any(lx[4:] | lz[4:])
    rows_x = np.vstack((lx, b['check_x'][b['junction_rows']]))
    rows_z = np.vstack((lz, b['check_z'][b['junction_rows']]))
    sx, sz, phase = pauli_product(rows_x, rows_z, [ls] + [0] * len(b['support']))
    expected = np.zeros(b['check_x'].shape[1], np.uint8)
    expected[b['leaf_columns']] = 1
    np.testing.assert_array_equal(sx if leaf == 'X' else sz, expected)
    assert not np.any(sz if leaf == 'X' else sx)
    assert phase == 2 * target_sign
    assert not in_rowspace(np.r_[sx, sz], np.hstack((b['check_x'], b['check_z'])))


@pytest.mark.parametrize('pauli', ['X', 'Z'])
def test_css_dual_output_for_pure_axis(pauli):
    x, z, signs = css_checks(four_qubit_code())
    px, pz, _ = parse_pauli(pauli * 2 + 'II', 4)
    b = core.append_branch(x, z, signs, px, pz, leaf_pauli=pauli)
    assert not np.any(np.any(b['check_x'], axis=1) & np.any(b['check_z'], axis=1))


def test_redundant_signed_checks_retained_and_consistent():
    x, z, signs = checks(['XX', 'ZZ', '-YY', 'II'])
    px, pz, _ = parse_pauli('YI', 2)
    b = core.append_branch(x, z, signs, px, pz)
    assert b['old_checks'] == 4
    audit_branch(b, x, z, signs)
    np.testing.assert_array_equal(b['check_x'][:4, :2], x)
    np.testing.assert_array_equal(b['check_z'][:4, :2], z)


@pytest.mark.parametrize('words', [['X', 'Z'], ['-II'], ['XX', 'ZZ', 'YY'], ['XX', '-XX']])
def test_invalid_input_commutation_or_minus_identity_rejected(words):
    x, z, signs = checks(words)
    px = np.ones(x.shape[1], np.uint8)
    with pytest.raises(ValueError, match='commute|contains -I'):
        core.append_branch(x, z, signs, px, np.zeros_like(px))


def test_no_incident_checks_and_no_old_checks():
    x, z, signs = checks(['ZI'])
    b = core.append_branch(x, z, signs, [0, 0], [1, 0])
    assert b['F'].shape == (0, 1) and b['T'].shape == (1, 0)
    audit_branch(b, x, z, signs)
    x = z = np.zeros((0, 2), np.uint8)
    b = core.append_branch(x, z, [], [1, 0], [1, 0])
    audit_branch(b, x, z, [])


def test_counterexample_lift_does_not_falsely_claim_disjoint_support():
    x, z, signs = checks(['XXXXXXXX', 'ZZZZZZZZ'])
    px, pz, _ = parse_pauli('XXXXIIII', 8)
    qx, qz, _ = parse_pauli('ZZIIZZII', 8)
    b = core.append_branch(x, z, signs, px, pz, leaf_pauli='X')
    bare_x, bare_z = np.pad(qx, (0, b['check_x'].shape[1]-8)), np.pad(qz, (0, b['check_x'].shape[1]-8))
    assert np.any((bare_x @ b['check_z'].T + bare_z @ b['check_x'].T) % 2)
    lx, lz, _ = core.lift_paulis(qx, qz, 0, b)
    assert not np.any((lx @ b['check_z'].T + lz @ b['check_x'].T) % 2)
    assert np.count_nonzero((lx | lz)[b['leaf_columns']]) == 2
    rx, rz = core.retract_paulis(lx, lz, b)
    np.testing.assert_array_equal(rx, qx); np.testing.assert_array_equal(rz, qz)


def test_budget_prevents_new_dense_matrix_allocation(monkeypatch):
    x, z, signs = css_checks(four_qubit_code())
    original = core.np.zeros
    allocations = []
    def guarded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            allocations.append(shape)
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(core.np, 'zeros', guarded)
    with pytest.raises(ValueError, match='dense entries'):
        core.append_branch(x, z, signs, [0]*4, [1]*4, max_entries=1)
    assert not allocations


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, None])
def test_invalid_budget(budget):
    x, z, signs = css_checks(four_qubit_code())
    with pytest.raises(ValueError, match='max_entries'):
        core.append_branch(x, z, signs, [0]*4, [1]*4, max_entries=budget)


def test_empty_support_invalid_shapes_and_nonbinary_inputs():
    x, z, signs = css_checks(four_qubit_code())
    with pytest.raises(ValueError, match='empty'):
        core.append_branch(x, z, signs, [0]*4, [0]*4)
    with pytest.raises(ValueError):
        core.append_branch(x, z, signs, [0]*3, [1]*4)
    with pytest.raises(ValueError):
        core.append_branch(x, z, signs, [0]*4, [2]*4)
    with pytest.raises(ValueError):
        core.append_branch(x, z, signs, [0]*4, [1]*4, leaf_pauli='Y')
    with pytest.raises(ValueError):
        core.append_branch(x, z[:, :3], signs, [0]*4, [1]*4)


def test_empty_operator_batches_preserve_shape():
    x, z, signs = css_checks(four_qubit_code())
    b = core.append_branch(x, z, signs, [0]*4, [1]*4)
    lx, lz, ls = core.lift_paulis(np.zeros((0,4)), np.zeros((0,4)), [], b)
    assert lx.shape == lz.shape == (0,b['check_x'].shape[1]) and ls.shape == (0,)
    rx, rz = core.retract_paulis(lx, lz, b)
    assert rx.shape == rz.shape == (0,4)


@pytest.mark.parametrize('leaf', ['X', 'Z'])
def test_noncss_eight_qubit_example_mixed_targets_and_full_quotient(leaf):
    # Appendix B.2 of arXiv:2503.05003v3: a genuinely non-CSS [[8,3,3]].
    def bits(words):
        return np.asarray([[int(bit) for bit in word] for word in words], np.uint8)
    x = bits(['11111111', '00000000', '01011010', '01010101', '01101001'])
    z = bits(['00000000', '11111111', '00001111', '00110011', '01010101'])
    ox = bits(['11000000', '10100000', '10001000', '00000000', '00000000', '00000000'])
    oz = bits(['00000101', '00010010', '00010100', '01010101', '00110011', '00001111'])
    signs = np.zeros(5, np.uint8)
    # Physical representatives of X1, Z2, Y3 have one common local axis/site.
    tx = np.asarray([ox[0], ox[4], ox[2] ^ ox[5]])
    tz = np.asarray([oz[0], oz[4], oz[2] ^ oz[5]])
    target_signs = np.array([0, 1, 1], np.uint8)
    for i in range(3):
        for j in range(i):
            assert not np.any((tx[i] * tz[j]) ^ (tz[i] * tx[j]))
    px, pz = np.bitwise_or.reduce(tx), np.bitwise_or.reduce(tz)
    b = core.append_branch(x, z, signs, px, pz, leaf_pauli=leaf)
    audit_branch(b, x, z, signs)
    lx, lz, _ = core.lift_paulis(ox, oz, np.zeros(6, np.uint8), b)
    assert not np.any((lx @ b['check_z'].T + lz @ b['check_x'].T) % 2)
    np.testing.assert_array_equal(commutation_matrix(lx, lz), commutation_matrix(ox, oz))
    assert rank(np.vstack((np.hstack((b['check_x'], b['check_z'])), np.hstack((lx, lz))))) == b['check_x'].shape[1] + 3
    ltx, ltz, lts = core.lift_paulis(tx, tz, target_signs, b)
    assert not np.any(ltx[:, 8:] | ltz[:, 8:])
    for i in range(3):
        mask = (tx[i] | tz[i])[b['support']].astype(bool)
        selected = b['junction_rows'][mask]
        sx, sz, phase = pauli_product(
            np.vstack((ltx[i], b['check_x'][selected])),
            np.vstack((ltz[i], b['check_z'][selected])),
            np.r_[lts[i], b['check_signs'][selected]])
        expected = np.zeros(b['check_x'].shape[1], np.uint8)
        expected[b['leaf_columns'][mask]] = 1
        np.testing.assert_array_equal(sx if leaf == 'X' else sz, expected)
        assert not np.any(sz if leaf == 'X' else sx)
        assert phase == 2 * int(target_signs[i])
    # No new check combination can accidentally read any old logical class.
    rx, rz = core.retract_paulis(b['check_x'], b['check_z'], b)
    assert not np.any((rx @ oz.T + rz @ ox.T) % 2)
