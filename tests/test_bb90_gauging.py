"""BB90 matrix/export checks; these tests do not recompute distance ten."""
import json

import numpy as np
import pytest

from codesurgery.examples import ibm_bb90_code
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank


def _coordinate_checks():
    """Evaluate the published monomials directly, without Kronecker products."""
    a = np.zeros((45, 45), dtype=np.int64)
    b = np.zeros_like(a)
    for i in range(15):
        for j in range(3):
            row = 3 * i + j
            for di, dj in ((9, 0), (0, 1), (0, 2)):
                a[row, 3 * ((i + di) % 15) + (j + dj) % 3] ^= 1
            for di, dj in ((0, 0), (2, 0), (7, 0)):
                b[row, 3 * ((i + di) % 15) + (j + dj) % 3] ^= 1
    return np.hstack((a, b)), np.hstack((b.T, a.T))


def _literal_product(cx, cz, signs, selected):
    """Independent tensor-Pauli multiplication using the one-qubit table.

    Labels 0, 1, 2, 3 are I, X, Z, Y. Product labels are their XOR;
    the table records the exponent of i, including XY=iZ and XZ=-iY.
    """
    local_phase = np.array([[0, 0, 0, 0], [0, 0, 3, 1],
                            [0, 1, 0, 3], [0, 3, 1, 0]], dtype=np.int64)
    labels = np.zeros(cx.shape[1], dtype=np.int64)
    phase = 0
    for index in selected:
        following = cx[index] + 2 * cz[index]
        phase = (phase + 2 * int(signs[index])
                 + int(local_phase[labels, following].sum())) % 4
        labels ^= following
    return labels & 1, (labels >> 1) & 1, phase


def _verify_actual_matrices(code, payload, pauli):
    """Recheck exported arrays, without trusting validation metadata."""
    cx, cz = (np.asarray(payload[key], dtype=np.int64)
              for key in ("check_x", "check_z"))
    assert cx.shape == cz.shape
    assert not np.any((cx @ cz.T + cz @ cx.T) % 2)
    assert cx.shape[1] - rank(np.hstack((cx, cz))) == 7

    # The memory part of each dressed check must still be its input check.
    oldx = np.vstack((code.hx, np.zeros_like(code.hz))).astype(np.int64)
    oldz = np.vstack((np.zeros_like(code.hx), code.hz)).astype(np.int64)
    np.testing.assert_array_equal(cx[:90, :90], oldx)
    np.testing.assert_array_equal(cz[:90, :90], oldz)

    # Eliminate all auxiliary support, then compute the complete readout space.
    cancel = nullspace(np.hstack((cx[:, 90:], cz[:, 90:])).T).astype(np.int64)
    readx, readz = (cancel @ cx[:, :90]) % 2, (cancel @ cz[:, :90]) % 2
    assert not np.any((readx @ code.hz.astype(np.int64).T) % 2)
    assert not np.any((readz @ code.hx.astype(np.int64).T) % 2)
    basis = code.standard_logicals()
    signatures = np.hstack(((readx @ basis.jz.astype(np.int64).T) % 2,
                            (readz @ basis.jx.astype(np.int64).T) % 2))
    expected = np.zeros(16, dtype=np.uint8)
    if pauli in "XY":
        expected[0] = 1
    if pauli in "ZY":
        expected[8] = 1
    assert rank(signatures) == 1
    assert in_rowspace(expected, signatures)
    assert in_rowspace(signatures, expected[None, :])

    # A full Pauli multiplication independently checks the sign and Y phase.
    signs = np.asarray(payload["check_signs"], dtype=np.int64)
    witness = np.flatnonzero(payload["measurement_witness"])
    x, z, phase = _literal_product(cx, cz, signs, witness)
    expected_x = basis.jx[0] if pauli in "XY" else np.zeros(90, dtype=np.uint8)
    expected_z = basis.jz[0] if pauli in "ZY" else np.zeros(90, dtype=np.uint8)
    np.testing.assert_array_equal(payload["measurement_x"], expected_x)
    np.testing.assert_array_equal(payload["measurement_z"], expected_z)
    np.testing.assert_array_equal(x, np.pad(expected_x, (0, cx.shape[1] - 90)))
    np.testing.assert_array_equal(z, np.pad(expected_z, (0, cx.shape[1] - 90)))
    assert phase == 2 * payload["measurement_sign"]
    for dependency in nullspace(np.hstack((cx, cz)).T):
        depx, depz, depphase = _literal_product(cx, cz, signs, np.flatnonzero(dependency))
        assert not np.any(depx | depz)
        assert depphase == 0  # No signed product equals -I.


def test_bb90_published_checks_and_standard_logical_pairing():
    code = ibm_bb90_code()
    expected_x, expected_z = _coordinate_checks()
    np.testing.assert_array_equal(code.hx, expected_x)
    np.testing.assert_array_equal(code.hz, expected_z)
    assert code.n == 90 and code.k == 8
    assert rank(code.hx) == rank(code.hz) == 41
    for matrix in (code.hx, code.hz):
        assert matrix.shape == (45, 90)
        assert np.all(matrix.sum(axis=1) == 6)
        assert np.all(matrix.sum(axis=0) == 3)
    assert not np.any((expected_x @ expected_z.T) % 2)
    basis = code.standard_logicals()
    jx, jz = basis.jx.astype(np.int64), basis.jz.astype(np.int64)
    np.testing.assert_array_equal((jx @ jz.T) % 2, np.eye(8, dtype=np.int64))
    assert not np.any((jx @ expected_z.T) % 2)
    assert not np.any((jz @ expected_x.T) % 2)


@pytest.fixture(scope="module", params=[("Z", 126), ("X", 111), ("Y", 147)])
def bb90_gadget(request):
    pauli, expected_n = request.param
    code = ibm_bb90_code()
    result = gauging_deform(code, pauli=pauli, logical_index=0,
                            require_distance=False, exact_limit=1)
    assert result.n == expected_n
    return code, result, pauli


def test_bb90_basic_gauging_commutes_and_reads_only_target(bb90_gadget):
    code, result, pauli = bb90_gadget
    assert result.validate()
    assert result.k_stabilizer == result.protected_k == 7
    assert result.is_css == (pauli != "Y")
    _verify_actual_matrices(code, vars(result), pauli)
    # Standard form has a single X/Z overlap, hence i*Xbar*Zbar is +literal Y.
    assert result.measurement_sign == 0


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_bb90_export_roundtrip_rechecks_full_matrices(bb90_gadget, tmp_path, suffix):
    code, result, pauli = bb90_gadget
    path = tmp_path / (f"bb90_{pauli}" + suffix)
    result.save(path)
    if suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key].copy() for key in saved.files if key != "information"}
            information = json.loads(saved["information"].item())
        payload["measurement_sign"] = information["measurement_sign"]
    for key in ("check_x", "check_z", "check_signs", "measurement_witness"):
        np.testing.assert_array_equal(payload[key], getattr(result, key))
    _verify_actual_matrices(code, payload, pauli)


def test_bb90_negative_y_readout_keeps_explicit_sign():
    code = ibm_bb90_code()
    basis = code.standard_logicals()
    target = dict(x=basis.jx[0], z=basis.jz[0], sign=1)
    result = gauging_deform(code, target, require_distance=False, exact_limit=1)
    assert result.n == 147 and result.measurement_sign == 1
    _verify_actual_matrices(code, vars(result), "Y")
