"""BB90 disjoint Z2*X5 readout and the complete ten-qubit adapter.

Distance ten is conditional on the published input distance: the adapter
certificate supplies the lower bound and all ten bridge Xs give an explicit
nontrivial logical of weight ten. No numerical distance optimizer is needed.
"""
import numpy as np
import pytest

from codesurgery.examples import direct_sum, four_qubit_code, ibm_bb90_code
from codesurgery.explicit_gadgets import gadget_pair_deform
from codesurgery.gauging_measurement import gauging_pair_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.logical_optimization import reduce_logical_weight
from codesurgery.universal_adapters import repetition_adapter, skip_tree

from test_bb90_gauging import _literal_product


@pytest.fixture(scope="module")
def bb90_zx_inputs():
    code = ibm_bb90_code()
    left = reduce_logical_weight(code, pauli="Z", logical_index=1,
                                 method="heuristic")
    right = reduce_logical_weight(code, pauli="X", logical_index=4,
                                  method="heuristic")
    assert left["threshold"] == right["threshold"] == 20
    assert (left["before_weight"], left["after_weight"]) == (22, 10)
    assert (right["before_weight"], right["after_weight"]) == (10, 10)
    assert right["status"] == "skipped_threshold"
    assert not np.any(np.asarray(left["z"]) & np.asarray(right["x"]))
    return code, left, right


@pytest.fixture(scope="module", params=[0, 1], ids=["positive", "negative_left"])
def bb90_zx_result(bb90_zx_inputs, request):
    code, left, right = bb90_zx_inputs
    left = dict(left, sign=request.param)
    # Both ten-vertex factors are enumerated exactly. The adapter theorem
    # certifies the composite graph without enumerating its twenty vertices.
    result = gadget_pair_deform(code, left, right, adapter_shift=6,
                                require_distance=False, exact_limit=10)
    return code, left, right, result


def test_bb90_zx_actual_commutation_and_only_joint_readout(bb90_zx_result):
    code, left, right, result = bb90_zx_result
    cx, cz = result.check_x.astype(np.int64), result.check_z.astype(np.int64)
    assert cx.shape == cz.shape == (132, 130)
    assert not np.any((cx @ cz.T + cz @ cx.T) % 2)
    assert rank(np.hstack((cx, cz))) == 123
    assert result.k_stabilizer == result.protected_k == 7
    assert not result.is_css  # The Z-factor vertices also carry bridge Xs.
    assert int((cx | cz).sum(axis=1).max()) == 7
    assert int((cx | cz).sum(axis=0).max()) == 11

    # Determine all memory-only products from the actual auxiliary columns.
    cancel = nullspace(np.hstack((cx[:, 90:], cz[:, 90:])).T).astype(np.int64)
    readx, readz = (cancel @ cx[:, :90]) % 2, (cancel @ cz[:, :90]) % 2
    assert not np.any((readx @ code.hz.astype(np.int64).T) % 2)
    assert not np.any((readz @ code.hx.astype(np.int64).T) % 2)
    basis = code.standard_logicals()
    signatures = np.hstack(((readx @ basis.jz.astype(np.int64).T) % 2,
                            (readz @ basis.jx.astype(np.int64).T) % 2))
    x5, z2 = np.eye(16, dtype=np.uint8)[4], np.eye(16, dtype=np.uint8)[9]
    target = x5 ^ z2
    assert rank(signatures) == 1
    assert in_rowspace(target, signatures)
    assert in_rowspace(signatures, target[None, :])
    assert not in_rowspace(x5, signatures)
    assert not in_rowspace(z2, signatures)

    px, pz, phase = _literal_product(cx, cz, result.check_signs,
                                     np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(px, np.pad(right["x"], (0, 40)))
    np.testing.assert_array_equal(pz, np.pad(left["z"], (0, 40)))
    assert phase == 2 * left["sign"] == 2 * result.measurement_sign
    for dependency in nullspace(np.hstack((cx, cz)).T):
        dx, dz, dp = _literal_product(cx, cz, result.check_signs,
                                      np.flatnonzero(dependency))
        assert not np.any(dx | dz)
        assert dp == 0  # No dependent signed stabilizer product is -I.


def test_bb90_zx_rotated_skiptree_and_complete_cycle_space(bb90_zx_result):
    _, _, _, result = bb90_zx_result
    adapter = result.metadata["adapter"]
    factors = [info["restriction_matrix"] for info in result.metadata["factors"]]
    fl, fr = factors
    for f in factors:
        assert f.shape == (15, 10) and rank(f) == 9
        assert np.all(f.sum(axis=1) == 2)
        assert np.all(f.sum(axis=0) == 3)
    b, c = adapter["incidence"], adapter["cycle_basis"]
    assert b.shape == (40, 20) and rank(b) == 19
    assert c.shape == (22, 40) and rank(c) == 21
    assert not np.any((c.astype(np.int64) @ b.astype(np.int64)) % 2)
    assert adapter["adapter_size"] == 10 and adapter["adapter_shift"] == 6
    for side, f in (("left", fl), ("right", fr)):
        t, p = adapter["T_" + side], adapter["P_" + side]
        np.testing.assert_array_equal((t @ f @ p) % 2, adapter["H_C"])
        assert t.sum(axis=1).max() <= 3 and t.sum(axis=0).max() <= 2
    unshifted = repetition_adapter(fl, fr)
    np.testing.assert_array_equal(adapter["T_right"],
                                  np.roll(unshifted["T_right"], 6, axis=0))
    np.testing.assert_array_equal(adapter["P_right"],
                                  np.roll(unshifted["P_right"], 6, axis=1))
    connector = c[adapter["connector_cycle_rows"]]
    np.testing.assert_array_equal(connector,
        np.hstack((adapter["T_left"], adapter["H_C"], adapter["T_right"])))
    assert connector.sum(axis=1).max() == 6
    assert connector.sum(axis=0).max() <= 2


def test_bb90_zx_factors_leave_ten_bridge_xs_and_product_cancels(bb90_zx_result):
    _, left, right, result = bb90_zx_result
    cx, cz = result.check_x.astype(np.int64), result.check_z.astype(np.int64)
    bridge = 90 + np.asarray(result.metadata["adapter"]["edge_layout"]["adapter"])
    expected_bridge_x = np.zeros(130, dtype=np.int64)
    expected_bridge_x[bridge] = 1
    products = []
    for factor, selected in ((left, range(90, 100)), (right, range(100, 110))):
        x, z, phase = _literal_product(cx, cz, result.check_signs, selected)
        expected_x = expected_bridge_x.copy()
        expected_x[:90] = factor["x"]
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(z, np.pad(factor["z"], (0, 40)))
        products.append((x, z))
    assert not np.any((products[0][0] ^ products[1][0])[90:])
    assert not np.any((products[0][1] ^ products[1][1])[90:])

    # All ten bridge Xs form a surviving logical, not a measured stabilizer.
    witness = np.concatenate((expected_bridge_x, np.zeros(130, dtype=np.int64)))
    assert expected_bridge_x.sum() == 10
    assert not np.any((cz @ expected_bridge_x) % 2)
    assert not in_rowspace(witness, np.hstack((cx, cz)))
    assert result.metadata["adapter_distance_proof"]["condition_met"]
    assert result.metadata["distance_lower_bound"] == 10


def test_bb90_zx_one_bridge_has_a_weight_one_logical(bb90_zx_inputs):
    code, left, right = bb90_zx_inputs
    result = gadget_pair_deform(code, left, right, adapter_size=1,
                                require_distance=False, exact_limit=10)
    bridge = code.n + result.metadata["adapter"]["edge_layout"]["adapter"][0]
    x = np.zeros(result.n, dtype=np.uint8)
    x[bridge] = 1
    assert not np.any(result.check_z @ x % 2)
    assert not in_rowspace(np.concatenate((x, np.zeros_like(x))),
                           np.hstack((result.check_x, result.check_z)))
    assert not result.metadata["distance_preservation_guaranteed"]


@pytest.mark.parametrize("route", [gadget_pair_deform, gauging_pair_deform])
def test_pair_adapter_shift_zero_keeps_the_default_matrices(route):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    defaults = dict(left="ZZIIIIII", right="IIIIXXII", require_distance=False)
    original = route(code, **defaults)
    zero = route(code, **defaults, adapter_shift=0)
    for attribute in ("check_x", "check_z", "check_signs", "measurement_witness"):
        np.testing.assert_array_equal(getattr(original, attribute), getattr(zero, attribute))
    adapter = zero.metadata["adapter"]
    right_edge_rows = adapter["edge_layout"]["right"]
    left_vertex_count = adapter["P_left"].shape[0]
    unrotated_right = skip_tree(adapter["incidence"][right_edge_rows, left_vertex_count:])
    np.testing.assert_array_equal(adapter["T_right"], unrotated_right["T"])
    np.testing.assert_array_equal(adapter["P_right"], unrotated_right["P"])


@pytest.mark.parametrize("route", [gadget_pair_deform, gauging_pair_deform])
@pytest.mark.parametrize("shift", [-1, 2, 0.5, True, "0", None])
def test_pair_apis_reject_invalid_adapter_shifts(route, shift):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(ValueError, match="adapter_shift"):
        route(code, "ZZIIIIII", "IIIIXXII", adapter_size=2,
              adapter_shift=shift, require_distance=False)
