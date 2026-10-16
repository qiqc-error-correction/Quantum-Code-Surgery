"""Single explicit gadgets use even Tanner incidence; only adapters need edges."""
import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform, GaugeMeasurementRequired
from codesurgery.explicit_gadgets.expansion import exact_cheeger
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.symplectic import pauli_product


def assert_only_signed_target(code, result):
    """Derive the memory readout from the returned checks, independently."""
    assert result.validate()
    x, z = result.check_x, result.check_z
    assert not np.any((x @ z.T + z @ x.T) % 2)
    cancellation = nullspace(np.hstack([x[:, code.n:], z[:, code.n:]]).T)
    image = cancellation @ np.hstack([x[:, :code.n], z[:, :code.n]]) % 2
    stabilizers = np.block([[code.hx, np.zeros_like(code.hx)],
                           [np.zeros_like(code.hz), code.hz]])
    target = np.concatenate([result.measurement_x, result.measurement_z])
    assert rank(np.vstack([stabilizers, image])) == rank(stabilizers) + 1
    assert in_rowspace(image, np.vstack([stabilizers, target]))
    assert in_rowspace(target, image)
    for dependency in nullspace(np.hstack([x, z]).T):
        assert pauli_product(x, z, result.check_signs, np.flatnonzero(dependency))[2] == 0
    px, pz, phase = pauli_product(x, z, result.check_signs,
                                 np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(px[:code.n], result.measurement_x)
    np.testing.assert_array_equal(pz[:code.n], result.measurement_z)
    assert not np.any(px[code.n:] | pz[code.n:])
    assert phase == 2 * result.measurement_sign
    assert result.k_stabilizer == code.k - 1


def weight_four_code():
    return CSSCode([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0]],
                   np.zeros((0, 5), dtype=np.uint8), distance=1)


@pytest.mark.parametrize("target", ["ZZZZI", "-YYYYI"])
def test_single_even_weight_four_row_is_preserved(target):
    code = weight_four_code()
    result = gadget_deform(code, target, require_distance=False)
    assert_only_signed_target(code, result)
    f = result.metadata["restriction_matrix"]
    np.testing.assert_array_equal(f, code.hx[:, :4])
    np.testing.assert_array_equal(nullspace(f), [[1, 1, 1, 1]])
    assert result.metadata["cycle_basis"].shape == (0, 3)
    assert result.n == 8
    assert result.measurement_sign == target.startswith("-")
    # This is the parity boundary: a weight-four row does not count a cut
    # whenever that cut meets it on two vertices.
    assert exact_cheeger(f)["h"] == pytest.approx(0.5)
    assert result.metadata["relative_expansion"] is None
    assert result.metadata["boundary_expansion"]["h"] == pytest.approx(0.5)
    assert not result.metadata["distance_preservation_guaranteed"]


@pytest.mark.parametrize("target", ["ZZZZI", "-YYYYI"])
def test_the_same_weight_four_factor_cannot_enter_skiptree(target):
    with pytest.raises(GaugeMeasurementRequired, match="weight.*2"):
        gadget_pair_deform(weight_four_code(), target, "IIIIZ", require_distance=False)


def test_single_mixed_xz_and_y_target_with_weight_four_row():
    code = CSSCode([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0]], [[1, 1, 1, 1, 1, 0]], distance=1)
    result = gadget_deform(code, "-YZZZXI", require_distance=False)
    assert_only_signed_target(code, result)
    f = result.metadata["restriction_matrix"]
    np.testing.assert_array_equal(f, [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0],
                                     [1, 0, 1, 0, 0], [1, 0, 0, 0, 1]])
    assert rank(f) == 4
    assert result.n == 10 and result.measurement_sign == 1


def test_supported_stabilizers_allow_lower_rank_and_require_full_closure():
    # Two directions of ker F give existing Z stabilizers; only ZZZZI is new.
    # Redundant X checks are retained, producing a genuine closure generator.
    code = CSSCode([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]],
                   [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]])
    result = gadget_deform(code, "ZZZZI", require_distance=False)
    assert_only_signed_target(code, result)
    f, cycles = result.metadata["restriction_matrix"], result.metadata["cycle_basis"]
    assert rank(f) == 1 < f.shape[1] - 1
    assert rank(cycles) == len(f) - rank(f) == 1
    assert not np.any(cycles @ f % 2)
    assert exact_cheeger(f)["h"] == 0
    assert result.metadata["precheck"]["method"] == "kernel_signature"
    with pytest.raises(ValueError, match="cycle|closure|nullspace"):
        gadget_deform(code, "ZZZZI", cycle_basis=np.zeros((0, 2), dtype=np.uint8), require_distance=False)


def test_disconnected_supported_stabilizer_component_is_safe_below_2d():
    code = CSSCode([[1, 1, 1, 1, 0]],
                   [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1]], distance=2)
    result = gadget_deform(code, "ZZIIZ", require_distance=False)
    assert_only_signed_target(code, result)
    np.testing.assert_array_equal(result.metadata["restriction_matrix"], [[1, 1, 0]])
    assert result.metadata["precheck"]["method"] == "weight_lt_2d"
    assert not result.metadata["precheck"]["kernel_enumerated"]
    assert result.k_stabilizer == 1


def test_single_hyperedge_still_rejects_extra_logical_information():
    code = CSSCode([[1, 1, 1, 1, 0]], np.zeros((0, 5), dtype=np.uint8))
    with pytest.raises(GaugeMeasurementRequired, match="unintended") as error:
        gadget_deform(code, "ZZZZI", require_distance=False)
    assert error.value.details["logical_rank"] == 3


def test_default_and_supplied_logical_bases_keep_their_meaning():
    code = four_qubit_code()
    standard = code.standard_logicals()
    default = gadget_deform(code, require_distance=False)
    np.testing.assert_array_equal(default.measurement_z, standard.jz[0])
    assert not np.any(default.measurement_x)
    assert_only_signed_target(code, default)
    supplied = dict(jx=standard.jx[::-1], jz=standard.jz[::-1])
    custom = gadget_deform(code, pauli="Y", logical_basis=supplied, require_distance=False)
    np.testing.assert_array_equal(custom.measurement_x, supplied["jx"][0])
    np.testing.assert_array_equal(custom.measurement_z, supplied["jz"][0])
    assert_only_signed_target(code, custom)


def test_supplied_even_hyperedges_use_tanner_distance_certificate():
    code = weight_four_code()
    # Retain the entire raw F and append a second copy, including weight 4.
    # Every parity boundary doubles, so h rises from 1/2 to 1.
    result = gadget_deform(code, "-YYYYI", extra_rows=code.hx[:, :4], require_distance=False)
    assert_only_signed_target(code, result)
    np.testing.assert_array_equal(result.metadata["restriction_matrix"],
                                  np.vstack([code.hx[:, :4], code.hx[:, :4]]))
    assert result.metadata["boundary_expansion"]["h"] == 1
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] == code.distance
    assert rank(result.metadata["cycle_basis"]) == 3
