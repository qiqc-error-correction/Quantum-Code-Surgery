import json
import numpy as np
import pytest
from codesurgery import CSSCode, ckbb_deform, ckbb_batch, UnintendedLogicalMeasurementError
from codesurgery.ckbb import _build_css_primitive
from codesurgery.examples import four_qubit_code, steane_code, direct_sum
from codesurgery.gf2 import rank, in_rowspace, nullspace
from codesurgery.symplectic import pauli_product


@pytest.mark.parametrize("pauli", ["X", "Z"])
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_layer_matrices_and_witness(pauli, layers):
    code = four_qubit_code()
    result = ckbb_deform(code, pauli=pauli, layers=layers)
    assert result.validate()
    assert result.n == code.n + layers + 2 * (layers - 1)
    assert not np.any((result.hx @ result.hz.T) % 2)
    assert result.protected_k == 1
    assert result.metadata["total_auxiliary_tanner_layers"] == 2 * layers - 1
    assert result.metadata["distance_lower_bound"] == min(2, layers)


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_raw_disjoint_product_is_rejected_without_inventing_a_bridge(pauli):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    vector = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    for bridge in (False, True):
        with pytest.raises(UnintendedLogicalMeasurementError):
            ckbb_deform(code, vector, pauli=pauli, bridge=bridge)


@pytest.mark.parametrize("pauli", ["X", "Z"])
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_explicit_disjoint_factors_use_fig3_without_reading_factors(pauli, layers):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    factors = [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0]]
    result = ckbb_deform(code, logical_factors=factors, pauli=pauli, layers=layers)
    vector = np.array(factors[0]) ^ np.array(factors[1])
    assert np.array_equal(result.measurement_x if pauli == "X" else result.measurement_z, vector)
    assert result.metadata["logical_image_rank_before_bridges"] == 2
    assert result.metadata["logical_image_rank_after_bridges"] == 1
    assert result.metadata["bridge_pairs"]
    checks = result.hx if pauli == "X" else result.hz
    first_factor = np.pad(factors[0], (0, result.n - code.n))
    assert not in_rowspace(first_factor, checks)
    assert result.metadata["measurement_validation"]["logical_rank"] == 1
    assert result.validate()


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_fig3_signed_factor_product_and_explicit_target_agree(pauli):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    factor1 = "-" + pauli * 2 + "I" * 6
    factor2 = "I" * 4 + pauli * 2 + "I" * 2
    target = "-" + pauli * 2 + "I" * 2 + pauli * 2 + "I" * 2
    result = ckbb_deform(code, target, logical_factors=[factor1, factor2], layers=2)
    assert result.measurement_sign == 1
    assert result.validate()
    for dependency in nullspace(np.hstack((result.check_x, result.check_z)).T):
        x, z, phase = pauli_product(result.check_x, result.check_z,
                                    result.check_signs, np.flatnonzero(dependency))
        assert not np.any(x | z)
        assert phase == 0
    with pytest.raises(ValueError):
        ckbb_deform(code, target[1:], logical_factors=[factor1, factor2], layers=2)
    with pytest.raises(ValueError):
        ckbb_deform(code, logical_factors=[factor1, factor2], bridge=False)


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_fig3_rejects_overlapping_supports_and_unsafe_branch(pauli):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(ValueError):
        ckbb_deform(code, logical_factors=[pauli + "I" + pauli + "I" * 5,
                                         pauli + "II" + pauli + "I" * 4])
    three = direct_sum(four_qubit_code(), four_qubit_code(), four_qubit_code())
    # The first purported branch itself reads two independent parities.
    # A single Fig.3 connection to branch 2 cannot remove the remaining one.
    factors = [[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]]
    with pytest.raises(UnintendedLogicalMeasurementError):
        ckbb_deform(three, logical_factors=factors, pauli=pauli, layers=2)


def test_fig3_copies_a_shared_original_opposite_check_into_each_branch():
    code = CSSCode([[1, 1, 1, 1, 1, 1]], np.zeros((1, 6), dtype=np.uint8))
    result = ckbb_deform(code, logical_factors=["ZZIIII", "IIZZII"], layers=2)
    md = result.metadata
    assert [indices.tolist() for indices in md["factor_check_indices"]] == [[0], [0]]
    np.testing.assert_array_equal(md["restriction_before_bridges"],
                                  [[1, 1, 0, 0], [0, 0, 1, 1]])
    np.testing.assert_array_equal(md["pasting_matrix"][:, :2], [[1, 1]])
    assert len(md["restriction_matrix"]) == 3
    assert md["measurement_validation"]["logical_rank"] == 1
    assert result.validate()


def test_fig3_requires_two_or_more_factors_of_one_pure_type():
    code = CSSCode(np.zeros((0, 3)), np.zeros((0, 3)))
    for factors in (["ZII"], ["XII", "IZI"], ["YII", "IZI"]):
        with pytest.raises(ValueError):
            ckbb_deform(code, logical_factors=factors, layers=1)


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_gauge_cycles_and_redundant_checks(pauli):
    code = direct_sum(steane_code(), steane_code())
    # Retain duplicated and zero check rows, rather than changing the user's LDPC presentation.
    code = CSSCode(np.vstack((code.hx, code.hx[0], np.zeros(code.n))),
                   np.vstack((code.hz, code.hz[0], np.zeros(code.n))), distance=3)
    vector = np.array([1, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=np.uint8)
    result = ckbb_deform(code, vector, pauli=pauli)
    assert len(result.gauge_x) == result.gauge_qubits
    assert result.gauge_qubits >= 1
    assert np.array_equal((result.gauge_x @ result.gauge_z.T) % 2,
                          np.eye(result.gauge_qubits))
    assert not np.any((result.gauge_x @ result.hz.T) % 2)
    assert not np.any((result.gauge_z @ result.hx.T) % 2)
    assert result.protected_k == code.k - 1


def test_no_distance_uses_conservative_n_layers():
    code = CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]])
    result = ckbb_deform(code)
    assert result.metadata["layers"] == 4
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] is None


def test_raw_mixed_product_with_extra_readout_is_rejected():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(UnintendedLogicalMeasurementError):
        ckbb_deform(code, "YYIIYYII")


def test_mixed_overlap_is_allowed_but_disjoint_connector_requires_bridge():
    assert ckbb_deform(four_qubit_code(), "YYII", bridge=False).validate()
    bare = CSSCode(np.zeros((0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        ckbb_deform(bare, "ZX", layers=1, bridge=False)
    assert ckbb_deform(bare, "ZX", layers=1).validate()


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_one_logical_class_does_not_imply_irreducible_support(pauli):
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]], distance=1)
    result = ckbb_deform(code, [1, 1, 1], pauli=pauli, bridge=False)
    assert result.metadata["logical_image_rank_before_bridges"] == 1
    assert result.metadata["support_kernel_dimension_before_completion"] == 2
    assert not result.metadata["irreducible_support_before_completion"]
    assert result.metadata["distance_lower_bound"] is None
    assert not result.metadata["distance_preservation_guaranteed"]
    assert result.validate()


def test_default_and_independent_batch():
    code = four_qubit_code()
    results = ckbb_batch(code)
    assert len(results) == 2
    assert np.array_equal(ckbb_deform(code).measurement_z, code.logical())
    assert all(r.protected_k == 1 for r in results)


def test_default_y_uses_one_paired_logical_index():
    code = four_qubit_code()
    result = ckbb_deform(code, pauli="Y", logical_index=1)
    expected = np.zeros(2 * code.k, dtype=np.uint8)
    expected[1] = expected[code.k + 1] = 1
    np.testing.assert_array_equal(result.metadata["measurement_validation"]["target_signature"], expected)
    assert result.metadata["measurement_validation"]["logical_rank"] == 1
    assert result.validate()
    assert len(ckbb_batch(code, pauli="Y")) == code.k


def test_invalid_inputs_and_memory_limit():
    code = four_qubit_code()
    for operator in ([0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1], [2, 0, 0, 0]):
        with pytest.raises(ValueError):
            ckbb_deform(code, operator)
    for layers in (0, 1.5, True):
        with pytest.raises(ValueError):
            ckbb_deform(code, layers=layers)
    with pytest.raises(ValueError, match="max_entries"):
        ckbb_deform(code, max_entries=1)
    with pytest.raises(ValueError, match="Invalid CSS"):
        CSSCode([[1, 0]], [[1, 0]])


def test_export_roundtrip(tmp_path):
    result = ckbb_deform(four_qubit_code())
    result.save(tmp_path / "code.npz")
    result.save(tmp_path / "code.json")
    with np.load(tmp_path / "code.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["hx"], result.hx)
        assert json.loads(str(saved["information"]))["summary"]["deformed_n"] == result.n
    assert json.loads((tmp_path / "code.json").read_text())["summary"]["protected_k"] == 1
