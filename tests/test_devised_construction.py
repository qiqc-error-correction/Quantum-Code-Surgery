"""Direct equation checks and negative readout/export/CLI cases."""

import json

import numpy as np
import pytest

from codesurgery import CSSCode
from codesurgery.devised_sticking import DevisedReadoutError, devised_deform, measurement_sticker
from codesurgery.devised_sticking.cli import main
from codesurgery.devised_sticking.construction import assemble_deformed, validate_readout
from codesurgery.examples import four_qubit_code


@pytest.mark.parametrize("d_r", [1, 2, 3])
def test_measurement_sticker_matches_kronecker_equations(d_r):
    h_g = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    repetition = np.zeros((d_r - 1, d_r), dtype=np.uint8)
    for row in range(d_r - 1):
        repetition[row, row:row + 2] = 1
    expected_x = np.hstack((np.kron(np.eye(d_r - 1, dtype=np.uint8), h_g),
                            np.kron(repetition, np.eye(len(h_g), dtype=np.uint8))))
    expected_z = np.hstack((np.kron(repetition.T, np.eye(h_g.shape[1], dtype=np.uint8)),
                            np.kron(np.eye(d_r, dtype=np.uint8), h_g.T)))
    actual_x, actual_z = measurement_sticker(h_g, d_r)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_z, expected_z)


def test_pasting_and_signed_input_do_not_change_original_memory_checks():
    code = four_qubit_code()
    result = devised_deform(code, ["-ZZII", "ZIZI"], d_R=2)
    sticker_x, sticker_z = measurement_sticker(result.glue.h_g, 2)
    np.testing.assert_array_equal(result.hx[len(code.hx):, code.n:], sticker_x)
    np.testing.assert_array_equal(result.hz[len(code.hz):, code.n:], sticker_z)
    np.testing.assert_array_equal(result.hx[:len(code.hx), :code.n], code.hx)
    np.testing.assert_array_equal(result.hz[:len(code.hz), :code.n], code.hz)
    memory_rows = list(range(len(code.hx))) + list(range(len(result.hx), len(result.hx) + len(code.hz)))
    assert not np.any(result.check_signs[memory_rows])


def test_final_validator_rejects_extra_and_wrong_logical_space():
    code = CSSCode(np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8))
    empty = np.zeros((0, 2), dtype=np.uint8)
    with pytest.raises(DevisedReadoutError) as extra:
        validate_readout(code, empty, np.eye(2, dtype=np.uint8), [[1, 1]], pauli="Z")
    assert extra.value.logical_rank == 2
    with pytest.raises(DevisedReadoutError) as wrong:
        validate_readout(code, empty, [[0, 1]], [[1, 0]], pauli="Z")
    assert wrong.value.logical_rank == 1
    with pytest.raises(ValueError, match="Incompatible glue"):
        assemble_deformed(four_qubit_code(), {"h_g": [[1, 0]], "s": [[1, 0, 0, 0], [0, 1, 0, 0]], "t": [[1]]}, d_R=1)


def test_invalid_mutated_result_does_not_overwrite_existing_export(tmp_path):
    result = devised_deform(four_qubit_code(), d_R=2)
    result.measurement_z[0] = [1, 1, 0, 0]
    destination = tmp_path / "result.json"
    destination.write_text("keep existing artifact")
    with pytest.raises(ValueError):
        result.save(destination)
    assert destination.read_text() == "keep existing artifact"


def test_mutated_glue_is_rejected_before_export(tmp_path):
    result = devised_deform(four_qubit_code(), d_R=2)
    result.glue.s[0] = [0, 1, 0, 0]
    # This retains compatibility for the all-ones original X check, but
    # changes the actual boundary represented by the exported S matrix.
    with pytest.raises(ValueError, match="glue does not match"):
        result.save(tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_separate_cli_writes_result_and_rejects_invalid_input_before_export(tmp_path):
    source = tmp_path / "input.json"
    destination = tmp_path / "devised.json"
    config = {"hx": [[1, 1, 1, 1]], "hz": [[1, 1, 1, 1]], "distance": 2,
              "logicals": ["ZZII", "ZIZI"]}
    source.write_text(json.dumps(config))
    assert main([str(source), "-o", str(destination)]) == 0
    saved = json.loads(destination.read_text())
    assert saved["summary"]["scheme"] == "zhang_li_devised_sticking"
    assert saved["summary"]["logical_measurement_rank"] == 2
    before = destination.read_bytes()
    config["logicals"] = ["ZZII", "IIZZ"]
    source.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as error:
        main([str(source), "-o", str(destination)])
    assert error.value.code == 2
    assert destination.read_bytes() == before


def test_resource_guard_catches_redundant_check_coefficient_space():
    code = CSSCode(np.zeros((30, 1), dtype=np.uint8), np.zeros((0, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="max_entries"):
        devised_deform(code, "Z", d_R=1, max_entries=1000)


def test_standard_form_dressing_attains_optimized_count():
    code = CSSCode(np.zeros((0, 6), dtype=np.uint8), np.zeros((0, 6), dtype=np.uint8))
    targets = [[1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 1]]
    result = devised_deform(code, targets, d_R=1)
    metadata = result.glue.metadata
    assert metadata["canonical_standard_form_representatives"]
    assert metadata["optimized_standard_form_size_bound_claimed"]
    assert metadata["dressing_nonzeros"] <= metadata["redundancy_number"] * (len(targets) + 1)
    assert result.validate()
