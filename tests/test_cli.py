"""CLI rejection must happen before exporting a deformed-code artifact."""

import json

import pytest

from codesurgery.cli import main
from codesurgery.examples import direct_sum, four_qubit_code


def write_input(path, code, **extra):
    path.write_text(json.dumps({"hx": code.hx.tolist(), "hz": code.hz.tolist(),
                                "distance": code.distance, **extra}))


def test_unsafe_raw_target_does_not_overwrite_existing_output(tmp_path, capsys):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    source, output = tmp_path / "input.json", tmp_path / "result.json"
    write_input(source, code, logical="ZZIIZZII")
    output.write_text("keep this existing artifact")
    with pytest.raises(SystemExit) as error:
        main([str(source), "-o", str(output)])
    assert error.value.code == 2
    assert "2 independent logical observables" in capsys.readouterr().err
    assert output.read_text() == "keep this existing artifact"
    assert not (tmp_path / "result_logical_basis.json").exists()


def test_json_explicit_factors_export_only_the_requested_product(tmp_path):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    source, output = tmp_path / "input.json", tmp_path / "result.json"
    write_input(source, code, logical_factors=["ZZIIIIII", "IIIIZZII"])
    assert main([str(source), "-o", str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved["summary"]["measurement"] == "ZZIIZZII"
    assert saved["summary"]["logical_measurement_rank"] == 1
    assert saved["metadata"]["measurement_validation"]["validated_against_actual_deformed_checks"]


def test_default_logical_y_is_available_from_cli(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "y.json"
    write_input(source, four_qubit_code())
    assert main([str(source), "--pauli", "Y", "--no-bridge", "-o", str(output)]) == 0
    saved = json.loads(output.read_text())
    assert saved["summary"]["is_css"] is False
    assert saved["metadata"]["measurement_validation"]["target_signature"] == [1, 0, 1, 0]
    assert saved["summary"]["logical_measurement_rank"] == 1
