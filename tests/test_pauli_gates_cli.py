"""Command-line contracts and actual small-code gadget exports."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.pauli_gates import cli


def _document(output):
    return json.loads((output / "plan.json").read_text())


def _measurements(document):
    return [item for item in document["operations"] if item["kind"] == "measure"]


def test_bb90_decomposition_keeps_original_labels_and_full_joint_observables(tmp_path, capsys):
    output = tmp_path / "bb90"
    assert cli.main(["--example", "bb90", "--gate", "H:1", "--gate", "S:1",
                     "--gate", "CNOT:1:2", "--decompose-only", "-o", str(output)]) == 0
    data = _document(output)
    assert data["summary"]["k"] == 8
    assert data["summary"]["ancilla"] == 0
    assert data["summary"]["measurement_count"] == 11
    assert data["summary"]["compiled_measurements"] == 0
    assert data["deformed_code_files"] == {}
    measurements = _measurements(data)
    assert [item["logical_pauli"] for item in measurements] == [
        "ZIIIIIII", "XXIIIIII", "XZIIIIII", "ZIIIIIII",
        "XIIIIIII", "YZIIIIII", "ZIIIIIII",
        "XIIIIIII", "ZZIIIIII", "XIXIIIII", "ZIIIIIII",
    ]
    assert json.loads(capsys.readouterr().out)["measurement_circuit_distance_certified"] is False


@pytest.mark.parametrize("matrix_format", ["npz", "json"])
def test_real_gauging_gate_program_exports_commuting_full_matrices(tmp_path, matrix_format):
    output = tmp_path / matrix_format
    assert cli.main(["--example", "eight", "--gate", "H:1", "--gate", "S:1",
                     "--gate", "CNOT:1:2", "--no-ensure-distance",
                     "--matrix-format", matrix_format, "-o", str(output)]) == 0
    data = _document(output)
    assert data["summary"]["compiled_measurements"] == 11
    assert data["summary"]["unique_deformations"] == 7
    assert data["summary"]["k"] == 4
    assert data["backend_options"]["require_distance"] is False
    assert len(data["deformed_code_files"]) == 11
    for name in data["deformed_code_files"].values():
        path = output / name
        if matrix_format == "npz":
            with np.load(path, allow_pickle=False) as archive:
                x, z = archive["check_x"], archive["check_z"]
        else:
            matrix = json.loads(path.read_text())
            x, z = np.asarray(matrix["check_x"]), np.asarray(matrix["check_z"])
        assert x.shape == z.shape and x.shape[1] >= 8
        assert not np.any((x @ z.T + z @ x.T) % 2)


def test_json_input_basis_and_cli_gate_override(tmp_path):
    code = four_qubit_code()
    basis = code.standard_logicals()
    config = tmp_path / "input.json"
    config.write_text(json.dumps(dict(hx=code.hx.tolist(), hz=code.hz.tolist(), distance=2,
        logical_basis=dict(jx=basis.jx.tolist(), jz=basis.jz.tolist()),
        gates=[["S", 1]], build=False, ancilla=0)))
    output = tmp_path / "custom"
    assert cli.main([str(config), "--gate", "H:1", "-o", str(output)]) == 0
    data = _document(output)
    assert data["gates"] == [{"gate": "H", "qubits": [1]}]
    assert data["logical_basis"]["jx"] == basis.jx.tolist()
    assert data["summary"]["measurement_count"] == 4


def test_outcomes_eigenvalues_and_frame_are_saved_without_sampling(tmp_path, capsys):
    output = tmp_path / "resolved"
    outcomes = tmp_path / "actual-outcomes.json"
    outcomes.write_text(json.dumps({"g0_prep": -1, "g0_m1": 1, "g0_m2": -1, "g0_m3": 1}))
    assert cli.main(["--example", "four", "--gate", "H:1", "--decompose-only",
                     "--outcomes", f"@{outcomes}", "--outcome-format", "eigenvalues",
                     "--initial-frame", '"IX"', "-o", str(output)]) == 0
    correction = json.loads((output / "corrections.json").read_text())
    assert correction["raw_outcomes"] == {"g0_prep": 1, "g0_m1": 0, "g0_m2": 1, "g0_m3": 0}
    assert len(correction["correction_pauli"]) == 2
    assert correction["data_correction_pauli"][0] == "I"
    assert json.loads(capsys.readouterr().out)["corrections"] == correction


def test_without_outcomes_no_corrections_are_invented(tmp_path):
    output = tmp_path / "no-outcomes"
    cli.main(["--example", "four", "--gate", "S:1", "--decompose-only", "-o", str(output)])
    assert not (output / "corrections.json").exists()


def test_cli_default_preserves_backend_options_and_explicit_override_is_local(tmp_path, monkeypatch):
    real_compile = cli.compile_gates
    seen = []

    def capture(code, gates, **kwargs):
        seen.append(kwargs.copy())
        return real_compile(code, gates, **{**kwargs, "build": False})

    monkeypatch.setattr(cli, "compile_gates", capture)
    cli.main(["--example", "four", "--gate", "H:1", "-o", str(tmp_path / "default")])
    assert seen[-1]["build"] is True
    assert seen[-1]["backend_options"] == {}
    cli.main(["--example", "four", "--gate", "H:1", "--backend", "gauge",
              "--backend-options", '{"require_distance": true, "seed": 4}',
              "--no-ensure-distance", "-o", str(tmp_path / "disabled")])
    assert seen[-1]["backend"] == "gauging"
    assert seen[-1]["backend_options"] == {"require_distance": False, "seed": 4}


@pytest.mark.parametrize("backend,key", [("explicit", "require_distance"), ("high_rate", "ensure_distance")])
def test_distance_disable_flag_uses_backend_specific_option(tmp_path, backend, key):
    output = tmp_path / backend
    cli.main(["--example", "four", "--gate", "H:1", "--backend", backend,
              "--decompose-only", "--no-ensure-distance", "-o", str(output)])
    assert _document(output)["backend_options"] == {key: False}


@pytest.mark.parametrize("args,message", [
    (["--gate", "H:0"], "reserved"),
    (["--gate", "CNOT:1:2"], "indices"),
    (["--gate", "H:one"], "integer"),
    (["--gate", "H:1:2"], "syntax"),
    (["--gate", "H:1", "--backend", "missing"], "Unknown backend"),
    (["--gate", "H:1", "--backend-options", "[]"], "object"),
    (["--gate", "H:1", "--outcomes", "[0,0]"], "count"),
    (["--gate", "H:1", "--initial-frame", '"II"'], "requires outcomes"),
    (["--gate", "H:1", "--backend", "ckbb", "--no-ensure-distance"], "supports"),
])
def test_invalid_requests_fail_before_output(tmp_path, args, message, capsys):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        cli.main(["--example", "four", "--decompose-only", "-o", str(output), *args])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_overwrite_requires_explicit_flag(tmp_path):
    output = tmp_path / "plan"
    args = ["--example", "four", "--gate", "H:1", "--decompose-only", "-o", str(output)]
    cli.main(args)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    assert cli.main([*args, "--overwrite"]) == 0


def test_overwrite_without_outcomes_removes_stale_corrections(tmp_path):
    output = tmp_path / "plan"
    args = ["--example", "four", "--gate", "H:1", "--decompose-only", "-o", str(output)]
    cli.main([*args, "--outcomes", "[0,0,0,0]"])
    assert (output / "corrections.json").exists()
    cli.main([*args, "--overwrite"])
    assert not (output / "corrections.json").exists()


def test_module_entrypoint_from_separate_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-m", "codesurgery.pauli_gates",
                             "--example", "four", "--gate", "S:1", "--decompose-only"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["measurement_count"] == 3
    assert (tmp_path / "pauli-gates-output" / "plan.json").exists()
