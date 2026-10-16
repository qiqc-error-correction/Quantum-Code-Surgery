"""Real CLI constructions, target semantics, exports, and certification failures."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.gf2 import matmul, nullspace, rank, row_basis
from codesurgery.high_rate_surgery import cli
from codesurgery.symplectic import pauli_product


def assert_export(path, expected_rank):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files if key != "information"}
            data.update(json.loads(archive["information"].item()))
    else:
        data = json.loads(path.read_text())
    x, z = np.asarray(data["check_x"], dtype=np.uint8), np.asarray(data["check_z"], dtype=np.uint8)
    signs = np.asarray(data["check_signs"], dtype=np.uint8)
    assert x.shape == z.shape
    assert not np.any(matmul(x, z.T) ^ matmul(z, x.T))
    n = data["summary"]["original_n"]
    jx, jz = np.asarray(data["logical_jx"], np.uint8), np.asarray(data["logical_jz"], np.uint8)
    cancel = nullspace(np.hstack((x[:, n:], z[:, n:])).T)
    rx, rz = matmul(cancel, x[:, :n]), matmul(cancel, z[:, :n])
    measured = row_basis(np.hstack((matmul(rx, jz.T), matmul(rz, jx.T))))
    tx, tz = np.asarray(data["target_x"], np.uint8), np.asarray(data["target_z"], np.uint8)
    requested = row_basis(np.hstack((matmul(tx, jz.T), matmul(tz, jx.T))))
    assert len(measured) == len(requested) == expected_rank
    assert rank(np.vstack((measured, requested))) == expected_rank
    for i, witness in enumerate(data["measurement_witnesses"]):
        px, pz, phase = pauli_product(x, z, signs, np.flatnonzero(witness))
        assert np.array_equal(px[:n], tx[i]) and np.array_equal(pz[:n], tz[i])
        assert not np.any(px[n:] | pz[n:])
        assert phase == 2 * (int(data["target_signs"][i]) ^ int(data["readout_sign_corrections"][i]))
    weights = x | z
    assert weights.sum(axis=1).max() <= data["summary"]["check_weight_limit"]
    assert weights.sum(axis=0).max() <= data["summary"]["qubit_degree_limit"]
    return data


def test_default_targets_are_all_z_and_json_reports_unmeasured_fault_distance(tmp_path, capsys):
    output = tmp_path / "all.json"
    assert cli.main(["--example", "four", "--no-ensure-distance", "-o", str(output)]) == 0
    data = assert_export(output, 2)
    summary = json.loads(capsys.readouterr().out)
    assert summary["target_count"] == 2
    assert summary["protected_k"] == 0
    assert summary["measurement_fault_distance_certified"] is False
    assert data["metadata"]["status"] == "constructed_distance_disabled"


@pytest.mark.parametrize("targets,rank_expected", [(["ZI", "IZ"], 2), (["ZZ"], 1)])
def test_parallel_vs_joint_target_semantics(tmp_path, targets, rank_expected):
    output = tmp_path / "result.npz"
    args = ["--example", "four", "--no-ensure-distance", "-o", str(output)]
    for target in targets:
        args.extend(["--logical-pauli", target])
    assert cli.main(args) == 0
    data = assert_export(output, rank_expected)
    assert data["summary"]["target_count"] == len(targets)


def test_explicit_signed_physical_target_and_basis(tmp_path):
    code = four_qubit_code()
    basis = code.standard_logicals()
    config, output = tmp_path / "input.json", tmp_path / "signed.json"
    config.write_text(json.dumps(dict(hx=code.hx.tolist(), hz=code.hz.tolist(), distance=2,
        targets=["-ZZII"], logical_basis=dict(jx=basis.jx.tolist(), jz=basis.jz.tolist()),
        optimize_support=False, ensure_distance=False)))
    assert cli.main([str(config), "-o", str(output)]) == 0
    data = assert_export(output, 1)
    assert data["target_signs"] == [1]
    assert data["basis_source"] == "user supplied paired CSS basis"


def test_cli_target_overrides_json_selector_and_flags_override_options(tmp_path):
    config, output = tmp_path / "input.json", tmp_path / "override.json"
    config.write_text(json.dumps(dict(example="four", logical_paulis=["ZI", "IZ"],
        ensure_distance=True, max_entries=1, trials=1)))
    assert cli.main([str(config), "--target", "ZZII", "--no-ensure-distance",
                     "--max-entries", "50000000", "--trials", "4", "-o", str(output)]) == 0
    data = assert_export(output, 1)
    assert data["summary"]["target_count"] == 1
    assert data["metadata"]["status"] == "constructed_distance_disabled"


def test_zero_based_indices_and_x_dual(tmp_path):
    output = tmp_path / "x.json"
    assert cli.main(["--example", "four", "--logical-indices", "1", "--pauli", "X",
                     "--no-ensure-distance", "-o", str(output)]) == 0
    data = assert_export(output, 1)
    assert not np.any(data["target_z"])
    assert data["summary"]["is_css"] is True


def test_noninteractive_default_enables_distance_without_prompt(tmp_path, capsys, monkeypatch):
    output = tmp_path / "enum.json"
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["--example", "four", "--logical-pauli", "ZZ", "--distance-method", "enum",
                     "--distance-time-limit", "2", "--max-steps", "0", "-o", str(output)]) == 0
    data = assert_export(output, 1)
    assert data["metadata"]["status"] != "constructed_distance_disabled"
    printed = capsys.readouterr()
    assert "非交互模式默认开启" in printed.err
    assert json.loads(printed.out)["scheme"] == "high_rate_surgery"


def test_terminal_prompt_retries_and_keeps_stdout_json(tmp_path, capsys, monkeypatch):
    output = tmp_path / "prompt.json"
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["invalid", "n"])
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    assert cli.main(["--example", "four", "--logical-pauli", "ZZ", "-o", str(output)]) == 0
    data = assert_export(output, 1)
    printed = capsys.readouterr()
    assert "是否需要保证码距" in printed.err and "请输入 y 或 n" in printed.err
    assert json.loads(printed.out)["status"] == "constructed_distance_disabled"
    assert data["summary"]["protected_distance_certified"] is False


def test_single_y_gauge_free_full_symplectic_distance_can_be_certified(tmp_path, capsys):
    output = tmp_path / "y-certified.json"
    assert cli.main(["--example", "four", "--logical-pauli", "YI", "--ensure-distance",
                     "--distance-method", "enum", "--strict", "-o", str(output)]) == 0
    data = assert_export(output, 1)
    summary = json.loads(capsys.readouterr().out)
    assert summary["gauge_qubits"] == 0
    assert summary["protected_distance_certified"] is True
    assert summary["measurement_fault_distance_certified"] is False
    assert data["metadata"]["distance"]["scope"] == "protected_mixed_stabilizer_gauge_free"
    assert data["metadata"]["distance"]["proven_lower_bound"] >= 2


def test_strict_unattained_distance_saves_valid_candidate_and_fails(tmp_path, capsys):
    output = tmp_path / "y.json"
    assert cli.main(["--example", "four", "--logical-pauli", "YI", "--ensure-distance",
                     "--distance-method", "enum", "--target-distance", "3", "--max-steps", "0",
                     "--strict", "-o", str(output)]) == 2
    data = assert_export(output, 1)
    summary = json.loads(capsys.readouterr().out)
    assert summary["strict_failed"] is True and summary["candidate_saved"] is True
    assert data["metadata"]["status"] == "step_limit"
    assert data["metadata"]["distance"]["upper_bound"] < 3
    assert data["summary"]["protected_distance_certified"] is False


def test_strict_without_output_does_not_create_default_file(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--example", "four", "--logical-pauli", "YI", "--ensure-distance",
                     "--distance-method", "enum", "--target-distance", "3", "--max-steps", "0",
                     "--strict"]) == 2
    assert not (tmp_path / "high-rate-deformed.json").exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("extra,error", [
    ({"targets": ["ZZII"], "logical_paulis": ["ZI"]}, "not multiple"),
    ({"logical_paulis": "ZI"}, "JSON list"),
    ({"logical_paulis": ["XI", "ZI"]}, "commute"),
    ({"optimize_support": "true"}, "true or false"),
    ({"ensure_distance": "yes"}, "true or false"),
    ({"strict": "false"}, "true or false"),
    ({"max_check_weight": 3}, "smaller than the original"),
    ({"max_entries": 0}, "max_entries"),
    ({"trials": 0}, "trials"),
])
def test_invalid_input_does_not_replace_existing_artifact(tmp_path, capsys, extra, error):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(dict(example="four", ensure_distance=False) | extra))
    output.write_text("existing artifact")
    with pytest.raises(SystemExit) as exception:
        cli.main([str(source), "-o", str(output)])
    assert exception.value.code == 2
    assert error in capsys.readouterr().err
    assert output.read_text() == "existing artifact"


def test_globally_commuting_but_non_qwc_targets_are_not_split(tmp_path, capsys):
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit) as exception:
        cli.main(["--example", "four", "--target", "XXII", "--target", "ZZII",
                  "--no-optimize-support", "--no-ensure-distance", "-o", str(output)])
    assert exception.value.code == 2
    assert "QWC" in capsys.readouterr().err
    assert not output.exists()


def test_module_entrypoint_subprocess(tmp_path):
    output = tmp_path / "subprocess.npz"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    run = subprocess.run([sys.executable, "-m", "codesurgery.high_rate_surgery",
        "--example", "four", "--logical-pauli", "ZZ", "--no-ensure-distance", "-o", str(output)],
        text=True, capture_output=True, env=env, check=True)
    assert json.loads(run.stdout)["logical_measurement_rank"] == 1
    assert_export(output, 1)
