"""Separate branching CLI: exported matrices, option forwarding, no measurement."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.gf2 import rank
from codesurgery.parallel_branching import cli


def write_input(path, **extra):
    code = four_qubit_code()
    path.write_text(json.dumps(dict(hx=code.hx.tolist(), hz=code.hz.tolist(),
                                   distance=code.distance, branching_mode="always", **extra)))


def assert_saved_branch(path, expected_n=None):
    data = json.loads(path.read_text())
    x, z = np.asarray(data["check_x"]), np.asarray(data["check_z"])
    assert x.shape == z.shape
    assert not np.any((x @ z.T + z @ x.T) % 2)
    assert x.shape[1] - rank(np.hstack([x, z])) == 2
    if expected_n is not None:
        assert x.shape[1] == expected_n
    lx, lz = np.asarray(data["leaf_x"]), np.asarray(data["leaf_z"])
    support = lx | lz
    overlaps = support @ support.T
    np.fill_diagonal(overlaps, 0)
    assert not np.any(overlaps)
    assert not np.any((x @ lz.T + z @ lx.T) % 2)
    return data


def test_cli_overlapping_physical_targets_outputs_branch_not_measurement(tmp_path, capsys):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, targets=["ZZII", "ZIZI"])
    assert cli.main([str(source), "-o", str(output), "--representatives", "preserve"]) == 0
    data = assert_saved_branch(output, expected_n=10)
    assert len(data["leaf_x"]) == 2
    displayed = json.loads(capsys.readouterr().out)
    assert displayed["logical_measurement_performed"] is False
    assert displayed["stage"] == "branching_only"
    assert "distance_calculation" not in data["metadata"]


def test_cli_default_targets_select_all_z_logicals(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source)
    assert cli.main([str(source), "-o", str(output)]) == 0
    data = assert_saved_branch(output)
    assert len(data["leaf_x"]) == four_qubit_code().k


def test_cli_mixed_logical_coordinates_and_explicit_basis(tmp_path):
    code = four_qubit_code()
    basis = code.standard_logicals()
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, logical_paulis=["XI", "IZ"],
        logical_basis={"jx": basis.jx.tolist(), "jz": basis.jz.tolist()})
    assert cli.main([str(source), "-o", str(output)]) == 0
    data = assert_saved_branch(output)
    assert len(data["leaf_x"]) == 2


def test_cli_keeps_signed_physical_target(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, targets=["-ZZII"], representatives="preserve")
    assert cli.main([str(source), "-o", str(output), "--force-copy", "--leaf-pauli", "Z"]) == 0
    data = assert_saved_branch(output)
    assert data["leaf_signs"] == [1]


def test_cli_force_copy_y_and_flags_override_json(tmp_path, monkeypatch):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, logical_paulis=["YI"], force_copy=False,
                leaf_pauli="auto", representatives="auto", max_entries=1)
    calls = []
    original = cli.branching_deform
    def tracked(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(cli, "branching_deform", tracked)
    assert cli.main([str(source), "-o", str(output), "--force-copy",
        "--leaf-pauli", "Z", "--representatives", "preserve", "--max-entries", "50000000"]) == 0
    data = assert_saved_branch(output)
    assert len(data["check_x"][0]) > 4
    assert not np.any(data["leaf_x"])
    assert calls[0]["force_copy"] is True and calls[0]["leaf_pauli"] == "Z"
    assert calls[0]["representatives"] == "preserve" and calls[0]["max_entries"] == 50_000_000


@pytest.mark.parametrize("options", [{}, {"method": "cluster", "max_candidates": 1000}])
def test_cli_optional_distance_uses_validated_result_and_saves_report(tmp_path, monkeypatch, options):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, logical_paulis=["YI"], distance_options=options)
    seen = []
    report = dict(status="bounded", distance=None, proven_lower_bound=1, upper_bound=2,
                  distance_statement="0 < d <= 2")
    def fake_distance(result, **kwargs):
        result.validate()
        seen.append(kwargs)
        return report.copy()
    monkeypatch.setattr(cli, "code_distance", fake_distance)
    assert cli.main([str(source), "-o", str(output), "--compute-distance"]) == 0
    assert seen == [options]
    data = assert_saved_branch(output, expected_n=4)
    assert data["metadata"]["distance_calculation"] == report


def test_cli_npz_export(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.npz"
    write_input(source, targets=["ZZII", "ZIZI"], representatives="preserve")
    assert cli.main([str(source), "-o", str(output)]) == 0
    with np.load(output, allow_pickle=False) as data:
        assert data["check_x"].shape[1] == 10
        assert data["leaf_x"].shape[0] == 2
        assert not np.any((data["check_x"] @ data["check_z"].T
                          + data["check_z"] @ data["check_x"].T) % 2)


@pytest.mark.parametrize("config,error", [
    ({"targets": ["ZZII"], "logical_paulis": ["ZI"]}, "not both"),
    ({"logical_paulis": ["XI", "ZI"]}, "commut"),
    ({"compute_distance": "yes"}, "compute_distance"),
    ({"distance_options": []}, "distance_options"),
    ({"distance_options": {"method": "cluster"}}, "requires compute_distance"),
    ({"force_copy": "yes"}, "force_copy"),
    ({"max_entries": 0}, "max_entries"),
])
def test_cli_invalid_input_does_not_overwrite_output(tmp_path, capsys, config, error):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, **config)
    output.write_text("existing artifact")
    with pytest.raises(SystemExit) as raised:
        cli.main([str(source), "-o", str(output)])
    assert raised.value.code == 2
    assert error in capsys.readouterr().err
    assert output.read_text() == "existing artifact"


def test_module_entrypoint_help_clarifies_branching_only():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run([sys.executable, "-m", "codesurgery.parallel_branching", "--help"],
        text=True, capture_output=True, env=env, check=True)
    assert "no logical measurement is performed" in completed.stdout
    assert "--compute-distance" in completed.stdout
