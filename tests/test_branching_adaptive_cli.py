"""Conditional branching CLI defaults, option precedence, and safe failures."""

import json

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code
from codesurgery.parallel_branching import cli


def write_input(path, **extra):
    code = four_qubit_code()
    config = dict(hx=code.hx.tolist(), hz=code.hz.tolist(), distance=code.distance,
                  targets=["ZZII", "ZIZI"], representatives="preserve",
                  optimize_logical=False)
    config.update(extra)
    path.write_text(json.dumps(config))


def test_adaptive_cli_default_leaves_affordable_overlapping_targets_in_memory(tmp_path, capsys):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source)
    assert cli.main([str(source), "-o", str(output)]) == 0
    data = json.loads(output.read_text())
    x, z = np.asarray(data["check_x"]), np.asarray(data["check_z"])
    assert x.shape[1] == 4
    assert not np.any((x @ z.T + z @ x.T) % 2)
    support = np.asarray(data["leaf_x"]) | np.asarray(data["leaf_z"])
    assert np.any(support[0] & support[1])
    assert data["routing"]["direct_targets"] == [0, 1]
    assert data["routing"]["branched_targets"] == []
    assert max(data["routing"]["projected_original_row_growth"]) <= 2
    assert max(data["routing"]["projected_original_column_growth"]) <= 2
    preview_x = np.asarray(data["measurement_preview"]["check_x"])
    preview_z = np.asarray(data["measurement_preview"]["check_z"])
    assert preview_x.shape[1] > x.shape[1]
    assert not np.any((preview_x @ preview_z.T + preview_z @ preview_x.T) % 2)
    displayed = json.loads(capsys.readouterr().out)
    assert displayed["stage"] == "branching_only"
    assert displayed["logical_measurement_performed"] is False
    assert displayed["branching_mode"] == "adaptive"
    assert displayed["direct_target_count"] == 2
    assert displayed["branched_target_count"] == 0
    assert displayed["fully_disjoint"] is False


@pytest.mark.parametrize("config,flags,expected", [
    ({}, [], ("adaptive", 2, 2)),
    ({"branching_mode": "always", "max_check_increase": 7, "max_qubit_increase": 8},
     [], ("always", 7, 8)),
    ({"branching_mode": "always", "max_check_increase": 7, "max_qubit_increase": 8},
     ["--branching-mode", "adaptive", "--max-check-increase", "2", "--max-qubit-increase", "3"],
     ("adaptive", 2, 3)),
])
def test_adaptive_cli_forwards_defaults_json_and_flag_overrides(tmp_path, monkeypatch, config, flags, expected):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, **config)
    seen = []
    original = cli.branching_deform

    def tracked(*args, **kwargs):
        seen.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "branching_deform", tracked)
    assert cli.main([str(source), "-o", str(output), *flags]) == 0
    assert len(seen) == 1
    assert tuple(seen[0][key] for key in
                 ("branching_mode", "max_check_increase", "max_qubit_increase")) == expected


def test_always_cli_flag_recovers_complete_tree(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source)
    assert cli.main([str(source), "-o", str(output), "--branching-mode", "always"]) == 0
    data = json.loads(output.read_text())
    assert len(data["check_x"][0]) == 10
    support = np.asarray(data["leaf_x"]) | np.asarray(data["leaf_z"])
    assert not np.any(support[0] & support[1])


def test_third_overlap_automatically_branches_and_meets_each_source_budget(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, targets=["ZZII", "ZIZI", "ZIIZ"])
    assert cli.main([str(source), "-o", str(output)]) == 0
    data = json.loads(output.read_text())
    assert len(data["check_x"][0]) > 4
    assert data["routing"]["branched_targets"]
    assert max(data["routing"]["projected_original_row_growth"]) <= 2
    assert max(data["routing"]["projected_original_column_growth"]) <= 2


def test_single_target_can_branch_without_force_copy_when_check_growth_exceeds_two(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    source.write_text(json.dumps(dict(
        hx=[[1] * 8], hz=[[1] * 8], distance=2,
        targets=["ZZZZZZII"], representatives="preserve", optimize_logical=False)))
    assert cli.main([str(source), "-o", str(output)]) == 0
    data = json.loads(output.read_text())
    assert len(data["check_x"][0]) > 8
    assert data["routing"]["branched_targets"] == [0]
    assert data["routing"]["direct_targets"] == []
    assert max(data["routing"]["projected_original_row_growth"]) <= 2


@pytest.mark.parametrize("config,match", [
    ({"branching_mode": "sometimes"}, "branching_mode"),
    ({"max_check_increase": -1}, "max_check_increase"),
    ({"max_qubit_increase": -1}, "max_qubit_increase"),
    ({"max_check_increase": True}, "max_check_increase"),
    ({"max_qubit_increase": "2"}, "max_qubit_increase"),
])
def test_invalid_adaptive_options_preserve_existing_export(tmp_path, capsys, config, match):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, **config)
    output.write_text("existing artifact")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "-o", str(output)])
    assert exc.value.code == 2
    assert match in capsys.readouterr().err
    assert output.read_text() == "existing artifact"
