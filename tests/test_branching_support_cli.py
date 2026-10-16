"""CLI controls for support reduction before branching."""

import json

import numpy as np
import pytest

from codesurgery.parallel_branching import cli


def write_input(path, **extra):
    # A weight-five logical Z differs from a weight-one logical Z by ZZZZII.
    path.write_text(json.dumps(dict(
        n=6, hx=[], hz=[[1, 1, 1, 1, 0, 0]], distance=1,
        targets=["-ZZZZZI"], representatives="preserve", **extra)))


@pytest.mark.parametrize("config,flags,expected_weight", [
    ({}, [], 1),
    ({"optimize_logical": False}, [], 5),
    ({"optimize_logical": True}, ["--no-optimize-logical"], 5),
    ({"optimize_logical": False}, ["--optimize-logical"], 1),
])
def test_support_optimization_default_and_cli_precedence(tmp_path, config, flags, expected_weight):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, **config)
    assert cli.main([str(source), "-o", str(output), *flags]) == 0
    data = json.loads(output.read_text())
    leaf_x, leaf_z = np.asarray(data["leaf_x"]), np.asarray(data["leaf_z"])
    assert np.count_nonzero(leaf_x | leaf_z) == expected_weight
    assert data["leaf_signs"] == [1]
    # No forced copy: optimization changes the representative, not the source code.
    assert leaf_x.shape[1] == 6


def test_cli_forwards_reduction_options_without_modifying_them(tmp_path, monkeypatch):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    options = dict(method="heuristic", threshold=None, max_trials=3, seed=7)
    write_input(source, logical_optimization_options=options)
    seen = []
    original = cli.branching_deform

    def tracked(*args, **kwargs):
        seen.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "branching_deform", tracked)
    assert cli.main([str(source), "-o", str(output)]) == 0
    assert seen[0]["optimize_logical"] is True
    assert seen[0]["logical_optimization_options"] == options


@pytest.mark.parametrize("config,flags,match", [
    ({"optimize_logical": "true"}, [], "optimize_logical must be true or false"),
    ({"logical_optimization_options": []}, [], "logical_optimization_options must be a JSON object"),
    ({"optimize_logical": False, "logical_optimization_options": {}}, [], "requires optimize_logical"),
    ({"logical_optimization_options": {"seed": 7}}, ["--no-optimize-logical"], "requires optimize_logical"),
])
def test_invalid_optimization_settings_preserve_existing_output(tmp_path, capsys, config, flags, match):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, **config)
    output.write_text("existing artifact")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(source), "-o", str(output), *flags])
    assert exc.value.code == 2
    assert match in capsys.readouterr().err
    assert output.read_text() == "existing artifact"


def test_positive_cli_flag_can_enable_json_optimization_options(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "branched.json"
    write_input(source, optimize_logical=False,
                logical_optimization_options={"method": "heuristic", "threshold": None})
    assert cli.main([str(source), "-o", str(output), "--optimize-logical"]) == 0
    data = json.loads(output.read_text())
    assert np.count_nonzero(np.asarray(data["leaf_x"]) | np.asarray(data["leaf_z"])) == 1
