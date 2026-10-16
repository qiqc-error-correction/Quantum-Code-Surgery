"""The distance guarantee is a user choice, with no noninteractive prompts."""
import io
import json
import numpy as np
import pytest
from codesurgery.explicit_gadgets.cli import main


class Terminal(io.StringIO):
    def isatty(self):
        return True


def config(tmp_path, *, distance=3, **extra):
    h = [[(j >> b) & 1 for j in range(1, 16)] for b in (3, 2, 1, 0)]
    data = dict(hx=h, hz=h, logical="-YXXZZ" + "I" * 10, **extra)
    if distance is not None:
        data["distance"] = distance
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    return path, tmp_path / "out.json"


def test_noninteractive_defaults_to_basic_without_reading_stdin(tmp_path, monkeypatch, capsys):
    source, output = config(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("unexpected prompt"))
    assert main([str(source), "-o", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["summary"]["deformed_n"] == 19
    assert not payload["summary"]["distance_preservation_guaranteed"]
    assert "非交互" in capsys.readouterr().err


@pytest.mark.parametrize("answer,qubits", [("y", 19), ("n", 19), ("", 19)])
def test_interactive_choice(tmp_path, monkeypatch, answer, qubits):
    source, output = config(tmp_path)
    monkeypatch.setattr("sys.stdin", Terminal())
    prompts = []
    def respond(prompt):
        prompts.append(prompt)
        return answer
    monkeypatch.setattr("builtins.input", respond)
    assert main([str(source), "-o", str(output)]) == 0
    assert len(prompts) == 1 and "保证码距" in prompts[0]
    summary = json.loads(output.read_text())["summary"]
    assert summary["deformed_n"] == qubits
    assert summary["distance_preservation_guaranteed"] == (answer == "y")


def test_interactive_retry_and_automatically_computed_source_distance(tmp_path, monkeypatch):
    source, output = config(tmp_path, distance=None)
    monkeypatch.setattr("sys.stdin", Terminal())
    answers = iter(["maybe", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main([str(source), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["distance_lower_bound"] == 3


@pytest.mark.parametrize("flag,require", [("--require-distance", True), ("--no-require-distance", False)])
def test_explicit_flags_override_json_without_prompt(tmp_path, monkeypatch, flag, require):
    source, output = config(tmp_path, require_distance=not require)
    monkeypatch.setattr("sys.stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("unexpected prompt"))
    assert main([str(source), flag, "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["distance_preservation_guaranteed"] == require


@pytest.mark.parametrize("extra,match", [
    ({"distance_options": {"max_check_weight": 2}}, "sparsity"),
    ({"require_distance": "yes"}, "true or false"),
])
def test_failed_guarantee_does_not_overwrite(tmp_path, monkeypatch, capsys, extra, match):
    source, output = config(tmp_path, **extra)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    output.write_text("previous result")
    flags = [] if "require_distance" in extra else ["--require-distance"]
    with pytest.raises(SystemExit) as error:
        main([str(source), *flags, "-o", str(output)])
    assert error.value.code == 2
    assert match in capsys.readouterr().err
    assert output.read_text() == "previous result"


def test_noninteractive_guarantee_can_compute_missing_distance(tmp_path, monkeypatch, capsys):
    source, output = config(tmp_path, distance=None, scheme="gauging")
    monkeypatch.setattr("sys.stdin", io.StringIO())
    assert main([str(source), "--require-distance", "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["source_distance_calculation"]["distance"] == 3
    assert main([str(source), "--require-distance", "--distance", "3", "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["distance_required"]


def test_opt_out_ignores_stored_distance_limits(tmp_path, monkeypatch):
    source, output = config(tmp_path, distance_options={"max_base_degree": 1})
    monkeypatch.setattr("sys.stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main([str(source), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["deformed_n"] == 19


def test_missing_explicit_milp_dependency_has_clean_cli_error(tmp_path, monkeypatch, capsys):
    from codesurgery.distance_preservation import _distance_milp
    def unavailable(*args, **kwargs):
        raise ImportError("SciPy unavailable")
    monkeypatch.setattr(_distance_milp, "milp_distance", unavailable)
    source, output = config(tmp_path, distance_options={"distance_method": "milp"})
    output.write_text("previous result")
    with pytest.raises(SystemExit) as error:
        main([str(source), "--require-distance", "-o", str(output)])
    assert error.value.code == 2
    assert "codesurgery[distance]" in capsys.readouterr().err
    assert output.read_text() == "previous result"
