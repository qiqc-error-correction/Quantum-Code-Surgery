"""Automatic component adapters preserve exactly the requested logical readout."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from codesurgery.examples import direct_sum, four_qubit_code, steane_code
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.gauging_measurement.api import build_graph
from codesurgery.gauging_measurement.components import verify_component_adapter_expansion
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.symplectic import parse_pauli, pauli_product


def _assert_only_target(code, result, excluded=()):
    """Eliminate all auxiliary columns without relying on metadata's verdict."""
    assert result.validate()
    cx, cz = result.check_x, result.check_z
    assert not np.any((cx @ cz.T + cz @ cx.T) % 2)
    cancellation = nullspace(np.hstack([cx[:, code.n:], cz[:, code.n:]]).T)
    image = cancellation @ np.hstack([cx[:, :code.n], cz[:, :code.n]]) % 2
    old = np.block([[code.hx, np.zeros_like(code.hx)],
                    [np.zeros_like(code.hz), code.hz]])
    target = np.concatenate([result.measurement_x, result.measurement_z])
    assert rank(np.vstack([old, image])) == rank(old) + 1
    assert in_rowspace(image, np.vstack([old, target]))
    assert in_rowspace(target, image)
    for word in excluded:
        x, z, _ = parse_pauli(word, code.n)
        assert not in_rowspace(np.concatenate([x, z]), image)
    for dependency in nullspace(np.hstack([cx, cz]).T):
        assert pauli_product(cx, cz, result.check_signs,
                             np.flatnonzero(dependency))[2] == 0
    x, z, phase = pauli_product(cx, cz, result.check_signs,
                                np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(x[:code.n], result.measurement_x)
    np.testing.assert_array_equal(z[:code.n], result.measurement_z)
    assert not np.any(x[code.n:] | z[code.n:])
    assert phase == 2 * result.measurement_sign
    assert result.k_stabilizer == code.k - 1


def _assert_adapter_construction(result):
    """Check the emitted graph/cycle blocks, including redundant cyclic rows."""
    metadata = result.metadata
    b, c = metadata["graph_incidence"], metadata["cycle_basis"]
    report = metadata["graph_design"]
    assert np.all(b.sum(axis=1) == 2)
    assert rank(b) == b.shape[1] - 1
    assert not np.any(c @ b % 2)
    assert rank(c) == len(b) - b.shape[1] + 1
    assert report["final_cycle_rank"] == rank(c)
    assert len(report["adapters"]) == report["component_count"] - 1
    for adapter in report["adapters"]:
        edges, rows = adapter["edge_rows"], adapter["cycle_rows"]
        a = adapter["adapter_size"]
        assert len(edges) == a
        assert len(rows) == (a if a > 1 else 0)
        if rows:
            connector = c[rows]
            np.testing.assert_array_equal(connector[:, edges], adapter["H_C"])
            np.testing.assert_array_equal(connector[:, adapter["left_edge_rows"]],
                                          adapter["T_left"])
            np.testing.assert_array_equal(connector[:, adapter["right_edge_rows"]],
                                          adapter["T_right"])
            assert np.all(connector.sum(axis=1) <= 8)
            assert np.all(connector.sum(axis=0) <= 2)
            assert rank(adapter["H_C"]) == a - 1
    assert sum(a["adapter_size"] for a in report["adapters"]) == report["added_adapter_edges"]


@pytest.mark.parametrize("count", [2, 3])
def test_default_joins_each_component_with_full_repetition_adapter(count):
    code = direct_sum(*(four_qubit_code() for _ in range(count)))
    result = gauging_deform(code, "XXII" * count, require_distance=False)
    factors = ["IIII" * i + "XXII" + "IIII" * (count - i - 1)
               for i in range(count)]
    _assert_only_target(code, result, factors)
    _assert_adapter_construction(result)
    report = result.metadata["graph_design"]
    assert report["component_count"] == count
    assert report["initial_logical_rank"] == count
    assert all(a["adapter_size"] == 2 for a in report["adapters"])
    assert len(result.metadata["graph_incidence"]) == count + 2 * (count - 1)
    assert rank(result.metadata["cycle_basis"]) == count - 1
    assert report["added_adapter_cycle_rows"] == 2 * (count - 1)
    # Port order remains the physical target's sorted support order.
    np.testing.assert_array_equal(result.metadata["port_selector"],
                                  np.eye(code.n, dtype=np.uint8)[np.flatnonzero(result.measurement_x)])


@pytest.mark.parametrize("target", ["-YXZIZZII", "-YYIIXXII", "-ZZIIXXII"])
def test_signed_mixed_and_y_targets_keep_only_the_full_product(target):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_deform(code, target, require_distance=False)
    _assert_only_target(code, result,
                        [target[1:5] + "IIII", "IIII" + target[5:]])
    _assert_adapter_construction(result)
    assert result.measurement_sign == 1
    report = result.metadata["graph_design"]
    assert report["component_count"] == 2
    assert report["initial_logical_rank"] == 2


def test_three_bit_adapter_retains_three_cycle_generators_of_rank_two():
    code = direct_sum(steane_code(), steane_code())
    basis = code.standard_logicals()
    target = {"x": basis.jx[0] ^ basis.jx[1], "z": np.zeros(code.n, dtype=np.uint8)}
    result = gauging_deform(code, target, component_adapter_shift=2, require_distance=False)
    _assert_only_target(code, result)
    _assert_adapter_construction(result)
    adapter = result.metadata["graph_design"]["adapters"][0]
    assert adapter["adapter_size"] == 3
    assert len(adapter["cycle_rows"]) == 3
    assert rank(result.metadata["cycle_basis"][adapter["cycle_rows"]]) == 2


def test_interleaved_component_vertices_preserve_matching_and_physical_port_order():
    original = direct_sum(four_qubit_code(), four_qubit_code())
    permutation = [0, 4, 1, 5, 2, 6, 3, 7]
    code = CSSCode(original.hx[:, permutation], original.hz[:, permutation], distance=2)
    result = gauging_deform(code, "XXXXIIII", require_distance=False)
    _assert_only_target(code, result, ["XIXIIIII", "IXIXIIII"])
    _assert_adapter_construction(result)
    report = result.metadata["graph_design"]
    assert [c["vertices"] for c in report["components"]] == [[0, 2], [1, 3]]
    np.testing.assert_array_equal(result.metadata["port_selector"],
                                  np.eye(code.n, dtype=np.uint8)[:4])
    boundary = np.vstack([np.zeros((len(code.hx), 4), dtype=np.uint8), code.hz[:, :4]])
    np.testing.assert_array_equal(result.metadata["matching_matrix"]
                                  @ result.metadata["graph_incidence"] % 2, boundary)


def test_stabilizer_component_is_connected_without_claiming_extra_logical():
    h = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=np.uint8)
    code = CSSCode(h, h, distance=2)
    result = gauging_deform(code, "XXIIXX", require_distance=False)
    _assert_only_target(code, result)
    _assert_adapter_construction(result)
    report = result.metadata["graph_design"]
    assert report["component_count"] == 2
    assert report["initial_logical_rank"] == 1
    assert [c["kind"] for c in report["components"]] == ["logical", "stabilizer"]
    assert not np.any(report["component_signature_matrix"][1])


def test_dependent_nonzero_component_signatures_are_not_counted_as_three_logicals():
    code = CSSCode([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]],
                   [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1]], distance=2)
    result = gauging_deform(code, "XXXXXX", require_distance=False)
    _assert_only_target(code, result)
    _assert_adapter_construction(result)
    report = result.metadata["graph_design"]
    assert report["component_count"] == 3
    assert report["initial_logical_rank"] == 1
    signatures = np.asarray(report["component_signature_matrix"])
    assert np.all(signatures.any(axis=1))
    assert np.array_equal(signatures[0], signatures[1])
    assert np.array_equal(signatures[0], signatures[2])
    np.testing.assert_array_equal(signatures.sum(axis=0) % 2,
                                  report["target_signature"])


def test_connected_automatic_graph_keeps_existing_deformed_matrices():
    code = four_qubit_code()
    automatic = gauging_deform(code, "-YXZI", require_distance=False)
    chain = gauging_deform(code, "-YXZI", component_connection="chain", require_distance=False)
    for field in ("check_x", "check_z", "check_signs", "measurement_witness"):
        np.testing.assert_array_equal(getattr(automatic, field), getattr(chain, field))
    assert automatic.metadata["graph_design"]["component_count"] == 1
    assert automatic.metadata["graph_design"]["adapters"] == []
    _assert_only_target(code, automatic)


def test_explicit_chain_retains_old_single_bridge_and_size_one_is_explicit():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    target = "XXIIXXII"
    chain = gauging_deform(code, target, component_connection="chain", require_distance=False)
    thin = gauging_deform(code, target, component_adapter_size=1, require_distance=False)
    for result in (chain, thin):
        _assert_only_target(code, result, ["XXIIIIII", "IIIIXXII"])
        assert len(result.metadata["graph_incidence"]) == 3
        assert rank(result.metadata["cycle_basis"]) == 0
        assert not result.metadata["distance_preservation_guaranteed"]
    _assert_adapter_construction(thin)


def test_adapter_shift_changes_connections_while_preserving_signed_readout():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    first = gauging_deform(code, "-XXIIXXII", require_distance=False)
    shifted = gauging_deform(code, "-XXIIXXII", component_adapter_shift=1, require_distance=False)
    assert not np.array_equal(first.metadata["graph_incidence"], shifted.metadata["graph_incidence"])
    _assert_only_target(code, shifted, ["XXIIIIII", "IIIIXXII"])
    _assert_adapter_construction(shifted)
    assert shifted.metadata["graph_design"]["adapters"][0]["adapter_shift"] == 1


def test_user_graph_is_unchanged_and_disconnected_graph_still_rejected():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    graph = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    result = gauging_deform(code, "XXIIXXII", graph=graph, require_distance=False)
    np.testing.assert_array_equal(result.metadata["graph_incidence"], graph)
    assert "graph_design" not in result.metadata
    _assert_only_target(code, result)
    with pytest.raises(ValueError, match="connected"):
        gauging_deform(code, "XXIIXXII", graph=graph[[0, 2]], require_distance=False)


@pytest.mark.parametrize("options,match", [
    ({"component_connection": "unknown"}, "component_connection"),
    ({"component_adapter_size": 0}, "size"),
    ({"component_adapter_size": True}, "size"),
    ({"component_adapter_size": 1.5}, "size"),
    ({"component_adapter_size": 3}, "size"),
    ({"component_adapter_shift": -1}, "shift"),
    ({"component_adapter_shift": True}, "shift"),
    ({"component_adapter_shift": 0.5}, "shift"),
    ({"component_adapter_shift": 2}, "shift"),
    ({"component_connection": "chain", "component_adapter_size": 1}, "adapter"),
])
def test_invalid_component_adapter_options_raise(options, match):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(ValueError, match=match):
        gauging_deform(code, "XXIIXXII", require_distance=False, **options)


@pytest.mark.parametrize("options", [
    {"component_adapter_size": 1}, {"component_adapter_shift": 1},
    {"component_connection": "chain"},
])
def test_component_options_cannot_silently_modify_user_supplied_graph(options):
    with pytest.raises(ValueError, match="automatic graph"):
        gauging_deform(four_qubit_code(), "XXII", graph=[[1, 1]],
                       require_distance=False, **options)


def test_default_distance_pipeline_has_no_single_qubit_logical_after_adapter():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_deform(code, "XXIIXXII")
    _assert_only_target(code, result, ["XXIIIIII", "IIIIXXII"])
    post = result.metadata["distance_postprocessing"]
    assert post["required"] and post["distance_lower_bound"] >= 2
    assert post["sparsity_certificate"]["condition_met"]
    assert post["sparsity_certificate"]["verified_against_full_check_matrix"]
    stabilizers = np.hstack([result.check_x, result.check_z])
    for q in range(result.n):
        for px, pz in ((1, 0), (0, 1), (1, 1)):
            x, z = np.zeros(result.n, dtype=np.uint8), np.zeros(result.n, dtype=np.uint8)
            x[q], z[q] = px, pz
            if not np.any((x @ result.check_z.T + z @ result.check_x.T) % 2):
                assert in_rowspace(np.concatenate([x, z]), stabilizers)


def test_custom_logical_basis_controls_component_and_target_signatures():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    standard = code.standard_logicals()
    permutation = [2, 3, 0, 1]
    basis = {"jx": standard.jx[permutation], "jz": standard.jz[permutation]}
    coefficients = [1, 0, 1, 0]
    result = gauging_deform(code, {"logical_x": coefficients}, logical_basis=basis,
                            require_distance=False)
    _assert_only_target(code, result)
    _assert_adapter_construction(result)
    np.testing.assert_array_equal(result.measurement_x, basis["jx"][0] ^ basis["jx"][2])
    expected = coefficients + [0, 0, 0, 0]
    np.testing.assert_array_equal(result.metadata["graph_design"]["target_signature"], expected)
    np.testing.assert_array_equal(result.metadata["measurement_validation"]["target_signature"], expected)


def test_component_construction_respects_dense_allocation_budget():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(ValueError, match="max_entries"):
        gauging_deform(code, "XXIIXXII", max_entries=100, require_distance=False)


def _cli_command(source, output, *arguments):
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    return subprocess.run([sys.executable, "-m", "codesurgery.gauging_measurement",
                           str(source), "--no-require-distance", "-o", str(output), *arguments],
                          cwd=project, env=env, input="", text=True, capture_output=True,
                          timeout=30)


def test_cli_exports_repetition_adapter_matrices(tmp_path):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    source, output = tmp_path / "input.json", tmp_path / "deformed.json"
    source.write_text(json.dumps({"hx": code.hx.tolist(), "hz": code.hz.tolist(),
                                  "distance": 2, "logical": "-XXIIXXII"}))
    process = _cli_command(source, output, "--component-adapter-size", "2")
    assert process.returncode == 0, process.stderr
    payload = json.loads(output.read_text())
    design = payload["metadata"]["graph_design"]
    assert design["component_count"] == 2
    assert design["adapters"][0]["adapter_size"] == 2
    assert design["added_adapter_edges"] == 2
    assert payload["summary"]["measurement"] == "-XXIIXXII"
    cx, cz = np.asarray(payload["check_x"]), np.asarray(payload["check_z"])
    assert not np.any((cx @ cz.T + cz @ cx.T) % 2)


def test_cli_rejects_component_flags_for_explicit_scheme_without_overwriting(tmp_path):
    code = four_qubit_code()
    source, output = tmp_path / "input.json", tmp_path / "deformed.json"
    source.write_text(json.dumps({"hx": code.hx.tolist(), "hz": code.hz.tolist(),
                                  "distance": 2, "logical": "XXII"}))
    output.write_text("keep previous result")
    process = _cli_command(source, output, "--scheme", "explicit", "--component-adapter-size", "2")
    assert process.returncode == 2
    assert "gauging" in process.stderr
    assert output.read_text() == "keep previous result"


@pytest.mark.parametrize("kind,n", [("XX", 21), ("XZ", 22)])
def test_automatic_shift_preserves_steane_distance_within_original_row_cap(kind, n):
    code = direct_sum(steane_code(), steane_code())
    basis = code.standard_logicals()
    zero = np.zeros(code.n, dtype=np.uint8)
    target = {"x": basis.jx[0] ^ (basis.jx[1] if kind == "XX" else zero),
              "z": zero if kind == "XX" else basis.jz[1], "sign": int(kind == "XZ")}
    result = gauging_deform(code, target)
    _assert_only_target(code, result)
    _assert_adapter_construction(result)
    post = result.metadata["distance_postprocessing"]
    assert result.n == n
    assert post["distance_lower_bound"] == 3
    assert post["effective_row_threshold"] == 5
    assert post["changed"] is False
    assert int((result.check_x | result.check_z).sum(axis=1).max()) <= 5
    adapter = result.metadata["graph_design"]["adapters"][0]
    assert adapter["adapter_shift"] == 1
    unrotated = gauging_deform(code, target, component_adapter_shift=0, require_distance=False)
    assert int(unrotated.metadata["cycle_basis"].sum(axis=1).max()) == 6


def test_component_theorem_certificate_works_when_whole_graph_enumeration_is_disabled():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_deform(code, "XXIIXXII", exact_limit=2, require_distance=False)
    assert not result.metadata["relative_expansion"]["exact"]
    assert result.metadata["relative_expansion"]["condition_met"]
    proof = result.metadata["component_adapter_distance_proof"]
    assert proof["verified_against_actual_graph"] and proof["condition_met"]
    assert result.metadata["distance_lower_bound"] == 2


def test_forged_positive_component_distance_flag_cannot_certify_a_thin_adapter():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    thin = gauging_deform(code, "XXIIXXII", component_adapter_size=1, require_distance=False)
    md = thin.metadata
    spoofed = deepcopy(md["graph_design"])
    spoofed["distance_preservation_guaranteed"] = True
    rebuilt = build_graph(code, (thin.measurement_x, thin.measurement_z, thin.measurement_sign),
                          md["logical_basis"], md["graph_incidence"], matching=md["matching_matrix"],
                          cycles=md["cycle_basis"], extra_metadata={"graph_design": spoofed,
                          "component_adapter_distance_proof": {"condition_met": True}})
    assert not rebuilt.metadata["component_adapter_distance_proof"]["condition_met"]
    assert not rebuilt.metadata["distance_preservation_guaranteed"]


def test_component_certificate_rejects_graph_edges_that_share_an_adapter_port():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_deform(code, "XXIIXXII", require_distance=False)
    md = result.metadata
    graph = md["graph_incidence"].copy()
    design = md["graph_design"]
    first, second = design["adapters"][0]["edge_rows"]
    left = design["components"][0]["vertices"]
    graph[second, left] = graph[first, left]
    with pytest.raises(ValueError, match="bijective"):
        verify_component_adapter_expansion(graph, md["ports"], design, code.distance)


def test_component_certificate_rejects_overlapping_reported_components():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    result = gauging_deform(code, "XXIIXXII", require_distance=False)
    md = result.metadata
    design = deepcopy(md["graph_design"])
    design["components"][1]["vertices"][0] = design["components"][0]["vertices"][0]
    with pytest.raises(ValueError, match="overlap"):
        verify_component_adapter_expansion(md["graph_incidence"], md["ports"], design, code.distance)


@pytest.mark.parametrize("require_distance", [False, True])
def test_pair_summary_includes_internal_component_and_outer_adapter_qubits(require_distance):
    code = direct_sum(*(four_qubit_code() for _ in range(3)))
    left, right = "XXIIXXIIIIII", "IIIIIIIIXXII"
    result = gauging_pair_deform(code, left, right, require_distance=require_distance)
    _assert_only_target(code, result, [left, right, "XXIIIIIIIIII", "IIIIXXIIIIII"])
    summary = result.summary()
    assert summary["adapter_qubits"] == 4
    assert summary["component_adapter_qubits"] == 2
    assert summary["initial_graph_components"] == 3
    assert summary["initial_component_logical_rank"] == 3
    assert result.metadata["factors"][0]["graph_design"]["component_count"] == 2
    assert result.metadata["factors"][1]["graph_design"]["component_count"] == 1


def test_outer_adapter_recomputes_nested_component_proof_without_full_graph_enumeration():
    code = direct_sum(*(four_qubit_code() for _ in range(3)))
    result = gauging_pair_deform(code, "XXIIXXIIIIII", "IIIIIIIIXXII",
                                 exact_limit=2, require_distance=False)
    _assert_only_target(code, result, ["XXIIXXIIIIII", "IIIIIIIIXXII"])
    outer = result.metadata["adapter_distance_proof"]
    assert outer["condition_met"]
    left, right = outer["factor_relative_expansion"]
    assert not left["exact"] and left["condition_met"]
    assert right["exact"] and right["condition_met"]
    nested = left["component_adapter_proof"]
    assert nested["condition_met"] and nested["verified_against_actual_graph"]
    assert nested["component_count"] == 2
    assert nested["adapters"][0]["adapter_size"] == 2
    assert all(item["condition_met"] for item in nested["factor_relative_expansion"])
    assert not result.metadata["relative_expansion"]["exact"]
    assert result.metadata["relative_expansion"]["condition_met"]
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] == 2
