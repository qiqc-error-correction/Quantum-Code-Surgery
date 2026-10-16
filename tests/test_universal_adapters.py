"""Independent graph, cycle-space and Pauli-readout checks."""
import itertools
import json
import numpy as np
import pytest
from codesurgery.universal_adapters import skip_tree, repetition_adapter, relative_expansion
from codesurgery.universal_adapters.graphs import cycle_basis
from codesurgery.explicit_gadgets import gadget_deform, gadget_pair_deform, GaugeMeasurementRequired
from codesurgery.gauging_measurement import gauging_deform, gauging_pair_deform
from codesurgery.models import CSSCode
from codesurgery.examples import four_qubit_code, direct_sum
from codesurgery.gf2 import in_rowspace, rank, nullspace
from codesurgery.symplectic import pauli_product


def graph(n, edges):
    f = np.zeros((len(edges), n), dtype=np.uint8)
    for i, (u, v) in enumerate(edges):
        f[i, [u, v]] = 1
    return f


def assert_only_target(code, result):
    assert result.validate()
    x, z = result.check_x, result.check_z
    u = nullspace(np.hstack([x[:, code.n:], z[:, code.n:]]).T)
    image = (u @ np.hstack([x[:, :code.n], z[:, :code.n]])) % 2
    old = np.block([[code.hx, np.zeros_like(code.hx)], [np.zeros_like(code.hz), code.hz]])
    target = np.concatenate([result.measurement_x, result.measurement_z])
    assert rank(np.vstack([old, image])) == rank(old) + 1
    assert in_rowspace(image, np.vstack([old, target]))
    assert in_rowspace(target, image)
    for dep in nullspace(np.hstack([x, z]).T):
        assert pauli_product(x, z, result.check_signs, np.flatnonzero(dep))[2] == 0
    px, pz, phase = pauli_product(x, z, result.check_signs, np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(px[:code.n], result.measurement_x)
    np.testing.assert_array_equal(pz[:code.n], result.measurement_z)
    assert not np.any(px[code.n:] | pz[code.n:])
    assert phase == 2 * result.measurement_sign
    assert result.k_stabilizer == code.k - 1


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 25])
@pytest.mark.parametrize("kind", ["path", "star", "random"])
def test_skip_tree_sparse_change_of_basis(n, kind):
    rng = np.random.default_rng(734 + n)
    edges = [(i, i - 1 if kind == "path" else 0 if kind == "star" else int(rng.integers(i))) for i in range(1, n)]
    if n > 2:
        edges.extend([(0, 1), (0, n - 1)])  # Includes parallel edges.
    f = graph(n, edges)
    result = skip_tree(f)
    t, p, h = result["T"], result["P"], result["H"]
    assert np.array_equal((t @ f @ p) % 2, h)
    assert rank(h) == n - 1
    assert set(result["order"]) == set(range(n))
    assert np.all(t.sum(axis=1) <= 3) and np.all(t.sum(axis=0) <= 2)


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_adapter_nonisomorphic_unequal_graphs_preserves_full_sparse_cycle_space(size):
    l = graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 1)])
    r = graph(8, [(0, v) for v in range(1, 8)] + [(2, 3)])
    nl, nr = cycle_basis(l), cycle_basis(r)
    result = repetition_adapter(l, r, adapter_size=size)
    f, cycles = result["incidence"], result["cycle_basis"]
    assert not np.any((cycles @ f) % 2)
    assert rank(cycles) == len(f) - f.shape[1] + 1
    assert np.array_equal(cycles[:len(nl), :len(l)], nl)
    connector = cycles[result["connector_cycle_rows"]]
    assert np.all(connector.sum(axis=1) <= 8)
    assert np.all(connector.sum(axis=0) <= 2)
    assert np.array_equal((result["T_left"] @ l) % 2, (result["H_C"] @ result["P_left"].T) % 2)
    assert np.array_equal((result["T_right"] @ r) % 2, (result["H_C"] @ result["P_right"].T) % 2)


def test_disconnected_adapter_subset_and_hyperedge_rejected():
    f = graph(4, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(ValueError, match="connected"):
        repetition_adapter(f, f, adapter_size=2, left_vertices=[0, 3])
    with pytest.raises(ValueError, match="weight"):
        skip_tree([[1, 1, 1, 1]])


def test_relative_expansion_uses_ports_not_dummy_vertices():
    f = graph(7, [(i, i + 1) for i in range(6)])
    report = relative_expansion(f, [0, 6], distance=2)
    assert report["beta"] == 1
    global_report = relative_expansion(f, list(range(7)))
    assert global_report["beta"] == pytest.approx(1 / 3)
    unknown = relative_expansion(f, list(range(7)), exact_limit=3)
    assert not unknown["exact"] and not unknown["condition_met"]


def test_less_than_2d_shortcut_and_equality_requires_precheck():
    c = four_qubit_code()
    result = gadget_deform(c, "XXII", require_distance=False)
    assert result.metadata["precheck"]["method"] == "weight_lt_2d"
    assert not result.metadata["precheck"]["kernel_enumerated"]
    # weight=2d: two independent branch readouts must be detected.
    joined = direct_sum(c, c)
    with pytest.raises(GaugeMeasurementRequired, match="unintended") as error:
        gadget_deform(joined, "XXIIXXII", require_distance=False)
    assert error.value.details["logical_rank"] == 2
    unknown_distance = CSSCode(c.hx, c.hz)
    checked = gadget_deform(unknown_distance, "XXII", require_distance=False)
    assert checked.metadata["precheck"]["method"] == "kernel_signature"


def test_single_hyperedge_extra_readout_rejected_but_stabilizer_component_allowed():
    c = CSSCode([[1, 1, 1, 1, 0]], np.zeros((0, 5), dtype=np.uint8))
    with pytest.raises(GaugeMeasurementRequired, match="unintended"):
        gadget_deform(c, "ZZZZI", require_distance=False)
    code = CSSCode([[1, 1, 1, 1, 0]], [[1, 1, 1, 1, 0], [0, 0, 0, 0, 1]], distance=2)
    # Weight three <2d, but the Z5 stabilizer is an isolated support component.
    result = gadget_deform(code, "ZZIIZ", require_distance=False)
    assert_only_target(code, result)
    assert result.metadata["precheck"]["method"] == "weight_lt_2d"
    assert not result.metadata["distance_preservation_guaranteed"]


@pytest.mark.parametrize("factor_type", ["X", "Z", "Y"])
@pytest.mark.parametrize("route", [gadget_pair_deform, gauging_pair_deform])
def test_disjoint_xz_and_y_products_do_not_measure_either_factor(factor_type, route):
    c = direct_sum(four_qubit_code(), four_qubit_code())
    left = "-" + ("YXZI" if factor_type == "Y" else factor_type * 2 + "II") + "IIII"
    right = "IIIIZZII"
    result = route(c, left, right, require_distance=False)
    assert_only_target(c, result)
    assert result.measurement_sign == 1
    from codesurgery.symplectic import parse_pauli
    x, z, _ = parse_pauli(left, c.n)
    factor = np.concatenate([np.pad(x, (0, result.n - c.n)), np.pad(z, (0, result.n - c.n))])
    assert not in_rowspace(factor, np.hstack([result.check_x, result.check_z]))
    assert result.metadata["adapter"]["connector_max_weight"] <= 8


def test_each_factor_checked_before_adapter():
    c = direct_sum(direct_sum(four_qubit_code(), four_qubit_code()), four_qubit_code())
    with pytest.raises(GaugeMeasurementRequired, match="unintended"):
        gadget_pair_deform(c, "XXIIXXIIIIII", "IIIIIIIIZZII", require_distance=False)
    result = gauging_pair_deform(c, "XXIIXXIIIIII", "IIIIIIIIZZII", require_distance=False)
    assert_only_target(c, result)


@pytest.mark.parametrize("target", ["ZZZZI", "-YYYYI"])
def test_gauging_splits_weight_four_boundaries_and_keeps_signed_target(target):
    c = CSSCode([[1, 1, 1, 1, 0]], np.zeros((0, 5), dtype=np.uint8))
    with pytest.raises(GaugeMeasurementRequired, match="gauging measurement"):
        gadget_deform(c, target, require_distance=False)
    result = gauging_deform(c, target, require_distance=False)
    assert_only_target(c, result)
    assert np.all(result.metadata["graph_incidence"].sum(axis=1) == 2)
    assert result.metadata["max_matching_weight"] == 2


def test_supplied_graph_dummy_ports_and_bad_matching():
    c = four_qubit_code()
    f = graph(3, [(0, 1), (1, 2)])
    result = gauging_deform(c, "-XXII", graph=f, ports=[0, 2], require_distance=False)
    assert_only_target(c, result)
    assert result.metadata["relative_expansion"]["condition_met"]
    with pytest.raises(ValueError, match="matching"):
        gauging_deform(c, "XXII", graph=f, ports=[0, 2], matching=np.zeros((2, 2), dtype=np.uint8), require_distance=False)
    with pytest.raises(ValueError, match="ports"):
        gauging_deform(c, "XXII", graph=f, ports=[0, 0], require_distance=False)


def test_custom_basis_default_y_and_logical_xz_coefficients():
    c = direct_sum(four_qubit_code(), four_qubit_code())
    standard = c.standard_logicals()
    permutation = [2, 3, 0, 1]
    basis = dict(jx=standard.jx[permutation], jz=standard.jz[permutation])
    y = gadget_deform(c, pauli="Y", logical_basis=basis, require_distance=False)
    assert_only_target(c, y)
    np.testing.assert_array_equal(y.measurement_x, basis["jx"][0])
    np.testing.assert_array_equal(y.measurement_z, basis["jz"][0])
    pair = gadget_pair_deform(c, {"logical_x": [1, 0, 0, 0]},
                             {"logical_z": [0, 0, 1, 0]}, logical_basis=basis, require_distance=False)
    assert_only_target(c, pair)
    invalid = dict(jx=basis["jx"], jz=basis["jz"][::-1])
    with pytest.raises(ValueError, match="JX JZ"):
        gadget_deform(c, logical_basis=invalid, require_distance=False)


def test_overlap_routes_to_single_hermitian_gauging_target():
    c = four_qubit_code()
    with pytest.raises(GaugeMeasurementRequired, match="overlapping"):
        gadget_pair_deform(c, "XXII", "ZIZI", require_distance=False)
    result = gauging_deform(c, "YXZI", require_distance=False)
    assert_only_target(c, result)


def test_adapter_composition_certificate_when_full_graph_enumeration_disabled():
    c = direct_sum(four_qubit_code(), four_qubit_code())
    result = gadget_pair_deform(c, "XXIIIIII", "IIIIZZII", exact_limit=2, require_distance=False)
    assert result.metadata["adapter_distance_proof"]["condition_met"]
    assert result.metadata["distance_lower_bound"] == 2
    assert not result.metadata["relative_expansion"]["exact"]


def test_no_false_distance_certificate_for_thin_adapter():
    c = direct_sum(four_qubit_code(), four_qubit_code())
    result = gadget_pair_deform(c, "XXIIIIII", "IIIIZZII", adapter_size=1, require_distance=False)
    assert_only_target(c, result)
    assert not result.metadata["adapter_distance_proof"]["condition_met"]
    assert not result.metadata["distance_preservation_guaranteed"]


def test_adapter_export_rejects_modified_cycles_or_certificate(tmp_path):
    c = direct_sum(four_qubit_code(), four_qubit_code())
    result = gadget_pair_deform(c, "XXIIIIII", "IIIIZZII", require_distance=False)
    path = tmp_path / "pair.json"
    result.save(path)
    before = path.read_bytes()
    result.metadata["adapter"]["T_left"][0, 0] ^= 1
    with pytest.raises(ValueError, match="changed"):
        result.save(path)
    assert path.read_bytes() == before


def test_all_nontrivial_four_qubit_logicals_with_both_signs_in_gauging_framework():
    c = four_qubit_code()
    checked = 0
    for letters in itertools.product("IXYZ", repeat=c.n):
        text = "".join(letters)
        from codesurgery.symplectic import parse_pauli
        x, z, _ = parse_pauli(text, c.n)
        if np.any((x @ c.hz.T) % 2) or np.any((z @ c.hx.T) % 2):
            continue
        if in_rowspace(x, c.hx) and in_rowspace(z, c.hz):
            continue
        for sign in ("", "-"):
            assert_only_target(c, gauging_deform(c, sign + text, require_distance=False))
            checked += 1
    assert checked == 120


def test_custom_logical_y_overlap_three_has_negative_literal_yyy():
    from codesurgery.gf2 import inverse
    c = CSSCode(np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8))
    jx = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=np.uint8)
    basis = dict(jx=jx, jz=inverse(jx).T)
    result = gauging_deform(c, pauli="Y", logical_basis=basis, require_distance=False)
    assert_only_target(c, result)
    assert np.all(result.measurement_x) and np.all(result.measurement_z)
    assert result.measurement_sign == 1


def test_hamming_xz_adapter_has_no_nontrivial_weight_one_or_two_pauli():
    h = np.array([[(j >> b) & 1 for j in range(1, 16)] for b in (3, 2, 1, 0)], dtype=np.uint8)
    c = CSSCode(h, h, 3)
    right = "".join("Z" if q in (4, 8, 12) else "I" for q in range(1, 16))
    result = gadget_pair_deform(c, "XXX" + "I" * 12, right, require_distance=False)
    assert_only_target(c, result)
    errors = []
    for weight in (1, 2):
        for support in itertools.combinations(range(result.n), weight):
            for types in itertools.product(((1, 0), (0, 1), (1, 1)), repeat=weight):
                e = np.zeros(2 * result.n, dtype=np.uint8)
                for q, (x, z) in zip(support, types):
                    e[q], e[result.n + q] = x, z
                errors.append(e)
    errors = np.asarray(errors)
    assert len(errors) == 2145
    syndrome = (errors[:, :result.n] @ result.check_z.T + errors[:, result.n:] @ result.check_x.T) % 2
    assert in_rowspace(errors[~syndrome.any(axis=1)], np.hstack([result.check_x, result.check_z]))


def test_cli_pair_and_gauging_routes_and_error_does_not_overwrite(tmp_path, capsys):
    from codesurgery.explicit_gadgets.cli import main
    c = direct_sum(four_qubit_code(), four_qubit_code())
    config = dict(hx=c.hx.tolist(), hz=c.hz.tolist(), distance=2,
                  logicals=["XXIIIIII", "IIIIZZII"])
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(config))
    assert main([str(source), "-o", str(output)]) == 0
    first = json.loads(output.read_text())
    assert first["summary"]["scheme"] == "webster_smith_cohen_with_universal_adapter"
    assert first["summary"]["logical_measurement_rank"] == 1
    assert main([str(source), "--scheme", "gauging", "-o", str(output)]) == 0
    assert json.loads(output.read_text())["summary"]["scheme"] == "gauging_measurement_with_universal_adapter"
    before = output.read_bytes()
    config.pop("logicals")
    config["logical"] = "XXIIZZII"
    source.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as error:
        main([str(source), "-o", str(output)])
    assert error.value.code == 2
    assert "gauging measurement" in capsys.readouterr().err
    assert output.read_bytes() == before


def test_compositional_distance_flag_cannot_be_forged():
    from codesurgery.gadget_common import prepare
    from codesurgery.gauging_measurement.api import build_graph
    c, target, basis = prepare(four_qubit_code(), "XXII")
    with pytest.raises(ValueError, match="actual adapter construction"):
        build_graph(c, target, basis, graph(2, [(0, 1)]),
                    extra_metadata={"adapter_distance_proof": {"condition_met": True}})
