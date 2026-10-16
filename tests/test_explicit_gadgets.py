"""Independent matrix, logical-readout, phase and distance QA for gadgets."""

import itertools
import json

import numpy as np
import pytest

from codesurgery.examples import direct_sum, four_qubit_code
from codesurgery.explicit_gadgets import GaugeMeasurementRequired, exact_cheeger, gadget_deform, gadget_pair_deform
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.explicit_gadgets.expansion import expand_restriction
from codesurgery.gf2 import in_rowspace, matmul, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.symplectic import commutation_matrix, pauli_product


def hamming15_code():
    # Every nonzero four-bit column appears once. These rows are pairwise
    # orthogonal, giving the [[15,7,3]] CSS quantum Hamming code.
    h = np.array([[(j >> bit) & 1 for j in range(1, 16)]
                  for bit in (3, 2, 1, 0)], dtype=np.uint8)
    return CSSCode(h, h, distance=3, name="Hamming [[15,7,3]]")


def original_stabilizers(code):
    return np.block([[code.hx, np.zeros_like(code.hx)],
                     [np.zeros_like(code.hz), code.hz]])


def assert_exact_measurement(code, result):
    """Eliminate all actual ancillas, rather than trusting metadata."""
    assert result.validate()
    cx, cz = result.check_x, result.check_z
    assert not np.any(commutation_matrix(cx, cz))
    coefficients = nullspace(np.hstack((cx[:, code.n:], cz[:, code.n:])).T)
    memory = matmul(coefficients, np.hstack((cx[:, :code.n], cz[:, :code.n])))
    old = original_stabilizers(code)
    target = np.concatenate((result.measurement_x, result.measurement_z))
    assert rank(np.vstack((old, memory))) - rank(old) == 1
    assert in_rowspace(memory, np.vstack((old, target)))
    assert in_rowspace(target, np.vstack((old, memory)))
    assert result.protected_k == code.k - 1
    assert result.k_stabilizer == result.protected_k
    assert result.gauge_qubits == 0

    # Every binary generator dependency must be +I, including dependencies
    # involving canonical Y phases and redundant original generators.
    for dependency in nullspace(np.hstack((cx, cz)).T):
        x, z, phase = pauli_product(cx, cz, result.check_signs,
                                    np.flatnonzero(dependency))
        assert not np.any(x | z)
        assert phase == 0
    x, z, phase = pauli_product(cx, cz, result.check_signs,
                                np.flatnonzero(result.measurement_witness))
    np.testing.assert_array_equal(x, np.pad(result.measurement_x, (0, result.n - code.n)))
    np.testing.assert_array_equal(z, np.pad(result.measurement_z, (0, result.n - code.n)))
    assert phase == 2 * result.measurement_sign
    assert "gadget" in result.summary()["scheme"] or "gauging" in result.summary()["scheme"] or "adapter" in result.summary()["scheme"]


def literal_pauli(x, z, sign=0):
    matrices = {(0, 0): np.eye(2, dtype=complex),
                (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
                (0, 1): np.diag([1, -1]).astype(complex),
                (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex)}
    out = np.array([[(-1) ** sign]], dtype=complex)
    for a, b in zip(x, z):
        out = np.kron(out, matrices[int(a), int(b)])
    return out


def independent_boundary_ratio(f):
    """Definition of h, independently enumerated for our tiny examples."""
    w = f.shape[1]
    values = []
    for size in range(1, w // 2 + 1):
        for subset in itertools.combinations(range(w), size):
            values.append(int((f[:, subset].sum(axis=1) % 2).sum()) / size)
    return min(values, default=float("inf"))


def test_pure_x_is_exact_single_layer_paper_check_group():
    code = four_qubit_code()
    result = gadget_deform(code, "XXII", expansion="none", require_distance=False)
    assert_exact_measurement(code, result)
    # Written directly from the paper, with one ancilla for its touching
    # Z check. Compare groups, allowing the implementation any row order.
    hx = np.array([[1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [0, 1, 0, 0, 1]], dtype=np.uint8)
    hz = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    expected = np.block([[hx, np.zeros_like(hx)],
                         [np.zeros_like(hz), hz]])
    actual = np.hstack((result.check_x, result.check_z))
    assert result.n == 5
    assert rank(actual) == 4
    assert in_rowspace(actual, expected) and in_rowspace(expected, actual)
    np.testing.assert_array_equal(result.metadata["restriction_matrix"], [[1, 1]])
    assert np.asarray(result.metadata["cycle_basis"]).shape == (0, 1)


@pytest.mark.parametrize("pauli", ["X", "Z", "Y"])
def test_default_is_one_paired_standard_form_logical(pauli):
    code = four_qubit_code()
    for index in range(code.k):
        result = gadget_deform(code, pauli=pauli, logical_index=index, require_distance=False)
        assert_exact_measurement(code, result)
        basis = code.standard_logicals()
        x = basis.jx[index] if pauli in ("X", "Y") else np.zeros(code.n, dtype=np.uint8)
        z = basis.jz[index] if pauli in ("Z", "Y") else np.zeros(code.n, dtype=np.uint8)
        expected = literal_pauli(x, np.zeros(code.n, dtype=np.uint8))
        if pauli == "Y":
            expected = 1j * expected @ literal_pauli(np.zeros(code.n, dtype=np.uint8), z)
        elif pauli == "Z":
            expected = literal_pauli(np.zeros(code.n, dtype=np.uint8), z)
        np.testing.assert_array_equal(literal_pauli(result.measurement_x,
                                      result.measurement_z, result.measurement_sign), expected)


def test_signed_three_y_product_matches_i_xxx_zzz_literal_phase():
    # i*XXX*ZZZ is -YYY. Default standard-form representatives overlap on
    # exactly one coordinate; this explicitly supplied representative pair
    # exercises a different physical phase without overriding the API.
    code = CSSCode(np.zeros((0, 3), dtype=np.uint8),
                   np.zeros((0, 3), dtype=np.uint8))
    result = gauging_deform(code, "-YYY", require_distance=False)
    assert_exact_measurement(code, result)
    assert result.measurement_sign == 1
    expected = 1j * literal_pauli([1, 1, 1], [0, 0, 0]) @ literal_pauli([0, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(literal_pauli(result.measurement_x,
                                  result.measurement_z, result.measurement_sign), expected)


@pytest.mark.parametrize("target", ["Y", "-Y"])
def test_bare_single_qubit_signed_y_literal_matrix(target):
    code = CSSCode(np.zeros((0, 1), dtype=np.uint8), np.zeros((0, 1), dtype=np.uint8))
    result = gadget_deform(code, target, require_distance=False)
    assert_exact_measurement(code, result)
    assert result.n == 1
    product = np.eye(2, dtype=complex)
    for i in np.flatnonzero(result.measurement_witness):
        product = product @ literal_pauli(result.check_x[i], result.check_z[i],
                                          int(result.check_signs[i]))
    np.testing.assert_array_equal(product, literal_pauli([1], [1], int(target.startswith("-"))))


def test_hamming_mixed_raw_path_has_small_expansion_without_distance_claim():
    code = hamming15_code()
    result = gadget_deform(code, "YXXZZ" + "I" * 10, expansion="none", require_distance=False)
    assert_exact_measurement(code, result)
    assert result.n == 19
    assert rank(np.hstack((result.check_x, result.check_z))) == 13
    assert result.protected_k == 6
    f = np.asarray(result.metadata["restriction_matrix"], dtype=np.uint8)
    assert f.shape == (4, 5)
    assert independent_boundary_ratio(f) == 0.5
    assert not result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] is None
    assert not result.metadata["automatic_expansion"]


def test_hamming_mixed_expansion_preserves_distance_against_every_weight_two_pauli():
    code = hamming15_code()
    result = gadget_deform(code, "-YXXZZ" + "I" * 10, expansion="auto", require_distance=False)
    assert_exact_measurement(code, result)
    f = np.asarray(result.metadata["restriction_matrix"], dtype=np.uint8)
    assert independent_boundary_ratio(f) >= 1
    assert result.metadata["distance_preservation_guaranteed"]
    assert result.metadata["distance_lower_bound"] == 3
    assert result.metadata["automatic_expansion"]
    # Exhaust every X/Y/Z error on every single site and pair. A syndrome
    # free error of weight <3 must be a stabilizer, not a surviving logical.
    candidates = []
    for size in (1, 2):
        for support in itertools.combinations(range(result.n), size):
            for types in itertools.product(((1, 0), (0, 1), (1, 1)), repeat=size):
                error = np.zeros(2 * result.n, dtype=np.uint8)
                for q, (x, z) in zip(support, types):
                    error[q], error[result.n + q] = x, z
                candidates.append(error)
    candidates = np.asarray(candidates, dtype=np.uint8)
    syndrome = (candidates[:, :result.n] @ result.check_z.T
                + candidates[:, result.n:] @ result.check_x.T) % 2
    undetected = candidates[~np.any(syndrome, axis=1)]
    assert in_rowspace(undetected, np.hstack((result.check_x, result.check_z)))


@pytest.mark.parametrize("expansion", ["none", "auto", "star", "complete"])
@pytest.mark.parametrize("target", ["ZZIIZZII", "XXIIZZII"])
def test_extra_readout_is_rejected_before_any_optional_expansion(expansion, target):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    with pytest.raises(GaugeMeasurementRequired, match="unintended"):
        gadget_deform(code, target, expansion=expansion, require_distance=False)
    left, right = target[:4] + "IIII", "IIII" + target[4:]
    result = gadget_pair_deform(code, left, right, require_distance=False)
    assert_exact_measurement(code, result)
    factor = np.zeros(2 * result.n, dtype=np.uint8)
    offset = 0 if target.startswith("X") else result.n
    factor[offset:offset + 2] = 1
    assert not in_rowspace(factor, np.hstack((result.check_x, result.check_z)))


def test_extra_rows_cannot_bypass_raw_readout_rejection():
    code = direct_sum(four_qubit_code(), four_qubit_code())
    edges = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    with pytest.raises(GaugeMeasurementRequired, match="unintended"):
        gadget_deform(code, "ZZIIZZII", extra_rows=edges, require_distance=False)


def test_actual_gauge_dimension_with_redundant_and_zero_checks():
    code = CSSCode([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                   [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], distance=2)
    result = gadget_deform(code, "XXII", expansion="none", require_distance=False)
    assert_exact_measurement(code, result)
    f = np.asarray(result.metadata["restriction_matrix"], dtype=np.uint8)
    g = np.asarray(result.metadata["cycle_basis"], dtype=np.uint8)
    assert f.shape == (2, 2)
    assert rank(g) == len(f) - rank(f) == 1
    assert not np.any(matmul(g, f))
    assert result.n == 6


@pytest.mark.parametrize("expansion", ["none", "auto"])
@pytest.mark.parametrize("sign", ["", "-"])
def test_supported_negative_stabilizer_relation_does_not_create_minus_identity(expansion, sign):
    # XX*ZZ=-YY on the first two sites. The supported kernel has a known
    # stabilizer direction in addition to the one logical Y on site three.
    h = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    code = CSSCode(h, h)
    result = gadget_deform(code, sign + "YYY", expansion=expansion, require_distance=False)
    assert_exact_measurement(code, result)
    assert result.measurement_sign == int(bool(sign))


@pytest.mark.parametrize("target", ["IIII", "XXXX", "XIII", "YIII", "ZXII",
                                   [2, 0, 0, 0], [1, 0], {"x": [1, 0, 0, 0], "z": [0, 0, 0, 0]}])
def test_rejects_zero_stabilizer_noncommuting_and_malformed_targets(target):
    with pytest.raises(ValueError):
        gadget_deform(four_qubit_code(), target, require_distance=False)


@pytest.mark.parametrize("rows", [[[1, 0]], [[2, 0]], [[1, 1, 0]], [1, 1]])
def test_rejects_invalid_extra_rows(rows):
    with pytest.raises(ValueError):
        gadget_deform(four_qubit_code(), "XXII", extra_rows=rows, require_distance=False)


def test_unknown_distance_is_not_invented():
    h = [[1, 1, 1, 1]]
    result = gadget_deform(CSSCode(h, h), "XXII", require_distance=False)
    assert result.metadata["distance_lower_bound"] is None


@pytest.mark.parametrize("suffix", [".json", ".npz"])
def test_export_validates_mutated_phase_before_overwriting(tmp_path, suffix):
    result = gadget_deform(four_qubit_code(), "-XXII", require_distance=False)
    output = tmp_path / ("gadget" + suffix)
    result.save(output)
    before = output.read_bytes()
    if suffix == ".json":
        payload = json.loads(before)
        assert payload["summary"]["scheme"] == "webster_smith_cohen_explicit_gadget"
        assert payload["measurement_sign"] == 1
    else:
        with np.load(output, allow_pickle=False) as payload:
            np.testing.assert_array_equal(payload["check_x"], result.check_x)
    witness_row = int(np.flatnonzero(result.measurement_witness)[0])
    result.check_signs[witness_row] ^= 1
    with pytest.raises(ValueError):
        result.save(output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("budget", [1, 10, 100])
def test_dense_memory_and_validation_allocations_are_bounded(budget):
    with pytest.raises(ValueError, match="max_entries"):
        gadget_deform(hamming15_code(), "YXXZZ" + "I" * 10, max_entries=budget, require_distance=False)


def test_rejects_unknown_mode_and_invalid_logical_index():
    with pytest.raises(ValueError):
        gadget_deform(four_qubit_code(), expansion="unknown", require_distance=False)
    for index in (-1, 2, True, 0.5):
        with pytest.raises(ValueError):
            gadget_deform(four_qubit_code(), logical_index=index, require_distance=False)


def test_exact_cheeger_counts_parity_boundaries_and_parallel_ancillas():
    # A weight-four hyperedge touches {0,1} twice, hence contributes zero
    # boundary. Ordinary neighbor counting would incorrectly return >0.
    hyperedge = np.ones((1, 4), dtype=np.uint8)
    report = exact_cheeger(hyperedge)
    assert report["exact"] and report["h"] == 0
    assert not report["distance_condition_met"]
    # Parallel rows are distinct ancillas and must all count, even though
    # a GF(2) row-basis reduction would retain only one row.
    parallel = np.ones((3, 2), dtype=np.uint8)
    report = exact_cheeger(parallel)
    assert report["h"] == 3
    assert report["distance_condition_met"]


def test_exact_cheeger_reports_true_worst_cut_and_refuses_sampling():
    path = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=np.uint8)
    report = exact_cheeger(path)
    assert report["h"] == independent_boundary_ratio(path) == 0.5
    selected = report["worst_cut"]
    assert len(selected) == report["cut_size"] == 2
    assert int((path[:, selected].sum(axis=1) % 2).sum()) == report["boundary_weight"] == 1
    with pytest.raises(ValueError, match="exact_limit"):
        exact_cheeger(path, exact_limit=4)


def test_legacy_star_helper_requires_explicit_selection_and_cannot_hide_extra_readout():
    code = CSSCode(np.zeros((0, 7), dtype=np.uint8), np.zeros((0, 7), dtype=np.uint8))
    with pytest.raises(GaugeMeasurementRequired, match="unintended"):
        gadget_deform(code, "X" * 7, expansion="auto", exact_limit=3, require_distance=False)
    f, metadata = expand_restriction(np.zeros((0, 7), dtype=np.uint8), mode="star", exact_limit=3)
    report = metadata["final_cheeger"]
    assert not report["exact"] and report["h"] is None
    assert report["h_lower_bound"] >= 1
    assert independent_boundary_ratio(f) >= 1


def test_graph_edits_cannot_export_a_stale_distance_certificate(tmp_path):
    result = gadget_deform(hamming15_code(), "YXXZZ" + "I" * 10, expansion="auto", require_distance=False)
    output = tmp_path / "gadget.json"
    result.save(output)
    before = output.read_bytes()
    f = result.metadata["restriction_matrix"]
    assert independent_boundary_ratio(f) >= 1
    # Remove the automatically added edge while keeping the same number of
    # qubits. Update the actual check rows and full cycle closure as well:
    # this remains a valid rank-one measurement, but its old expansion
    # certificate no longer applies. Check-group rebuilding alone cannot
    # catch this mutation, so the result must reject its changed construction.
    f[-1] = 0
    g = nullspace(f.T)
    result.metadata["cycle_basis"] = g
    old_rows = len(hamming15_code().hx) + len(hamming15_code().hz)
    w = f.shape[1]
    result.check_x[old_rows:old_rows + w, result.original_n:] = f.T
    result.check_z[old_rows + w:, result.original_n:] = g
    assert independent_boundary_ratio(f) == 0.5
    assert not np.any(commutation_matrix(result.check_x, result.check_z))
    for operation in (result.validate, result.summary, lambda: result.save(output)):
        with pytest.raises(ValueError):
            operation()
    assert output.read_bytes() == before


@pytest.mark.parametrize("field", ["cheeger_constant", "distance_lower_bound",
                                   "max_check_weight", "max_qubit_degree"])
def test_public_certificates_cannot_be_mutated_independently_of_checks(field):
    result = gadget_deform(hamming15_code(), "YXXZZ" + "I" * 10, expansion="auto", require_distance=False)
    result.metadata[field] += 1
    with pytest.raises(ValueError):
        result.validate()
    with pytest.raises(ValueError):
        result.summary()


def test_source_resource_guard_runs_before_clone_allocates(monkeypatch):
    code = hamming15_code()
    clone_calls = []

    def forbidden_clone(self):
        clone_calls.append(self)
        raise AssertionError("source allocation occurred before its resource guard")

    # Dataclass initialization calls __post_init__, which copies the input
    # matrices. Instrument it only after the caller's source already exists.
    monkeypatch.setattr(CSSCode, "__post_init__", forbidden_clone)
    with pytest.raises(ValueError, match="max_entries"):
        gadget_deform(code, "YXXZZ" + "I" * 10, max_entries=1, require_distance=False)
    assert not clone_calls


@pytest.mark.parametrize("target", [None, "XIIIIIIIII"])
def test_full_logical_basis_is_bounded_before_default_or_explicit_target(monkeypatch, target):
    # The one-row output would require only 20 entries, but the two paired
    # logical matrices require 200 entries. The basis is required even for
    # an explicit target by the complete logical-readout-space check.
    code = CSSCode(np.zeros((0, 10), dtype=np.uint8), np.zeros((0, 10), dtype=np.uint8))
    basis_calls = []

    def forbidden_basis(self):
        basis_calls.append(self)
        raise AssertionError("full logical basis was allocated before its resource guard")

    monkeypatch.setattr(CSSCode, "standard_logicals", forbidden_basis)
    with pytest.raises(ValueError, match="max_entries"):
        gadget_deform(code, target, max_entries=100, require_distance=False)
    assert not basis_calls
