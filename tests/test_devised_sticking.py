"""Independent glue-space, simultaneous-readout and phase QA for Zhang--Li."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from codesurgery import CSSCode
from codesurgery.devised_sticking import build_devised_glue, devised_deform
from codesurgery.examples import direct_sum, four_qubit_code
from codesurgery.gf2 import in_rowspace, matmul, nullspace, rank
from codesurgery.symplectic import pauli_product


def max_degree(matrix):
    return max(int(matrix.sum(axis=0).max(initial=0)), int(matrix.sum(axis=1).max(initial=0)))


class DevisedStickingTests(unittest.TestCase):
    def assert_glue(self, code, glue, targets, pauli):
        opposite = code.hx if pauli == "Z" else code.hz
        conjugate = code.standard_logicals().jx if pauli == "Z" else code.standard_logicals().jz
        h_g, s, t = glue.h_g, glue.s, glue.t
        self.assertEqual(s.shape, (h_g.shape[1], code.n))
        self.assertEqual(t.shape, (len(opposite), len(h_g)))
        np.testing.assert_array_equal(matmul(opposite, s.T), matmul(t, h_g))
        kernel = nullspace(h_g)
        memory_image = matmul(kernel, s)
        self.assertFalse(np.any(matmul(memory_image, opposite.T)))
        # Duplication creates unique extensions of naked-support words,
        # rather than introducing internal kernel vectors invisible to data.
        self.assertEqual(rank(memory_image), len(kernel))
        self.assertTrue(in_rowspace(targets, memory_image))
        signature = matmul(memory_image, conjugate.T)
        target_signature = matmul(targets, conjugate.T)
        self.assertEqual(rank(signature), len(targets))
        self.assertTrue(in_rowspace(signature, target_signature))
        self.assertTrue(in_rowspace(target_signature, signature))

    def assert_result(self, code, result, pauli):
        self.assertTrue(result.validate())
        targets = result.measurement_z if pauli == "Z" else result.measurement_x
        q = len(targets)
        self.assertEqual(targets.shape, (q, code.n))
        self.assertEqual(result.measurement_signs.shape, (q,))
        self.assertEqual(result.measurement_witnesses.shape, (q, len(result.check_x)))
        self.assertEqual(result.protected_k, code.k - q)
        self.assertEqual(result.gauge_qubits, result.k_stabilizer - result.protected_k)
        self.assertGreaterEqual(result.gauge_qubits, 0)
        self.assertFalse(np.any(matmul(result.hx, result.hz.T)))
        self.assert_glue(code, result.glue, targets, pauli)
        for x, z, sign, witness in zip(result.measurement_x, result.measurement_z,
                                       result.measurement_signs, result.measurement_witnesses):
            wx, wz, phase = pauli_product(result.check_x, result.check_z, result.check_signs,
                                          np.flatnonzero(witness))
            np.testing.assert_array_equal(wx, np.pad(x, (0, result.n - code.n)))
            np.testing.assert_array_equal(wz, np.pad(z, (0, result.n - code.n)))
            self.assertEqual(phase, 2 * int(sign))
        full = np.hstack((result.check_x, result.check_z))
        for dependency in nullspace(full.T):
            x, z, phase = pauli_product(result.check_x, result.check_z, result.check_signs,
                                        np.flatnonzero(dependency))
            self.assertFalse(np.any(x | z))
            self.assertEqual(phase, 0)
        # Independently eliminate every ancillary X and Z component in the
        # *actual output*, then compare complete logical spaces, not counts.
        aux = np.hstack((result.check_x[:, code.n:], result.check_z[:, code.n:]))
        coefficients = nullspace(aux.T)
        memory_x = matmul(coefficients, result.check_x[:, :code.n])
        memory_z = matmul(coefficients, result.check_z[:, :code.n])
        self.assertFalse(np.any(matmul(memory_x, code.hz.T)))
        self.assertFalse(np.any(matmul(memory_z, code.hx.T)))
        basis = code.standard_logicals()
        signature = np.hstack((matmul(memory_x, basis.jz.T), matmul(memory_z, basis.jx.T)))
        expected = np.hstack((matmul(result.measurement_x, basis.jz.T),
                              matmul(result.measurement_z, basis.jx.T)))
        self.assertEqual(rank(expected), q)
        self.assertEqual(rank(signature), q)
        self.assertTrue(in_rowspace(signature, expected))
        self.assertTrue(in_rowspace(expected, signature))
        px, pz = result.protected_x, result.protected_z
        self.assertEqual(px.shape, (code.k - q, result.n))
        self.assertEqual(pz.shape, px.shape)
        self.assertFalse(np.any(matmul(px, result.hz.T)))
        self.assertFalse(np.any(matmul(pz, result.hx.T)))
        np.testing.assert_array_equal(matmul(px, pz.T), np.eye(code.k - q))

    def test_overlapping_simultaneous_targets_and_x_z_duality(self):
        code = four_qubit_code()
        targets = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
        for d_r in (1, 2):
            z = devised_deform(code, ["ZZII", "ZIZI"], d_R=d_r)
            x = devised_deform(code, ["XXII", "XIXI"], pauli="X", d_R=d_r)
            self.assert_result(code, z, "Z")
            self.assert_result(code, x, "X")
            np.testing.assert_array_equal(z.measurement_z, targets)
            np.testing.assert_array_equal(x.measurement_x, targets)
            np.testing.assert_array_equal(z.hx, x.hz)
            np.testing.assert_array_equal(z.hz, x.hx)

    def test_glue_removes_extra_union_factor_readout(self):
        code = direct_sum(four_qubit_code(), four_qubit_code())
        target = np.array([[1, 1, 0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        support = np.flatnonzero(target[0])
        naked_kernel = nullspace(code.hx[:, support])
        naked_image = np.zeros((len(naked_kernel), code.n), dtype=np.uint8)
        naked_image[:, support] = naked_kernel
        self.assertEqual(rank(matmul(naked_image, code.standard_logicals().jx.T)), 2)
        for d_r in (1, 2):
            result = devised_deform(code, "ZZIIZZII", d_R=d_r)
            self.assert_result(code, result, "Z")
            first_factor = np.pad([1, 1, 0, 0, 0, 0, 0, 0], (0, result.n - code.n))
            self.assertFalse(in_rowspace(first_factor, result.hz))

    def test_default_without_logical_redundancy_keeps_naked_glue(self):
        code = four_qubit_code()
        for pauli in ("X", "Z"):
            target = code.logical(pauli)[None, :]
            support = np.flatnonzero(target[0])
            opposite = code.hx if pauli == "Z" else code.hz
            incident = np.flatnonzero(np.any(opposite[:, support], axis=1))
            naked = opposite[np.ix_(incident, support)]
            for ldpc in (False, True):
                glue = build_devised_glue(code, target, pauli=pauli, ldpc=ldpc)
                self.assert_glue(code, glue, target, pauli)
                np.testing.assert_array_equal(glue.h_g, naked)
                result = devised_deform(code, pauli=pauli, d_R=1, ldpc=ldpc)
                self.assert_result(code, result, pauli)

    def test_stabilizer_decorated_support_and_signed_witnesses(self):
        # Preserve the supported 110 stabilizer direction as well as logical
        # 001; a literal kernel-image==target-span check would be too strong.
        code = CSSCode([[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                       [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
        for pauli in ("X", "Z"):
            result = devised_deform(code, "-" + pauli * 3, pauli=pauli, d_R=2)
            self.assert_result(code, result, pauli)
            self.assertEqual(result.measurement_signs.tolist(), [1])
            image = matmul(nullspace(result.glue.h_g), result.glue.s)
            self.assertTrue(in_rowspace([1, 1, 0], image))
            self.assertEqual(rank(image), 2)
        signed = devised_deform(four_qubit_code(), ["-ZZII", "ZIZI"], d_R=2)
        self.assert_result(four_qubit_code(), signed, "Z")
        self.assertEqual(signed.measurement_signs.tolist(), [1, 0])

    def test_dense_dressing_sparsification_preserves_fine_space_and_degree_bound(self):
        code = CSSCode(np.zeros((0, 9)), np.zeros((0, 9)))
        # Independent but heavily overlapping rows require both bit-degree
        # reduction and splitting dense dressing checks.
        targets = np.hstack((np.eye(4, dtype=np.uint8), np.ones((4, 5), dtype=np.uint8)))
        dense = build_devised_glue(code, targets, ldpc=False)
        sparse = build_devised_glue(code, targets, ldpc=True)
        self.assert_glue(code, dense, targets, "Z")
        self.assert_glue(code, sparse, targets, "Z")
        self.assertGreater(max_degree(dense.h_g), 3)
        self.assertLessEqual(max_degree(sparse.h_g), 3)
        self.assertGreater(sparse.h_g.shape[1], dense.h_g.shape[1])
        self.assertEqual(len(nullspace(dense.h_g)), len(nullspace(sparse.h_g)))
        self.assertLessEqual(max_degree(sparse.s), 1)
        self.assertLessEqual(max_degree(sparse.t), 1)
        result = devised_deform(code, targets, d_R=1)
        self.assert_result(code, result, "Z")

    def test_random_canonical_and_stabilizer_decorated_target_spaces(self):
        # Random CSS spaces exercise nonsymmetric basis transforms, general
        # pivot positions and support-contained stabilizers. Target rows are
        # independent logical combinations, rather than only basis rows.
        rng = np.random.default_rng(20261012)
        checked = 0
        for n in range(3, 8):
            for trial in range(3):
                hx = rng.integers(0, 2, (trial, n), dtype=np.uint8)
                z_kernel = nullspace(hx)
                hz = matmul(rng.integers(0, 2, (trial, len(z_kernel)), dtype=np.uint8),
                            z_kernel)
                # Redundant and zero rows must not alter the construction's
                # interpretation of stabilizer classes or signed relations.
                hx = np.vstack((hx, hx[:1], np.zeros((1, n), dtype=np.uint8)))
                hz = np.vstack((hz, hz[:1], np.zeros((1, n), dtype=np.uint8)))
                code = CSSCode(hx, hz)
                if code.k == 0:
                    continue
                q = min(code.k, 1 + trial)
                while True:
                    coefficients = rng.integers(0, 2, (q, code.k), dtype=np.uint8)
                    if rank(coefficients) == q:
                        break
                basis = code.standard_logicals()
                for pauli, generators, same_checks in (("X", basis.jx, code.hx),
                                                       ("Z", basis.jz, code.hz)):
                    canonical = matmul(coefficients, generators)
                    decoration = matmul(rng.integers(0, 2, (q, len(same_checks)), dtype=np.uint8),
                                        same_checks)
                    for targets in (canonical, canonical ^ decoration):
                        result = devised_deform(code, targets, pauli=pauli,
                                                 d_R=1 + trial % 2,
                                                 ldpc=trial != 1)
                        self.assert_result(code, result, pauli)
                        metadata = result.glue.metadata
                        if in_rowspace(targets, generators):
                            self.assertTrue(metadata["canonical_standard_form_representatives"])
                            budget = metadata["redundancy_number"] * (q + 1)
                            self.assertLessEqual(metadata["dressing_nonzeros"], budget)
                            self.assertLessEqual(result.glue.h_g.shape[1],
                                                 metadata["n_N"] + 2 * budget)
                            self.assertLessEqual(result.glue.h_g.shape[0],
                                                 metadata["r_N"] + 2 * budget)
                            self.assertTrue(metadata["optimized_standard_form_size_bound_claimed"])
                        checked += 1
        self.assertGreaterEqual(checked, 40)

    def test_rejects_out_of_scope_and_invalid_target_sets(self):
        code = four_qubit_code()
        for target in ("YIII", "ZXII", "IIII", "ZZZZ", "ZIII", [2, 0, 0, 0],
                       ["ZZII", "ZZII"], ["ZZII", "IIZZ"], []):
            with self.assertRaises(ValueError):
                devised_deform(code, target, d_R=1)
        with self.assertRaises(ValueError):
            devised_deform(code, pauli="Y", d_R=1)
        for d_r in (0, True, 1.5):
            with self.assertRaises(ValueError):
                devised_deform(code, d_R=d_r)
        with self.assertRaises(ValueError):
            devised_deform(code, max_entries=1)
        with self.assertRaises(ValueError):
            devised_deform(code, ldpc="yes")

    def test_export_contains_multiple_measurements_and_glue_arrays(self):
        result = devised_deform(four_qubit_code(), ["-ZZII", "ZIZI"], d_R=1)
        with tempfile.TemporaryDirectory() as directory:
            json_path, npz_path = Path(directory) / "result.json", Path(directory) / "result.npz"
            result.save(json_path)
            result.save(npz_path)
            saved = json.loads(json_path.read_text())
            self.assertEqual(saved["measurement_signs"], [1, 0])
            self.assertEqual(saved["summary"]["logical_measurement_rank"], 2)
            self.assertEqual(saved["summary"]["scheme"], "zhang_li_devised_sticking")
            with np.load(npz_path, allow_pickle=False) as saved_npz:
                for name in ("h_g", "s", "t", "measurement_witnesses", "protected_x", "protected_z"):
                    self.assertIn(name, saved_npz)
                np.testing.assert_array_equal(saved_npz["measurement_signs"], [1, 0])


if __name__ == "__main__":
    unittest.main()
