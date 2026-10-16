"""Logical-class and phase safeguards for standalone branching inputs."""

import unittest

import numpy as np

from codesurgery import CSSCode
from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.parallel_branching.targets import prepare_targets, qubitwise_commuting
from codesurgery.symplectic import pauli_product, pauli_string


class BranchingTargetTests(unittest.TestCase):
    def assert_witnesses(self, result):
        for i, w in enumerate(result["canonicalization_witnesses"]):
            x = np.vstack((result["original_target_x"][i], result["source_check_x"]))
            z = np.vstack((result["original_target_z"][i], result["source_check_z"]))
            signs = np.concatenate(([result["original_target_signs"][i]], result["source_check_signs"]))
            px, pz, phase = pauli_product(x, z, signs, [0, *(np.flatnonzero(w) + 1).tolist()])
            np.testing.assert_array_equal(px, result["target_x"][i])
            np.testing.assert_array_equal(pz, result["target_z"][i])
            self.assertEqual(phase, 2 * int(result["target_signs"][i]))

    def test_default_standard_basis_preserves_sparse_redundant_checks(self):
        h = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        checks = np.vstack((h, h, np.zeros_like(h)))
        code = CSSCode(checks, checks)
        result = prepare_targets(code)
        basis = code.standard_logicals()
        np.testing.assert_array_equal(result["target_z"], basis.jz)
        np.testing.assert_array_equal(result["source_check_x"][:3], checks)
        np.testing.assert_array_equal(result["source_check_z"][3:], checks)
        np.testing.assert_array_equal(result["logical_signatures"], np.hstack((np.zeros((2, 2)), np.eye(2))))
        self.assertEqual(result["basis_source"], "Zhang--Li standard form")
        self.assertEqual(result["target_logical_rank"], 2)
        self.assert_witnesses(result)
        result["source_check_x"][0, 0] = 0
        self.assertEqual(code.hx[0, 0], 1)

    def test_signed_physical_qwc_inputs_are_kept_exactly(self):
        result = prepare_targets(four_qubit_code(), ["-YZZY", "+YZZY"])
        np.testing.assert_array_equal(result["target_x"], result["original_target_x"])
        np.testing.assert_array_equal(result["target_z"], result["original_target_z"])
        np.testing.assert_array_equal(result["target_signs"], [1, 0])
        self.assertEqual(result["representation_action"], "preserved")
        self.assertEqual(result["target_logical_rank"], 1)
        self.assertFalse(np.any(result["canonicalization_witnesses"]))

    def test_auto_same_class_rewrite_includes_minus_sign(self):
        # YZZY is equivalent to -IXXI after multiplying XXXX and ZZZZ.
        # Its overlap with ZIIZ is globally commuting but is not QWC.
        result = prepare_targets(four_qubit_code(), ["YZZY", "ZIIZ"])
        output = [pauli_string(x, z, sign) for x, z, sign in
                  zip(result["target_x"], result["target_z"], result["target_signs"])]
        self.assertEqual(output, ["-IXXI", "ZIIZ"])
        np.testing.assert_array_equal(result["canonicalization_witnesses"][0], [1, 1])
        np.testing.assert_array_equal(result["logical_signatures"], [[1, 0, 0, 0], [0, 0, 0, 1]])
        self.assertFalse(result["qwc_before"])
        self.assertTrue(result["qwc_after"])
        self.assert_witnesses(result)

    def test_canonical_rewrites_even_a_single_qwc_input(self):
        result = prepare_targets(four_qubit_code(), "-YZZY", representatives="canonical")
        self.assertEqual(pauli_string(result["target_x"][0], result["target_z"][0],
                                     result["target_signs"][0]), "IXXI")
        self.assert_witnesses(result)

    def test_preserve_rejects_local_conflict_even_when_global_commutation_holds(self):
        with self.assertRaisesRegex(ValueError, "qubit-wise"):
            prepare_targets(four_qubit_code(), ["YZZY", "ZIIZ"], representatives="preserve")

    def test_arbitrary_commuting_products_are_not_claimed_to_be_qwc(self):
        # No checks: these same classes have no alternative representatives.
        code = CSSCode(np.zeros((0, 2)), np.zeros((0, 2)))
        for mode in ("auto", "canonical", "preserve"):
            with self.assertRaisesRegex(ValueError, "qubit-wise"):
                prepare_targets(code, logical_paulis=["XX", "ZZ"], representatives=mode)

    def test_logical_y_phase_uses_supplied_basis(self):
        code = steane_code()
        basis = {"jx": np.ones((1, 7), dtype=np.uint8), "jz": np.ones((1, 7), dtype=np.uint8)}
        result = prepare_targets(code, logical_paulis=["Y", "-Y"], logical_basis=basis)
        np.testing.assert_array_equal(result["target_x"], np.ones((2, 7)))
        np.testing.assert_array_equal(result["target_z"], np.ones((2, 7)))
        np.testing.assert_array_equal(result["target_signs"], [1, 0])
        self.assertEqual(result["basis_source"], "user supplied paired CSS basis")
        self.assertEqual(result["target_logical_rank"], 1)
        self.assert_witnesses(result)

    def test_mixed_single_logical_terms_use_default_qwc_basis(self):
        result = prepare_targets(four_qubit_code(), logical_paulis=["YI", "IX"])
        self.assertTrue(qubitwise_commuting(result["target_x"], result["target_z"]))
        np.testing.assert_array_equal(result["logical_signatures"], [[1, 0, 1, 0], [0, 1, 0, 0]])

    def test_no_basis_change_when_user_basis_is_not_qwc(self):
        code = four_qubit_code()
        basis = code.standard_logicals()
        jx = basis.jx.copy()
        jx[0] ^= code.hx[0]
        with self.assertRaisesRegex(ValueError, "selected basis"):
            prepare_targets(code, logical_paulis=["XI", "IZ"],
                            logical_basis={"jx": jx, "jz": basis.jz})

    def test_reducible_target_and_logical_dependencies_are_allowed(self):
        # With no checks, ZZI contains two independent logical classes.
        code = CSSCode(np.zeros((0, 3)), np.zeros((0, 3)))
        result = prepare_targets(code, ["ZZI", "ZII", "IZI", "ZZI"])
        self.assertEqual(len(result["target_x"]), 4)
        self.assertEqual(result["target_logical_rank"], 2)
        self.assert_witnesses(result)

    def test_index_mode_supports_y_and_repeated_indices(self):
        result = prepare_targets(four_qubit_code(), pauli="Y", logical_indices=[1, 0, 1])
        np.testing.assert_array_equal(result["logical_signatures"],
                                      [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        self.assertEqual(result["target_logical_rank"], 2)

    def test_invalid_inputs_are_rejected(self):
        code = four_qubit_code()
        invalid = [
            {"targets": []}, {"logical_paulis": []}, {"logical_indices": []},
            {"targets": ["IIII"]}, {"targets": ["-XXXX"]},
            {"targets": ["XIII"]},  # Not a normalizer.
            {"logical_paulis": ["XI", "ZI"]},  # Anticommuting.
            {"logical_paulis": ["II"]}, {"logical_paulis": ["X"]},
            {"targets": ["ZIZI"], "logical_paulis": ["ZI"]},
            {"targets": ["ZIZI"], "logical_indices": [0]},
            {"logical_paulis": ["ZI"], "pauli": "X"},
            {"pauli": "I"}, {"representatives": "search"},
            {"logical_indices": [True]}, {"logical_indices": [-1]},
            {"logical_indices": [2]}, {"logical_indices": [0.5]},
            {"targets": {"x": "ZIZI"}}, {"targets": [None]},
            {"logical_basis": {"jx": [[1, 0, 0, 0]], "jz": [[1, 0, 0, 0]]}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_targets(code, **kwargs)
        with self.assertRaises(TypeError):
            prepare_targets({"hx": code.hx, "hz": code.hz})
        with self.assertRaisesRegex(ValueError, "At least one"):
            prepare_targets(CSSCode(np.eye(2, dtype=np.uint8), np.zeros((0, 2))))


if __name__ == "__main__":
    unittest.main()
