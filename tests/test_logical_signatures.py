"""Support normalizers and their logical coordinates in the paired basis."""

import unittest

import numpy as np

from codesurgery import CSSCode, ckbb_deform, UnintendedLogicalMeasurementError
from codesurgery.examples import direct_sum, four_qubit_code
from codesurgery.gf2 import in_rowspace, matmul, nullspace, rank


class LogicalSignatureTests(unittest.TestCase):
    def assert_signature_metadata(self, code, result, pauli, expected_target=None):
        md = result.metadata
        basis = code.standard_logicals()
        conjugate = basis.jz if pauli == "X" else basis.jx
        same, opposite = (code.hx, code.hz) if pauli == "X" else (code.hz, code.hx)
        target = result.measurement_x if pauli == "X" else result.measurement_z
        selector = md["support_selector"]
        np.testing.assert_array_equal(md["conjugate_logical_generators"], conjugate)
        self.assertEqual(selector.shape, (len(md["support"]), code.n))
        np.testing.assert_array_equal(selector.sum(axis=1), np.ones(len(selector)))
        np.testing.assert_array_equal(np.argmax(selector, axis=1), md["support"])
        np.testing.assert_array_equal(matmul(np.ones(len(selector), dtype=np.uint8), selector), target)
        expected_target = matmul(target, conjugate.T) if expected_target is None else expected_target
        np.testing.assert_array_equal(md["target_logical_signature"], expected_target)
        self.assertTrue(np.any(expected_target))

        for suffix, rank_key in (
            ("_before_completion", "logical_image_rank_before_bridges"),
            ("", "logical_image_rank_after_bridges"),
        ):
            kernel = md["kernel_basis" + suffix]
            normalizers = md["support_normalizers" + suffix]
            signatures = md["logical_signature" + suffix]
            np.testing.assert_array_equal(normalizers, matmul(kernel, selector))
            np.testing.assert_array_equal(signatures, matmul(normalizers, conjugate.T))
            self.assertFalse(np.any(matmul(normalizers, opposite.T)))
            # Stabilizer quotient rank is an independent check of the paired
            # logical-coordinate criterion, using the original check rows.
            quotient_rank = rank(np.vstack((same, normalizers))) - rank(same)
            self.assertEqual(rank(signatures), quotient_rank)
            self.assertEqual(md[rank_key], quotient_rank)
        np.testing.assert_array_equal(md["normalizer_syndrome"], matmul(md["support_normalizers"], opposite.T))
        self.assertFalse(np.any(md["normalizer_syndrome"]))
        # A nontrivial one-operator measurement has exactly the target line
        # in logical-coordinate space, even for a multi-qubit product.
        self.assertEqual(rank(md["logical_signature"]), 1)
        self.assertTrue(in_rowspace(md["logical_signature"], expected_target[None, :]))
        self.assertTrue(in_rowspace(expected_target, md["logical_signature"]))
        self.assertTrue(in_rowspace(target, md["support_normalizers"]))

    def test_first_default_generator_has_e0_signature(self):
        code = four_qubit_code()
        for pauli in ("X", "Z"):
            target_signature = np.zeros(code.k, dtype=np.uint8)
            target_signature[0] = 1
            result = ckbb_deform(code, pauli=pauli, layers=2)
            self.assert_signature_metadata(code, result, pauli, target_signature)
            for key in ("logical_signature_before_completion", "logical_signature"):
                self.assertFalse(np.any(result.metadata[key][:, 1:]))
                self.assertTrue(np.all(np.isin(result.metadata[key][:, 0], (0, 1))))

    def test_product_signature_can_have_several_nonzero_columns(self):
        code = four_qubit_code()
        coefficients = np.ones(code.k, dtype=np.uint8)
        for pauli in ("X", "Z"):
            result = ckbb_deform(code, code.logical_product(coefficients, pauli), pauli=pauli, layers=1)
            self.assert_signature_metadata(code, result, pauli, coefficients)
            self.assertEqual(int(result.metadata["target_logical_signature"].sum()), 2)

    def test_explicit_disjoint_factors_are_bridged_to_one_line(self):
        code = direct_sum(four_qubit_code(), four_qubit_code())
        target = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
        for pauli in ("X", "Z"):
            with self.assertRaises(UnintendedLogicalMeasurementError):
                ckbb_deform(code, target, pauli=pauli, layers=2)
            factors = [target * np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
                       target * np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)]
            result = ckbb_deform(code, logical_factors=factors, pauli=pauli, layers=2)
            self.assert_signature_metadata(code, result, pauli)
            self.assertEqual(rank(result.metadata["logical_signature_before_completion"]), 2)
            self.assertEqual(rank(result.metadata["logical_signature"]), 1)
            self.assertTrue(result.metadata["bridge_pairs"])

    def test_kernel_dimension_two_but_only_one_logical_class(self):
        # 110 is a stabilizer and 001 is the only logical class. Therefore
        # the support of 111 contains two independent normalizers, whose
        # logical signatures are respectively 0 and e0. The raw construction
        # therefore does not read an additional independent logical.
        code = CSSCode([[1, 1, 0]], [[1, 1, 0]], distance=1)
        for pauli in ("X", "Z"):
            result = ckbb_deform(code, [1, 1, 1], pauli=pauli, layers=2, bridge=False)
            self.assert_signature_metadata(code, result, pauli, np.ones(1, dtype=np.uint8))
            md = result.metadata
            self.assertEqual(len(md["kernel_basis_before_completion"]), 2)
            self.assertEqual(len(md["kernel_basis"]), 2)
            self.assertEqual(rank(md["logical_signature"]), 1)
            self.assertEqual(sorted(int(row[0]) for row in md["logical_signature"]), [0, 1])
            self.assertEqual(md["bridge_pairs"], [])

    def test_random_stabilizer_decorations_keep_known_logical_coordinates(self):
        rng = np.random.default_rng(89037)
        tested = 0
        for _ in range(24):
            n = int(rng.integers(1, 8))
            hx = rng.integers(0, 2, size=(int(rng.integers(0, n)), n), dtype=np.uint8)
            kernel_x = nullspace(hx)
            hz = matmul(rng.integers(0, 2, size=(int(rng.integers(0, len(kernel_x) + 1)), len(kernel_x)), dtype=np.uint8), kernel_x)
            if len(hx):
                hx = np.vstack((hx, hx[0], np.zeros(n, dtype=np.uint8)))
            if len(hz):
                hz = np.vstack((hz, hz[0], np.zeros(n, dtype=np.uint8)))
            code = CSSCode(hx, hz)
            if code.k == 0:
                continue
            basis = code.standard_logicals()
            for pauli, logicals, stabilizers in (("X", basis.jx, hx), ("Z", basis.jz, hz)):
                for _ in range(3):
                    coefficients = rng.integers(0, 2, code.k, dtype=np.uint8)
                    if not np.any(coefficients):
                        coefficients[int(rng.integers(0, code.k))] = 1
                    target = matmul(coefficients, logicals)
                    target ^= matmul(rng.integers(0, 2, len(stabilizers), dtype=np.uint8), stabilizers)
                    support = np.flatnonzero(target)
                    local_kernel = nullspace((hz if pauli == "X" else hx)[:, support])
                    normalizers = np.zeros((len(local_kernel), code.n), dtype=np.uint8)
                    normalizers[:, support] = local_kernel
                    quotient_rank = rank(np.vstack((stabilizers, normalizers))) - rank(stabilizers)
                    if quotient_rank > 1:
                        with self.assertRaises(UnintendedLogicalMeasurementError):
                            ckbb_deform(code, target, pauli=pauli, layers=1)
                        tested += 1
                        continue
                    result = ckbb_deform(code, target, pauli=pauli, layers=1)
                    # Expected coefficients are fixed before building the
                    # deformation, and survive arbitrary stabilizer factors.
                    self.assert_signature_metadata(code, result, pauli, coefficients)
                    tested += 1
        self.assertGreater(tested, 50)


if __name__ == "__main__":
    unittest.main()
