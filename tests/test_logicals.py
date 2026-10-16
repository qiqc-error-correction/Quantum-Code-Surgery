"""Algebraic tests for default logical operators, independent of CKBB."""

import unittest

import numpy as np

from codesurgery.gf2 import (
    binary_matrix,
    binary_vector,
    in_rowspace,
    inverse,
    matmul,
    nullspace,
    rank,
    row_basis,
    rref,
    solve,
)
from codesurgery.logicals import standard_form_logicals


class GF2Tests(unittest.TestCase):
    def test_rref_kernel_and_redundant_rows(self):
        a = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 0]])
        reduced, pivots = rref(a)
        self.assertEqual(pivots, (0, 1))
        np.testing.assert_array_equal(reduced[:2], [[1, 0, 1, 1], [0, 1, 1, 0]])
        self.assertEqual(rank(a), 2)
        np.testing.assert_array_equal(nullspace(a), [[1, 1, 1, 0], [1, 0, 0, 1]])
        self.assertFalse(np.any(matmul(a, nullspace(a).T)))
        self.assertTrue(in_rowspace([1, 0, 1, 1], a))
        self.assertFalse(in_rowspace([1, 0, 0, 0], a))
        self.assertEqual(row_basis(a).shape, (2, 4))

    def test_linear_system_and_inverse(self):
        a = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        x = solve(a, [1, 0])
        np.testing.assert_array_equal(matmul(a, x), [1, 0])
        np.testing.assert_array_equal(x, [1, 0, 0])  # zero free variable
        square = np.array([[1, 1], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(matmul(square, inverse(square)), np.eye(2))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            solve([[1], [1]], [0, 1])
        with self.assertRaisesRegex(ValueError, "singular"):
            inverse([[1, 1], [1, 1]])

    def test_empty_shapes(self):
        a = binary_matrix([], n_cols=5)
        self.assertEqual(a.shape, (0, 5))
        self.assertEqual(rank(a), 0)
        np.testing.assert_array_equal(nullspace(a), np.eye(5))
        self.assertEqual(row_basis(a).shape, (0, 5))
        self.assertTrue(in_rowspace(np.zeros(5, dtype=np.uint8), a))
        self.assertFalse(in_rowspace(np.ones(5, dtype=np.uint8), a))
        self.assertEqual(inverse(np.zeros((0, 0))).shape, (0, 0))
        self.assertEqual(solve(np.zeros((0, 5)), np.zeros(0)).shape, (5,))
        self.assertEqual(nullspace(np.zeros((3, 0))).shape, (0, 0))

    def test_validation(self):
        for bad in ([[0, 2]], [[0, -1]], [[0, np.nan]], [["0", "1"]], [[0.5, 1]]):
            with self.assertRaises(ValueError):
                binary_matrix(bad)
        with self.assertRaises(ValueError):
            binary_matrix([0, 1])
        with self.assertRaises(ValueError):
            binary_matrix([[0, 1]], n_cols=3)
        with self.assertRaises(ValueError):
            binary_matrix([], n_cols=True)
        with self.assertRaises(ValueError):
            binary_vector([[0, 1]])
        with self.assertRaises(ValueError):
            binary_vector([0, 1], length=3)


class StandardFormTests(unittest.TestCase):
    def assert_logical_basis(self, hx, hz, basis):
        hx = binary_matrix(hx, n_cols=basis.n)
        hz = binary_matrix(hz, n_cols=basis.n)
        k, n = basis.k, basis.n
        r_x, r_z = rank(hx), rank(hz)
        self.assertEqual(k, n - r_x - r_z)
        self.assertFalse(np.any(matmul(hx, basis.jz.T)))
        self.assertFalse(np.any(matmul(hz, basis.jx.T)))
        np.testing.assert_array_equal(matmul(basis.jx, basis.jz.T), np.eye(k))
        self.assertEqual(rank(np.vstack((hx, basis.jx))), r_x + k)
        self.assertEqual(rank(np.vstack((hz, basis.jz))), r_z + k)
        permutation = basis.column_permutation
        np.testing.assert_array_equal(np.sort(permutation), np.arange(n))
        np.testing.assert_array_equal(basis.jx[:, permutation], basis.jx_standard)
        np.testing.assert_array_equal(basis.jz[:, permutation], basis.jz_standard)
        np.testing.assert_array_equal(basis.jx_standard[:, :k], np.eye(k))
        np.testing.assert_array_equal(basis.jz_standard[:, :k], np.eye(k))
        self.assertFalse(np.any(basis.jx_standard[:, k : k + r_x]))
        self.assertFalse(np.any(basis.jz_standard[:, k + r_x :]))
        np.testing.assert_array_equal(basis.hx_standard[:, k : k + r_x], np.eye(r_x))
        np.testing.assert_array_equal(basis.hz_standard[:, k + r_x :], np.eye(r_z))
        self.assertTrue(in_rowspace(basis.hx_standard[:, basis.inverse_column_permutation], hx))
        self.assertTrue(in_rowspace(basis.hz_standard[:, basis.inverse_column_permutation], hz))

    def test_four_qubit_code_exact_standard_form(self):
        h = [[1, 1, 1, 1]]
        basis = standard_form_logicals(h, h)
        self.assert_logical_basis(h, h, basis)
        np.testing.assert_array_equal(basis.column_permutation, [2, 3, 0, 1])
        np.testing.assert_array_equal(basis.jx, [[0, 1, 1, 0], [0, 1, 0, 1]])
        np.testing.assert_array_equal(basis.jz, [[1, 0, 1, 0], [1, 0, 0, 1]])

    def test_steane_code(self):
        h = [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1]]
        basis = standard_form_logicals(h, h)
        self.assert_logical_basis(h, h, basis)
        self.assertEqual(basis.k, 1)

    def test_redundancy_does_not_change_default_basis(self):
        h = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        redundant = np.vstack((h, h, np.zeros((1, 4), dtype=np.uint8)))
        a = standard_form_logicals(h, h)
        b = standard_form_logicals(redundant, redundant)
        np.testing.assert_array_equal(a.jx, b.jx)
        np.testing.assert_array_equal(a.jz, b.jz)
        np.testing.assert_array_equal(a.column_permutation, b.column_permutation)

    def test_empty_checks_and_no_logicals(self):
        for hx, hz in (
            (np.zeros((0, 4)), np.zeros((0, 4))),
            (np.eye(4), np.zeros((0, 4))),
            (np.zeros((0, 4)), np.eye(4)),
            (np.zeros((0, 0)), np.zeros((0, 0))),
            (np.zeros((0, 5)), [[0, 1, 0, 0, 1]]),
        ):
            basis = standard_form_logicals(hx, hz)
            self.assert_logical_basis(hx, hz, basis)
        self.assertEqual(standard_form_logicals([], np.eye(3)).k, 0)
        self.assertEqual(standard_form_logicals(np.eye(3), []).k, 0)

    def test_random_commuting_css_codes(self):
        # Generate H_Z from ker(H_X), independently testing paired logicals
        # for many ranks, overlapping supports, and redundant check sets.
        rng = np.random.default_rng(3921)
        for n in range(1, 24):
            for _ in range(8):
                hx = rng.integers(0, 2, size=(int(rng.integers(0, n + 3)), n), dtype=np.uint8)
                kernel_x = nullspace(hx)
                mixing = rng.integers(0, 2, size=(int(rng.integers(0, n + 3)), len(kernel_x)), dtype=np.uint8)
                hz = matmul(mixing, kernel_x)
                basis = standard_form_logicals(hx, hz)
                self.assert_logical_basis(hx, hz, basis)

    def test_invalid_css_inputs(self):
        with self.assertRaisesRegex(ValueError, "commute"):
            standard_form_logicals([[1, 0]], [[1, 1]])
        with self.assertRaisesRegex(ValueError, "same number"):
            standard_form_logicals([[1, 0]], [[1, 1, 0]])
        with self.assertRaises(ValueError):
            standard_form_logicals([[1, 2]], [[0, 0]])


if __name__ == "__main__":
    unittest.main()
