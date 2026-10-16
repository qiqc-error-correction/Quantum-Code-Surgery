"""Independent operation/gauge invariants for pure CKBB surgery."""

import itertools
import unittest

import numpy as np

from codesurgery import CSSCode, ckbb_deform, UnintendedLogicalMeasurementError
from codesurgery.examples import four_qubit_code
from codesurgery.gf2 import in_rowspace, matmul, nullspace, rank
from codesurgery.symplectic import pauli_product


class PureCKBBPropertyTests(unittest.TestCase):
    def assert_operation_and_gauges(self, code, result):
        self.assertTrue(result.validate())
        cx, cz = result.check_x, result.check_z
        full = np.hstack((cx, cz))
        original = np.block([
            [code.hx, np.zeros_like(code.hx)],
            [np.zeros_like(code.hz), code.hz],
        ])
        # Examine the complete subgroup with identity action on ancillas,
        # rather than relying on either the witness or the total code rank.
        cancel_aux = nullspace(np.hstack((cx[:, code.n :], cz[:, code.n :])).T)
        memory_image = matmul(cancel_aux, np.hstack((cx[:, : code.n], cz[:, : code.n])))
        self.assertEqual(rank(np.vstack((original, memory_image))) - rank(original), 1)
        allowed = np.vstack((original, np.concatenate((result.measurement_x, result.measurement_z))))
        self.assertTrue(in_rowspace(memory_image, allowed))
        # Check all independent signed dependencies: a commuting relation
        # yielding -I would make the alleged deformed code space empty.
        for dependency in nullspace(full.T):
            sx, sz, phase = pauli_product(cx, cz, result.check_signs, np.flatnonzero(dependency))
            self.assertFalse(np.any(sx | sz))
            self.assertEqual(phase, 0)
        gx, gz, g = result.gauge_x, result.gauge_z, result.gauge_qubits
        self.assertEqual(gx.shape, (g, result.n))
        self.assertEqual(gz.shape, (g, result.n))
        np.testing.assert_array_equal(matmul(gx, gz.T), np.eye(g))
        self.assertFalse(np.any(matmul(gx, result.hz.T)))
        self.assertFalse(np.any(matmul(gz, result.hx.T)))
        gauges = np.block([[gx, np.zeros_like(gx)], [np.zeros_like(gz), gz]])
        self.assertEqual(rank(np.vstack((full, gauges))), rank(full) + 2 * g)

    def test_arbitrary_pure_normalizer_representatives(self):
        rng = np.random.default_rng(15469)
        tested = 0
        for n in range(1, 8):
            for _ in range(5):
                hx = rng.integers(0, 2, size=(int(rng.integers(0, n)), n), dtype=np.uint8)
                kernel_x = nullspace(hx)
                hz = matmul(rng.integers(0, 2, size=(int(rng.integers(0, len(kernel_x) + 1)), len(kernel_x)), dtype=np.uint8), kernel_x)
                # Exercise redundant and zero rows without canonicalizing the
                # user's original check presentation.
                if len(hx):
                    hx = np.vstack((hx, hx[0], np.zeros(n, dtype=np.uint8)))
                if len(hz):
                    hz = np.vstack((hz, hz[0], np.zeros(n, dtype=np.uint8)))
                code = CSSCode(hx, hz)
                if code.k == 0:
                    continue
                for pauli, same, opposite in (("X", hx, hz), ("Z", hz, hx)):
                    normalizer = nullspace(opposite)
                    # A basis generator plus arbitrary stabilizer decoration
                    # ensures one example; random normalizer words also cover
                    # products of logical generators with varying support.
                    representative = code.logical(pauli)
                    if len(same):
                        representative ^= matmul(rng.integers(0, 2, len(same), dtype=np.uint8), same)
                    candidates = [representative]
                    candidates.extend(matmul(rng.integers(0, 2, len(normalizer), dtype=np.uint8), normalizer) for _ in range(5))
                    for vector in candidates:
                        if in_rowspace(vector, same):
                            continue
                        sign = int(rng.integers(0, 2))
                        target = {pauli.lower(): vector, "sign": sign}
                        mapped = np.zeros((len(nullspace(opposite[:, np.flatnonzero(vector)])), code.n), dtype=np.uint8)
                        mapped[:, np.flatnonzero(vector)] = nullspace(opposite[:, np.flatnonzero(vector)])
                        raw_logical_rank = rank(np.vstack((same, mapped))) - rank(same)
                        layers = int(rng.integers(1, 4))
                        if raw_logical_rank > 1:
                            with self.assertRaises(UnintendedLogicalMeasurementError):
                                ckbb_deform(code, target, layers=layers)
                            tested += 1
                            continue
                        result = ckbb_deform(code, target, layers=layers)
                        self.assert_operation_and_gauges(code, result)
                        tested += 1
        self.assertGreater(tested, 100)

    def test_default_with_both_or_one_empty_check_set(self):
        for hx, hz in (
            (np.zeros((0, 3)), np.zeros((0, 3))),
            ([[1, 1, 0]], []),
            ([], [[1, 1, 0]]),
        ):
            code = CSSCode(hx, hz)
            for pauli in ("X", "Z"):
                result = ckbb_deform(code, pauli=pauli, layers=2)
                np.testing.assert_array_equal(
                    result.measurement_x if pauli == "X" else result.measurement_z,
                    code.logical(pauli),
                )
                self.assert_operation_and_gauges(code, result)

    def test_four_qubit_protected_distance_preserved(self):
        code = four_qubit_code()
        for pauli in ("X", "Z"):
            for layers in (2, 3):
                result = ckbb_deform(code, pauli=pauli, layers=layers)
                # Dressed subsystem logicals are centralizers of the check
                # subgroup modulo the full check+gauge subgroup.
                gauge_group = np.vstack((
                    np.hstack((result.check_x, result.check_z)),
                    np.hstack((result.gauge_x, np.zeros_like(result.gauge_x))),
                    np.hstack((np.zeros_like(result.gauge_z), result.gauge_z)),
                ))
                min_weight = None
                for weight in (1, 2):
                    for support in itertools.combinations(range(result.n), weight):
                        for paulis in itertools.product((1, 2, 3), repeat=weight):
                            x, z = np.zeros(result.n, dtype=np.uint8), np.zeros(result.n, dtype=np.uint8)
                            for qubit, kind in zip(support, paulis):
                                x[qubit], z[qubit] = kind & 1, (kind >> 1) & 1
                            syndrome = (matmul(result.check_x, z) ^ matmul(result.check_z, x))
                            if not np.any(syndrome) and not in_rowspace(np.concatenate((x, z)), gauge_group):
                                min_weight = weight
                                break
                        if min_weight is not None:
                            break
                    if min_weight is not None:
                        break
                self.assertEqual(min_weight, 2)


if __name__ == "__main__":
    unittest.main()
