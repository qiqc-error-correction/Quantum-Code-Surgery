"""Non-CSS CKBB tests: phase, commutation and unintended measurements."""

import unittest

import numpy as np

from codesurgery.ckbb import _build_css_primitive, ckbb_deform
from codesurgery.gf2 import in_rowspace, matmul, nullspace, rank
from codesurgery.models import CSSCode
from codesurgery.measurement import UnintendedLogicalMeasurementError, validate_single_logical_measurement
from codesurgery.mixed import construct_mixed
from codesurgery.symplectic import commutation_matrix, pauli_product


def dense_pauli(x, z, sign=0):
    """Independent literal tensor matrices, used to check Y phases."""
    factors = {
        (0, 0): np.eye(2, dtype=complex),
        (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
        (0, 1): np.diag([1, -1]).astype(complex),
        (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex),
    }
    out = np.array([[(-1) ** sign]], dtype=complex)
    for a, b in zip(x, z):
        out = np.kron(out, factors[int(a), int(b)])
    return out


class MixedCKBBTests(unittest.TestCase):
    def assert_deformation(self, code, result):
        self.assertTrue(result.validate())
        self.assertFalse(np.any(commutation_matrix(result.check_x, result.check_z)))
        cx, cz, n = result.check_x, result.check_z, code.n
        # Inspect all products that have trivial ancillary action. Their
        # image modulo the *original* code's stabilizers must have rank one.
        cancellation = nullspace(np.hstack((cx[:, n:], cz[:, n:])).T)
        image = matmul(cancellation, np.hstack((cx[:, :n], cz[:, :n])))
        original = np.block([[code.hx, np.zeros_like(code.hx)],
                             [np.zeros_like(code.hz), code.hz]])
        self.assertEqual(rank(np.vstack((original, image))) - rank(original), 1)
        self.assertTrue(in_rowspace(image, np.vstack((original,
                            np.concatenate((result.measurement_x, result.measurement_z))))))
        # A binary zero does not determine the Pauli sign: test every
        # independent generator relation, including redundant input checks.
        dependencies = nullspace(np.hstack((cx, cz)).T)
        for relation in dependencies:
            sx, sz, phase = pauli_product(cx, cz, result.check_signs,
                                          np.flatnonzero(relation))
            self.assertFalse(np.any(sx | sz))
            self.assertEqual(phase, 0)

    def assert_dense_witness(self, result):
        product = np.eye(2 ** result.n, dtype=complex)
        for i in np.flatnonzero(result.measurement_witness):
            product = product @ dense_pauli(result.check_x[i], result.check_z[i],
                                             int(result.check_signs[i]))
        target = dense_pauli(np.pad(result.measurement_x, (0, result.n - result.original_n)),
                             np.pad(result.measurement_z, (0, result.n - result.original_n)),
                             result.measurement_sign)
        np.testing.assert_array_equal(product, target)

    def test_y_including_negative_phase_and_later_commuting_layers(self):
        code = CSSCode(np.zeros((0, 1), dtype=np.uint8), np.zeros((0, 1), dtype=np.uint8))
        for logical in ("Y", "-Y"):
            for layers in (1, 2, 3):
                result = ckbb_deform(code, logical, layers=layers)
                self.assert_deformation(code, result)
                self.assert_dense_witness(result)
                self.assertEqual(result.n, 2 * layers - 1)
                self.assertEqual(result.gauge_qubits, 0)

    def test_default_logical_y_uses_paired_standard_form_and_exact_phase(self):
        steane = [[1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 1]]
        codes = [
            CSSCode(np.zeros((0, 1), dtype=np.uint8), np.zeros((0, 1), dtype=np.uint8)),
            CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]]),
            CSSCode(steane, steane),
        ]
        for code in codes:
            basis = code.standard_logicals()
            for index in range(code.k):
                x, z = basis.jx[index], basis.jz[index]
                logical_y = 1j * dense_pauli(x, np.zeros(code.n, dtype=np.uint8)) @ dense_pauli(np.zeros(code.n, dtype=np.uint8), z)
                for layers in (1, 2):
                    result = ckbb_deform(code, pauli="Y", logical_index=index, layers=layers)
                    self.assert_deformation(code, result)
                    np.testing.assert_array_equal(result.measurement_x, x)
                    np.testing.assert_array_equal(result.measurement_z, z)
                    np.testing.assert_array_equal(dense_pauli(result.measurement_x,
                                               result.measurement_z, result.measurement_sign), logical_y)

    def test_default_y_phase_conversion_for_odd_overlap_three(self):
        # Alternative paired representatives can overlap at three sites.
        # i X(111) Z(111) = -YYY, so omitting the phase conversion is wrong.
        class ThreeOverlapCode(CSSCode):
            def logical(self, pauli="Z", index=0):
                if pauli not in ("X", "Z") or index != 0:
                    raise ValueError("This fixture defines only one paired logical")
                return np.ones(3, dtype=np.uint8)

        code = ThreeOverlapCode(np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8))
        # The raw YYY fusion would separately read all three Ys and must
        # fail closed. Parse the default target to test its phase without
        # asking that unsafe construction to return a deformed code.
        from codesurgery.ckbb import _parse_target
        x, z, sign = _parse_target(code, None, "Y", 0)
        self.assertEqual(sign, 1)
        expected = 1j * dense_pauli(x, np.zeros(3, dtype=np.uint8)) @ dense_pauli(np.zeros(3, dtype=np.uint8), z)
        np.testing.assert_array_equal(dense_pauli(x, z, sign), expected)
        with self.assertRaises(UnintendedLogicalMeasurementError):
            ckbb_deform(code, pauli="Y", logical_index=0, layers=1)

    def test_direct_mixed_builder_uses_raw_primitive_api_and_validates(self):
        code = CSSCode(np.zeros((0, 1), dtype=np.uint8), np.zeros((0, 1), dtype=np.uint8))
        calls = []

        def raw_builder(memory, vector, *, pauli, layers, allow_stabilizer, max_entries):
            calls.append(pauli)
            return _build_css_primitive(memory, vector, pauli=pauli, layers=layers,
                                        allow_stabilizer=allow_stabilizer, max_entries=max_entries)

        built = construct_mixed(code, [1], [1], 2, raw_builder, bridge=False)
        self.assertEqual(calls, ["X", "Z"])
        self.assertEqual(built["metadata"]["logical_measurement_rank"], 1)
        self.assertTrue(built["metadata"]["measurement_validation"]["validated_against_actual_deformed_checks"])
        # The very same check group still has rank-one readout when the
        # requested target is X instead of Y. Rank alone must not accept it.
        with self.assertRaises(UnintendedLogicalMeasurementError) as rejected:
            validate_single_logical_measurement(code, built["check_x"], built["check_z"], [1], [0])
        self.assertEqual(rejected.exception.logical_rank, 1)

    def test_even_overlap_mixed_product_is_not_two_measurements(self):
        code = CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]])
        for logical in ("YYII", "-YYII"):
            for layers in (1, 2):
                result = ckbb_deform(code, logical, layers=layers)
                self.assert_deformation(code, result)
                if layers == 1:
                    self.assert_dense_witness(result)
                self.assertEqual(result.metadata["overlap"], [0, 1])

    def test_overlapping_zx_requires_no_disjoint_connector(self):
        code = CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]])
        for layers in (1, 2):
            result = ckbb_deform(code, "XZYI", layers=layers, bridge=False)
            self.assert_deformation(code, result)
            self.assertEqual(result.metadata["overlap"], [2])
            self.assertEqual(result.metadata["connector_arms"], [])
            if layers == 1:
                self.assert_dense_witness(result)

    def test_multiple_pure_kernel_modes_can_fuse_into_only_target(self):
        # The X component XX is a stabilizer and its raw restriction has
        # two kernel modes. Fusion with ZZ still reads only logical YY;
        # requiring each pure kernel to have dimension one would reject
        # or alter this valid original construction unnecessarily.
        code = CSSCode([[1, 1]], np.zeros((0, 2), dtype=np.uint8))
        for layers in (1, 2):
            result = ckbb_deform(code, "YY", layers=layers, bridge=False)
            self.assert_deformation(code, result)
            self.assertEqual(len(nullspace(result.metadata["x_sticker"]["restriction_matrix"])), 2)
            if layers == 1:
                self.assert_dense_witness(result)

    def test_disjoint_xz_connector_arms(self):
        code = CSSCode(np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8))
        for layers in (1, 2, 3):
            result = ckbb_deform(code, "XZ", layers=layers)
            self.assert_deformation(code, result)
            self.assertEqual(result.n, 6 * layers - 2)
            self.assertEqual(len(result.metadata["connector_arms"]), 2)
            full = np.hstack((result.check_x, result.check_z))
            individual_x = np.zeros(2 * result.n, dtype=np.uint8)
            individual_z = individual_x.copy()
            individual_x[0] = 1
            individual_z[result.n + 1] = 1
            self.assertFalse(in_rowspace(individual_x, full))
            self.assertFalse(in_rowspace(individual_z, full))
            if layers == 1:
                self.assert_dense_witness(result)
        with self.assertRaisesRegex(ValueError, "connector arms"):
            ckbb_deform(code, "XZ", layers=1, bridge=False)

    def test_stabilizer_component_can_decorate_a_logical(self):
        code = CSSCode([[1, 0]], np.zeros((0, 2), dtype=np.uint8))
        for layers in (1, 3):
            result = ckbb_deform(code, "-XZ", layers=layers)
            self.assert_deformation(code, result)
            self.assertTrue(result.metadata["x_sticker"]["component_is_stabilizer"])

    def test_connector_attached_to_stabilized_submode_fails_closed(self):
        # An arm attached to the first support bit would connect a known
        # stabilizer instead of the logical parity. The implementation
        # preserves the original restrictions and rejects the extra
        # readout rather than adding an inferred path to change them.
        code = CSSCode([[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0]],
                       [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        target = {"x": [0, 0, 1, 0, 1, 0], "z": [0, 0, 0, 1, 0, 1]}
        for layers in (1, 2, 3):
            with self.assertRaises(UnintendedLogicalMeasurementError):
                ckbb_deform(code, target, layers=layers)

    def test_unconnected_overlap_components_read_extra_information(self):
        code = CSSCode(np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8))
        for layers in (1, 2):
            # The two distinct Y chains separately read Y_1 and Y_2.
            with self.assertRaises(UnintendedLogicalMeasurementError):
                ckbb_deform(code, "YY", layers=layers)

    def test_random_css_memory_and_mixed_physical_logicals(self):
        rng = np.random.default_rng(7641)
        examined, accepted, rejected = 0, 0, 0
        for _ in range(40):
            n = int(rng.integers(2, 8))
            hx = rng.integers(0, 2, size=(int(rng.integers(0, n)), n), dtype=np.uint8)
            zspace = nullspace(hx)
            hz = matmul(rng.integers(0, 2, size=(int(rng.integers(0, len(zspace) + 1)), len(zspace)), dtype=np.uint8), zspace)
            code = CSSCode(hx, hz)
            if code.k < 1:
                continue
            xs, zs = nullspace(hz), nullspace(hx)
            for _ in range(8):
                x = matmul(rng.integers(0, 2, size=len(xs), dtype=np.uint8), xs)
                z = matmul(rng.integers(0, 2, size=len(zs), dtype=np.uint8), zs)
                if not np.any(x) or not np.any(z) or (in_rowspace(x, hx) and in_rowspace(z, hz)):
                    continue
                target = dict(x=x, z=z, sign=int(rng.integers(0, 2)))
                try:
                    result = ckbb_deform(code, target, layers=int(rng.integers(1, 4)))
                except UnintendedLogicalMeasurementError:
                    rejected += 1
                else:
                    self.assert_deformation(code, result)
                    accepted += 1
                examined += 1
        self.assertGreater(examined, 100)
        self.assertGreater(accepted, 20)
        self.assertGreater(rejected, 0)

    def test_rejects_nonlogical_and_oversized_requests(self):
        code = CSSCode([[1, 0]], np.zeros((0, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "commute"):
            ckbb_deform(code, "YI", layers=2)
        bare = CSSCode(np.zeros((0, 2), dtype=np.uint8), np.zeros((0, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "max_entries"):
            ckbb_deform(bare, "XZ", layers=3, max_entries=100)


if __name__ == "__main__":
    unittest.main()
