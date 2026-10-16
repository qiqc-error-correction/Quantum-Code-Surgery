"""Inspect every memory-only product of the actual deformed checks."""

import numpy as np

from .gf2 import binary_matrix, binary_vector, matmul, nullspace, rank, row_basis


class UnintendedLogicalMeasurementError(ValueError):
    """The constructed checks read a logical space other than the target line."""

    def __init__(self, logical_rank, measured_signatures, target_signature):
        self.logical_rank = int(logical_rank)
        self.measured_signatures = np.array(measured_signatures, dtype=np.uint8, copy=True)
        self.target_signature = np.array(target_signature, dtype=np.uint8, copy=True)
        if logical_rank > 1:
            reason = f"CKBB construction would measure {logical_rank} independent logical observables; only the requested one is allowed"
        else:
            reason = "CKBB construction does not measure exactly the requested logical observable"
        super().__init__(f"{reason}. Measured logical signatures: {self.measured_signatures.tolist()}; target: {self.target_signature.tolist()}")


def validate_single_logical_measurement(code, check_x, check_z, target_x, target_z):
    """Require the final check group to read precisely one requested logical.

    First eliminate *all* ancillary X and Z support from check products.
    Pair their remaining memory support with a complete logical basis. The
    resulting logical signature space must equal the one-dimensional target
    line. This applies equally to CSS bridges, Y, and mixed ZX fusion.
    Operator signs and the signed target witness are checked separately.
    """
    cx = binary_matrix(check_x, name="check_x")
    cz = binary_matrix(check_z, name="check_z")
    if cx.shape != cz.shape or cx.shape[1] < code.n:
        raise ValueError("Deformed checks must have matching shapes and include all memory qubits")
    tx = binary_vector(target_x, length=code.n, name="target_x")
    tz = binary_vector(target_z, length=code.n, name="target_z")
    if np.any(matmul(tx, code.hz.T)) or np.any(matmul(tz, code.hx.T)):
        raise ValueError("Requested target must commute with the input checks")
    basis = code.standard_logicals()
    target_signature = np.concatenate((matmul(tx, basis.jz.T), matmul(tz, basis.jx.T)))
    if not np.any(target_signature):
        raise ValueError("Requested target is not a nontrivial logical observable")

    # A coefficient row u selects a check product. Requiring u A=0 removes
    # every auxiliary Pauli, including mixed X/Z support at the same site.
    auxiliary = np.hstack((cx[:, code.n:], cz[:, code.n:]))
    cancel_auxiliary = nullspace(auxiliary.T)
    memory_x = matmul(cancel_auxiliary, cx[:, :code.n])
    memory_z = matmul(cancel_auxiliary, cz[:, :code.n])
    if np.any(matmul(memory_x, code.hz.T)) or np.any(matmul(memory_z, code.hx.T)):
        raise ValueError("The deformed check group reads an operator outside the memory normalizer")
    signatures = np.hstack((matmul(memory_x, basis.jz.T), matmul(memory_z, basis.jx.T)))
    measured_basis = row_basis(signatures)
    logical_rank = len(measured_basis)
    if logical_rank != 1 or rank(np.vstack((measured_basis, target_signature))) != 1:
        raise UnintendedLogicalMeasurementError(logical_rank, measured_basis, target_signature)

    return {
        "logical_measurement_rank": logical_rank,
        "measurement_validation": {
            "auxiliary_cancellation_kernel": cancel_auxiliary,
            "memory_image_x": memory_x,
            "memory_image_z": memory_z,
            "logical_signature": signatures,
            "target_signature": target_signature,
            "measured_signature_basis": measured_basis,
            "logical_rank": logical_rank,
            "coordinate_order": "logical X coefficients, then logical Z coefficients",
            "validated_against_actual_deformed_checks": True,
        },
    }
