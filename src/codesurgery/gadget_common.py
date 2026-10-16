"""Shared signed-Pauli, logical-basis and readout checks for graph gadgets."""
import numpy as np
from .gf2 import binary_matrix, binary_vector, nullspace, rank, row_basis, solve, in_rowspace
from .models import CSSCode
from .symplectic import parse_pauli, pauli_product
from .universal_adapters.graphs import budget_check


class GaugeMeasurementRequired(ValueError):
    """The restricted explicit-gadget route cannot implement this input."""
    def __init__(self, reason, *, details=None):
        self.reason = reason
        self.details = details or {}
        self.logical_rank = self.details.get("logical_rank")
        self.measured_signatures = self.details.get("measured_signatures")
        self.target_signature = self.details.get("target_signature")
        super().__init__(f"{reason}. Use gauging measurement (gauge measurement) with a suitable connected auxiliary graph.")


def logical_basis_for(code, supplied=None):
    if supplied is None:
        basis = code.standard_logicals()
        return dict(jx=basis.jx, jz=basis.jz, origin="paired CSS standard form")
    if not isinstance(supplied, dict) or "jx" not in supplied or "jz" not in supplied:
        raise ValueError("logical_basis must be a dict with jx and jz (paired pure CSS logical rows)")
    jx = binary_matrix(supplied["jx"], n_cols=code.n, name="logical_basis.jx")
    jz = binary_matrix(supplied["jz"], n_cols=code.n, name="logical_basis.jz")
    if len(jx) != code.k or len(jz) != code.k:
        raise ValueError("logical_basis must contain exactly k paired logical rows")
    if np.any((jx @ code.hz.T) % 2) or np.any((jz @ code.hx.T) % 2):
        raise ValueError("logical_basis rows must commute with the code stabilizers")
    if not np.array_equal((jx @ jz.T) % 2, np.eye(code.k, dtype=np.uint8)):
        raise ValueError("logical_basis must satisfy JX JZ^T = I")
    return dict(jx=jx, jz=jz, origin="user supplied paired CSS basis")


def prepare(code, logical=None, *, pauli="Z", logical_index=0,
            logical_basis=None, max_entries=50_000_000):
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    r, n = len(code.hx) + len(code.hz), code.n
    budget_check(max_entries, 2 * r * n, r * r)
    budget_check(max_entries, 2 * n * code.k)
    source = CSSCode(code.hx, code.hz, code.distance, code.name)
    basis = logical_basis_for(source, logical_basis)
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Y", "Z"):
        raise ValueError("pauli must be X, Y, or Z")
    pauli = pauli.upper()
    if logical is None:
        if isinstance(logical_index, (bool, np.bool_)) or not isinstance(logical_index, (int, np.integer)) or not 0 <= logical_index < source.k:
            raise ValueError("logical_index must be an integer in [0,k)")
        a, b = np.zeros(source.k, dtype=np.uint8), np.zeros(source.k, dtype=np.uint8)
        if pauli in ("X", "Y"):
            a[logical_index] = 1
        if pauli in ("Z", "Y"):
            b[logical_index] = 1
        logical = dict(logical_x=a, logical_z=b)
    if isinstance(logical, str):
        x, z, sign = parse_pauli(logical, n)
    elif isinstance(logical, dict):
        sign = logical.get("sign", 0)
        if isinstance(sign, (bool, np.bool_)) or not isinstance(sign, (int, np.integer)) or sign not in (0, 1):
            raise ValueError("logical.sign must be 0 or 1")
        if "logical_x" in logical or "logical_z" in logical:
            if "x" in logical or "z" in logical:
                raise ValueError("Do not mix logical-basis coefficients with physical x/z vectors")
            a = binary_vector(logical.get("logical_x", np.zeros(source.k)), length=source.k)
            b = binary_vector(logical.get("logical_z", np.zeros(source.k)), length=source.k)
            x, z = (a @ basis["jx"]) % 2, (b @ basis["jz"]) % 2
            sign ^= ((int(np.sum(a * b)) - int(np.sum(x * z))) // 2) % 2
        else:
            if "x" not in logical and "z" not in logical:
                raise ValueError("Logical dict requires x/z or logical_x/logical_z")
            x = binary_vector(logical.get("x", np.zeros(n)), length=n, name="logical.x")
            z = binary_vector(logical.get("z", np.zeros(n)), length=n, name="logical.z")
    else:
        vector = binary_vector(logical, length=n, name="logical")
        x = vector.copy() if pauli in ("X", "Y") else np.zeros(n, dtype=np.uint8)
        z = vector.copy() if pauli in ("Z", "Y") else np.zeros(n, dtype=np.uint8)
        sign = 0
    if not np.any(x | z):
        raise ValueError("Cannot measure the identity")
    if np.any((x @ source.hz.T) % 2) or np.any((z @ source.hx.T) % 2):
        raise ValueError("Target must commute with all memory stabilizers")
    if in_rowspace(x, source.hx) and in_rowspace(z, source.hz):
        raise ValueError("Target is a stabilizer, not a nontrivial logical")
    if source.distance is not None and int(np.count_nonzero(x | z)) < source.distance:
        raise ValueError("Target weight contradicts the declared memory-code distance")
    return source, (x, z, int(sign)), basis


def memory_checks(code):
    return (np.vstack([code.hx, np.zeros_like(code.hz)]),
            np.vstack([np.zeros_like(code.hx), code.hz]))


def signature(x, z, basis):
    return np.concatenate([(x @ basis["jz"].T) % 2, (z @ basis["jx"].T) % 2], axis=-1)


def restriction(code, target):
    x, z, _ = target
    support = np.flatnonzero(x | z)
    cx, cz = memory_checks(code)
    full = (cx[:, support] * z[support] + cz[:, support] * x[support]) % 2
    rows = np.flatnonzero(np.any(full, axis=1))
    f = full[rows]
    pasting = np.zeros((len(cx), len(rows)), dtype=np.uint8)
    pasting[rows, np.arange(len(rows))] = 1
    return support, f, pasting, rows


def inspect_restriction(code, target, basis, *, require_graph=False, max_entries=50_000_000):
    """Check logical readout; ordinary-graph conditions apply only to adapters.

    The <2d shortcut applies only to the Pauli-aligned logical precheck.
    """
    support, f, pasting, rows = restriction(code, target)
    w = len(support)
    budget_check(max_entries, len(f) * w, w * w)
    weights = f.sum(axis=1)
    invalid = np.flatnonzero(weights != 2)
    if require_graph and len(invalid):
        raise GaugeMeasurementRequired("Joint-measurement adapter: restricted F row weight is not 2",
            details=dict(restriction_rows=invalid.tolist(), memory_check_rows=rows[invalid].tolist(),
                         row_weights=weights[invalid].tolist()))
    if code.distance is not None and w < 2 * code.distance:
        precheck = dict(method="weight_lt_2d", kernel_enumerated=False, support_weight=w,
                        asserted_distance=code.distance,
                        note="Certifies no independent Pauli-aligned logical submeasurement only")
    else:
        k = nullspace(f)
        qx = np.zeros((len(k), code.n), dtype=np.uint8)
        qz = np.zeros_like(qx)
        qx[:, support] = k * target[0][support]
        qz[:, support] = k * target[1][support]
        gamma = row_basis(signature(qx, qz, basis))
        a = signature(target[0], target[1], basis)
        if len(gamma) != 1 or not in_rowspace(a, gamma):
            raise GaugeMeasurementRequired("Restricted F reads unintended independent logical operators",
                details=dict(logical_rank=len(gamma), measured_signatures=gamma.tolist(), target_signature=a.tolist()))
        precheck = dict(method="kernel_signature", kernel_enumerated=True, kernel_basis=k,
                        logical_signature=gamma, support_weight=w)
    # A single Tanner gadget may have higher-weight rows or disconnected
    # components representing only supported stabilizers. SkipTree may not.
    if require_graph:
        from .universal_adapters.graphs import graph_matrix
        try:
            graph_matrix(f, max_entries=max_entries)
        except ValueError as error:
            raise GaugeMeasurementRequired("Restricted F is disconnected and cannot enter SkipTree",
                                           details=dict(cause=str(error))) from error
    return dict(support=support, restriction_matrix=f, pasting_matrix=pasting,
                touching_memory_check_rows=rows, precheck=precheck)


def validate_readout(code, cx, cz, target, basis):
    n = code.n
    u = nullspace(np.hstack([cx[:, n:], cz[:, n:]]).T)
    rx, rz = (u @ cx[:, :n]) % 2, (u @ cz[:, :n]) % 2
    if np.any((rx @ code.hz.T) % 2) or np.any((rz @ code.hx.T) % 2):
        raise ValueError("Deformed checks read a non-normalizer memory operator")
    gamma = row_basis(signature(rx, rz, basis))
    a = signature(target[0], target[1], basis)
    if len(gamma) != 1 or not in_rowspace(a, gamma):
        raise GaugeMeasurementRequired("Actual deformed checks read unintended logical operators",
            details=dict(logical_rank=len(gamma), measured_signatures=gamma.tolist(), target_signature=a.tolist()))
    return dict(logical_measurement_rank=1, measurement_validation=dict(
        auxiliary_cancellation_kernel=u, memory_image_x=rx, memory_image_z=rz,
        measured_signature_basis=gamma, target_signature=a, logical_rank=1,
        coordinate_order="logical X coefficients, then logical Z coefficients",
        validated_against_actual_deformed_checks=True))


def consistent_signs(cx, cz, witness, target_sign, fixed_positive):
    dependencies = nullspace(np.hstack([cx, cz]).T)
    equations, rhs = [], []
    for u in dependencies:
        _, _, phase = pauli_product(cx, cz, selected=np.flatnonzero(u))
        if phase % 2:
            raise ValueError("Non-Hermitian dependency in gadget")
        equations.append(u)
        rhs.append(phase // 2)
    _, _, phase = pauli_product(cx, cz, selected=np.flatnonzero(witness))
    if phase % 2:
        raise ValueError("Non-Hermitian target witness")
    equations.append(witness)
    rhs.append(target_sign ^ (phase // 2))
    for index in fixed_positive:
        row = np.zeros(len(cx), dtype=np.uint8)
        row[index] = 1
        equations.append(row)
        rhs.append(0)
    return solve(np.array(equations, dtype=np.uint8), np.array(rhs, dtype=np.uint8))


def prepare_pair(code, left=None, right=None, *, paulis=("Z", "Z"),
                 logical_indices=(0, 1), logical_basis=None, max_entries=50_000_000):
    if len(paulis) != 2 or len(logical_indices) != 2:
        raise ValueError("paulis and logical_indices must each have two entries")
    source, l, basis = prepare(code, left, pauli=paulis[0], logical_index=logical_indices[0],
                               logical_basis=logical_basis, max_entries=max_entries)
    _, r, _ = prepare(source, right, pauli=paulis[1], logical_index=logical_indices[1],
                       logical_basis=basis, max_entries=max_entries)
    overlap = np.flatnonzero((l[0] | l[1]) & (r[0] | r[1]))
    if len(overlap):
        raise GaugeMeasurementRequired(
            "The disjoint-support adapter route does not accept overlapping factors; "
            "supply their full Hermitian Pauli target to gauging_deform (use pauli='Y' for logical Y)",
            details=dict(overlap_qubits=overlap.tolist()))
    x, z, phase = pauli_product(np.vstack([l[0], r[0]]), np.vstack([l[1], r[1]]), [l[2], r[2]])
    if phase % 2:
        raise ValueError("Product is anti-Hermitian; supply a Hermitian observable")
    _, target, _ = prepare(source, dict(x=x, z=z, sign=phase // 2),
                           logical_basis=basis, max_entries=max_entries)
    return source, l, r, target, basis
