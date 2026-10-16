"""Zhang--Li measurement stickers and the actual deformed check group.

References: arXiv:2408.01339v3, Appendix C equations (37)--(38),
Appendix D equations (45)--(46), and Appendix E equations (78)--(79).
Columns are grouped as memory, primal layers, then dual layers.  ``d_R``
counts dual layers; there are ``d_R - 1`` primal layers.

These functions assemble a supplied compatible glue. They neither choose
additional glue checks nor use CKBB surgery to repair a readout mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..gf2 import binary_matrix, in_rowspace, inverse, matmul, nullspace, rank, row_basis, solve


class DevisedReadoutError(ValueError):
    """The actual devised-sticking readout differs from the target space."""

    def __init__(self, logical_rank, measured_signatures, target_signatures):
        self.logical_rank = int(logical_rank)
        self.measured_signatures = np.array(measured_signatures, dtype=np.uint8, copy=True)
        self.target_signatures = np.array(target_signatures, dtype=np.uint8, copy=True)
        self.target_rank = rank(self.target_signatures)
        super().__init__(
            "Devised sticking does not read exactly the requested logical space: "
            f"measured rank {self.logical_rank}, requested rank {self.target_rank}; "
            f"measured signatures {self.measured_signatures.tolist()}, "
            f"requested signatures {self.target_signatures.tolist()}"
        )


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _pauli_type(pauli):
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
        raise ValueError("Measurement-sticker pauli must be X or Z")
    return pauli.upper()


def _check_budget(rows, columns, max_entries):
    max_entries = _positive_integer(max_entries, "max_entries")
    if int(rows) * int(columns) > max_entries:
        raise ValueError("Dense devised-sticking result exceeds max_entries; reduce d_R or increase the explicit limit")


def _glue_field(glue, name):
    aliases = {"h_g": ("h_g", "hg", "H_G"), "s": ("s", "S"), "t": ("t", "T")}[name]
    for alias in aliases:
        if isinstance(glue, Mapping) and alias in glue:
            return glue[alias]
        if hasattr(glue, alias):
            return getattr(glue, alias)
    raise ValueError(f"glue must supply {name}")


def _glue_arrays(code, glue, pauli):
    hg = binary_matrix(_glue_field(glue, "h_g"), name="H_G")
    if hg.shape[1] < 1:
        raise ValueError("H_G must have at least one glue-bit column")
    s = binary_matrix(_glue_field(glue, "s"), n_cols=code.n, name="S")
    opposite = code.hx if pauli == "Z" else code.hz
    t = binary_matrix(_glue_field(glue, "t"), n_cols=len(hg), name="T")
    if s.shape[0] != hg.shape[1] or t.shape[0] != len(opposite):
        raise ValueError("Glue shapes must be H_G=(r_G,n_G), S=(n_G,n), T=(r_opposite,r_G)")
    if not np.array_equal(matmul(opposite, s.T), matmul(t, hg)):
        raise ValueError("Incompatible glue: H_opposite @ S.T must equal T @ H_G over GF(2)")
    return hg, s, t


def measurement_sticker(h_g, d_R, *, max_entries=50_000_000):
    """Return the CSS measurement sticker, equations (37)--(38).

    ``H_X=[I_(d_R-1) kron H_G | lambda_(d_R) kron I_rG]`` and
    ``H_Z=[lambda_(d_R).T kron I_nG | I_dR kron H_G.T]``.
    The implementation fills sparse blocks into dense outputs, checking
    their combined entry count before allocating them.
    """
    hg = binary_matrix(h_g, name="H_G")
    if hg.shape[1] < 1:
        raise ValueError("H_G must have at least one glue-bit column")
    d_R = _positive_integer(d_R, "d_R")
    rg, ng = hg.shape
    n_sticker = (d_R - 1) * ng + d_R * rg
    rows_x, rows_z = (d_R - 1) * rg, d_R * ng
    _check_budget(rows_x + rows_z, n_sticker, max_entries)
    hx = np.zeros((rows_x, n_sticker), dtype=np.uint8)
    hz = np.zeros((rows_z, n_sticker), dtype=np.uint8)
    primal = np.arange((d_R - 1) * ng).reshape(d_R - 1, ng)
    dual = np.arange((d_R - 1) * ng, n_sticker).reshape(d_R, rg)
    for layer in range(d_R - 1):
        rows = np.arange(layer * rg, (layer + 1) * rg)
        hx[np.ix_(rows, primal[layer])] = hg
        hx[rows, dual[layer]] = 1
        hx[rows, dual[layer + 1]] = 1
    for layer in range(d_R):
        rows = np.arange(layer * ng, (layer + 1) * ng)
        if layer > 0:
            hz[rows, primal[layer - 1]] = 1
        if layer < d_R - 1:
            hz[rows, primal[layer]] = 1
        hz[np.ix_(rows, dual[layer])] = hg.T
    return hx, hz


def assemble_deformed(code, glue, *, pauli="Z", d_R, max_entries=50_000_000):
    """Paste a supplied glue using equations (45)--(46).

    Z measurement extends memory X checks by T on the first dual layer,
    and adds S to the first block of sticker Z checks. X measurement uses
    the same construction with the input/output X and Z roles exchanged.
    The returned physical order is ``[memory, all primal, all dual]``.
    """
    pauli = _pauli_type(pauli)
    d_R = _positive_integer(d_R, "d_R")
    hg, s, t = _glue_arrays(code, glue, pauli)
    opposite = code.hx if pauli == "Z" else code.hz
    same = code.hz if pauli == "Z" else code.hx
    rg, ng = hg.shape
    n, n_sticker = code.n, (d_R - 1) * ng + d_R * rg
    rows_opposite = len(opposite) + (d_R - 1) * rg
    rows_same = len(same) + d_R * ng
    n_out = n + n_sticker
    _check_budget(rows_opposite + rows_same, n_out, max_entries)
    merged_opposite = np.zeros((rows_opposite, n_out), dtype=np.uint8)
    merged_same = np.zeros((rows_same, n_out), dtype=np.uint8)
    primal = np.arange(n, n + (d_R - 1) * ng).reshape(d_R - 1, ng)
    dual = np.arange(n + (d_R - 1) * ng, n_out).reshape(d_R, rg)
    merged_opposite[:len(opposite), :n] = opposite
    merged_same[:len(same), :n] = same
    merged_opposite[:len(opposite), dual[0]] = t
    measurement_rows = np.arange(len(same), rows_same).reshape(d_R, ng)
    opposite_rows = np.arange(len(opposite), rows_opposite).reshape(d_R - 1, rg)
    for layer in range(d_R - 1):
        rows = opposite_rows[layer]
        merged_opposite[np.ix_(rows, primal[layer])] = hg
        merged_opposite[rows, dual[layer]] = 1
        merged_opposite[rows, dual[layer + 1]] = 1
    for layer in range(d_R):
        rows = measurement_rows[layer]
        if layer == 0:
            merged_same[np.ix_(rows, np.arange(n))] = s
        if layer > 0:
            merged_same[rows, primal[layer - 1]] = 1
        if layer < d_R - 1:
            merged_same[rows, primal[layer]] = 1
        merged_same[np.ix_(rows, dual[layer])] = hg.T
    if np.any(matmul(merged_opposite, merged_same.T)):
        raise RuntimeError("Devised-sticking commutation invariant failed")
    hx, hz = (merged_opposite, merged_same) if pauli == "Z" else (merged_same, merged_opposite)
    return hx, hz, {
        "pauli": pauli, "d_R": d_R, "layers": d_R,
        "column_order": "memory, primal layers, dual layers",
        "memory_qubit_columns": np.arange(n),
        "primal_qubit_columns": primal, "dual_qubit_columns": dual,
        "measurement_check_rows": measurement_rows,
        "opposite_sticker_check_rows": opposite_rows,
        "original_same_check_rows": np.arange(len(same)),
        "original_opposite_check_rows": np.arange(len(opposite)),
        "n_G": ng, "r_G": rg, "deformed_n": n_out,
        "sticker_n": n_sticker,
        "induced_hamming_norm_s": int(np.max(np.sum(s, axis=1))),
    }


def validate_readout(code, hx, hz, targets, *, pauli, max_entries=50_000_000):
    """Require the complete memory-only logical readout to equal targets.

    Eliminate every auxiliary X and Z component from the actual returned
    CSS checks, then compare logical signature spaces. This is independent
    of any claimed glue kernel or of a supplied target witness. Dependent
    target rows are accepted; their independent logical rank determines q.
    """
    pauli = _pauli_type(pauli)
    hx = binary_matrix(hx, name="deformed H_X")
    hz = binary_matrix(hz, name="deformed H_Z")
    if hx.shape[1] != hz.shape[1] or hx.shape[1] < code.n:
        raise ValueError("Deformed CSS matrices must have matching widths and include all memory qubits")
    # The coefficient kernel can have one coordinate per check, even when
    # many redundant checks act on only a small number of physical qubits.
    check_count = len(hx) + len(hz)
    _check_budget(check_count, max(check_count, 2 * hx.shape[1]), max_entries)
    if np.any(matmul(hx, hz.T)):
        raise ValueError("Deformed CSS checks do not commute")
    targets = binary_matrix(targets, n_cols=code.n, name="targets")
    if len(targets) == 0 or np.any(~np.any(targets, axis=1)):
        raise ValueError("targets must contain at least one nonidentity operator and no identity rows")
    opposite = code.hx if pauli == "Z" else code.hz
    if np.any(matmul(targets, opposite.T)):
        raise ValueError("Requested targets must commute with the input memory checks")
    basis = code.standard_logicals()
    x_memory = np.vstack((hx[:, :code.n], np.zeros((len(hz), code.n), dtype=np.uint8)))
    z_memory = np.vstack((np.zeros((len(hx), code.n), dtype=np.uint8), hz[:, :code.n]))
    n_aux = hx.shape[1] - code.n
    auxiliary_x = np.vstack((hx[:, code.n:], np.zeros((len(hz), n_aux), dtype=np.uint8)))
    auxiliary_z = np.vstack((np.zeros((len(hx), n_aux), dtype=np.uint8), hz[:, code.n:]))
    cancel_auxiliary = nullspace(np.hstack((auxiliary_x, auxiliary_z)).T)
    memory_x, memory_z = matmul(cancel_auxiliary, x_memory), matmul(cancel_auxiliary, z_memory)
    if np.any(matmul(memory_x, code.hz.T)) or np.any(matmul(memory_z, code.hx.T)):
        raise ValueError("Devised checks read a memory operator outside its normalizer")
    signatures = np.hstack((matmul(memory_x, basis.jz.T), matmul(memory_z, basis.jx.T)))
    coefficient = matmul(targets, (basis.jx if pauli == "Z" else basis.jz).T)
    zeros = np.zeros_like(coefficient)
    target_signatures = np.hstack((zeros, coefficient)) if pauli == "Z" else np.hstack((coefficient, zeros))
    if np.any(~np.any(target_signatures, axis=1)):
        raise ValueError("Each requested target must be a nontrivial memory logical operator")
    measured_basis, target_basis = row_basis(signatures), row_basis(target_signatures)
    logical_rank, q = len(measured_basis), len(target_basis)
    if logical_rank != q or not in_rowspace(measured_basis, target_basis) or not in_rowspace(target_basis, measured_basis):
        raise DevisedReadoutError(logical_rank, measured_basis, target_basis)
    return {
        "logical_measurement_rank": logical_rank,
        "requested_logical_rank": q,
        "measurement_validation": {
            "auxiliary_cancellation_kernel": cancel_auxiliary,
            "memory_image_x": memory_x, "memory_image_z": memory_z,
            "logical_signature": signatures, "target_signature": target_signatures,
            "measured_signature_basis": measured_basis, "target_signature_basis": target_basis,
            "logical_rank": logical_rank, "requested_logical_rank": q,
            "coordinate_order": "logical X coefficients, then logical Z coefficients",
            "validated_against_actual_deformed_checks": True,
        },
    }


def protected_logical_generators(code, glue, targets, *, pauli="Z", d_R, max_entries=50_000_000):
    """Lift the unmeasured paired basis by equation (49), with Gamma.

    Return ``(jx_deformed, jz_deformed, metadata)``. If Q changes the
    measured-species logical rows, its conjugate basis changes by Q^{-T}.
    This transpose is required by the pairing convention; the inverse
    without transpose printed near equation (35) does not preserve it.
    The result defines the k-q protected qubits, rather than every raw
    stabilizer-code logical qubit of a deformation with gauge modes.
    """
    pauli = _pauli_type(pauli)
    d_R = _positive_integer(d_R, "d_R")
    hg, s, _ = _glue_arrays(code, glue, pauli)
    targets = binary_matrix(targets, n_cols=code.n, name="targets")
    basis = code.standard_logicals()
    same = basis.jz if pauli == "Z" else basis.jx
    conjugate = basis.jx if pauli == "Z" else basis.jz
    active_coefficients = row_basis(matmul(targets, conjugate.T))
    q, k = len(active_coefficients), code.k
    if q == 0:
        raise ValueError("targets must span a nontrivial logical subspace")
    transform = active_coefficients.copy()
    for unit in np.eye(k, dtype=np.uint8):
        if not in_rowspace(unit, transform):
            transform = np.vstack((transform, unit))
    same_rebased = matmul(transform, same)
    conjugate_rebased = matmul(inverse(transform).T, conjugate)
    memory_same, memory_conjugate = same_rebased[q:], conjugate_rebased[q:]
    pasted_conjugate = matmul(memory_conjugate, s.T)
    try:
        gamma = solve(hg.T, pasted_conjugate.T).T
    except ValueError as error:
        raise ValueError("Glue cannot lift the protected complement: J_conjugate,C @ S.T is not in rowspace(H_G)") from error
    rg, ng = hg.shape
    n_out = code.n + (d_R - 1) * ng + d_R * rg
    _check_budget(2 * (k - q), n_out, max_entries)
    same_out = np.zeros((k - q, n_out), dtype=np.uint8)
    conjugate_out = np.zeros_like(same_out)
    same_out[:, :code.n], conjugate_out[:, :code.n] = memory_same, memory_conjugate
    for layer in range(d_R - 1):
        start = code.n + layer * ng
        conjugate_out[:, start:start + ng] = pasted_conjugate
    last_dual = code.n + (d_R - 1) * ng + (d_R - 1) * rg
    conjugate_out[:, last_dual:last_dual + rg] = gamma
    jx, jz = (conjugate_out, same_out) if pauli == "Z" else (same_out, conjugate_out)
    if not np.array_equal(matmul(jx, jz.T), np.eye(k - q, dtype=np.uint8)):
        raise RuntimeError("Protected devised-sticking logical pairing failed")
    return jx, jz, {
        "protected_k": k - q, "logical_measurement_rank": q,
        "logical_basis_transform": transform, "gamma": gamma,
        "memory_same_complement": memory_same,
        "memory_conjugate_complement": memory_conjugate,
        "gamma_pauli": "X" if pauli == "Z" else "Z",
    }
