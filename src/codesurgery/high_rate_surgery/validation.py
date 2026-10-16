"""Independent Pauli, chain-map, and complete readout checks for high-rate surgery."""
from __future__ import annotations

import numpy as np

from ..gf2 import matmul, nullspace, rank, row_basis, solve
from ..symplectic import commutation_matrix, pauli_product


def merged_checks(source_x, source_z, state):
    """Append shared ancilla checks in order (source, attachment, cycles)."""
    b, c, t = state.boundary, state.cycles, state.gamma0
    px, pz = state.attachment_x, state.attachment_z
    m, n = source_x.shape
    a, s = b.shape
    x = np.zeros((m + s + len(c), n + a), np.uint8)
    z = np.zeros_like(x)
    x[:m, :n], z[:m, :n] = source_x, source_z
    z[:m, n:] = t
    x[m:m+s, :n], z[m:m+s, :n] = px, pz
    x[m:m+s, n:] = b.T
    z[m+s:, n:] = c
    return x, z


def weight_report(x, z):
    support = x | z  # A Y contributes one, not two.
    rows, cols = support.sum(axis=1), support.sum(axis=0)
    return dict(max_check_weight=int(rows.max(initial=0)),
                max_qubit_degree=int(cols.max(initial=0)),
                row_weights=rows, column_weights=cols)


def check_chain_maps(source_x, source_z, state):
    b, c, t = state.boundary, state.cycles, state.gamma0
    px, pz = state.attachment_x, state.attachment_z
    m, n = source_x.shape
    if source_z.shape != (m, n) or px.shape != pz.shape or px.shape != (b.shape[1], n):
        raise ValueError("High-rate attachment dimensions are inconsistent")
    if c.shape[1] != len(b) or t.shape != (m, len(b)):
        raise ValueError("High-rate boundary/cycle/chain-map dimensions are inconsistent")
    if np.any(matmul(c, b)):
        raise ValueError("Ancilla chain condition C @ B = 0 failed")
    f = matmul(source_x, pz.T) ^ matmul(source_z, px.T)
    if not np.array_equal(matmul(t, b), f):
        raise ValueError("Chain map condition Gamma0 @ B = S Omega P.T failed")
    if np.any(commutation_matrix(px, pz)):
        raise ValueError("Memory attachment Paulis do not commute with each other")


def inspect_readout(code, basis, x, z, target_signatures):
    """Eliminate ALL ancilla support from the actual final stabilizer group."""
    n = code.n
    eliminate = nullspace(np.hstack((x[:, n:], z[:, n:])).T)
    mx, mz = matmul(eliminate, x[:, :n]), matmul(eliminate, z[:, :n])
    if np.any(matmul(mx, code.hz.T)) or np.any(matmul(mz, code.hx.T)):
        raise ValueError("Readout includes a Pauli outside the original code normalizer")
    sig = np.hstack((matmul(mx, basis['jz'].T), matmul(mz, basis['jx'].T)))
    measured, requested = row_basis(sig), row_basis(target_signatures)
    if len(measured) != len(requested) or rank(np.vstack((measured, requested))) != len(requested):
        raise ValueError("High-rate surgery would measure an unintended logical space; "
                         "the actual readout must equal the requested span exactly")
    return dict(auxiliary_cancellation_kernel=eliminate, memory_x=mx, memory_z=mz,
                logical_signatures=sig, measured_signature_basis=measured,
                requested_signature_basis=requested, logical_rank=len(measured),
                validated_against_actual_deformed_checks=True)


def consistent_check_signs(code, source_x, source_z, x, z, readout, attachment_count):
    """Keep the initial data/ancilla Z sector positive, with no hidden -I.

    Every memory-only stabilizer product must have its original CSS sign.
    Measurement outcomes are encoded separately by signed readout corrections,
    so dependent or oppositely signed target requests remain meaningful.
    """
    sig = readout['logical_signatures']
    coeff = matmul(nullspace(sig.T), readout['auxiliary_cancellation_kernel'])
    memory = np.hstack((matmul(coeff, x[:, :code.n]), matmul(coeff, z[:, :code.n])))
    source_coeff = solve(np.hstack((source_x, source_z)).T, memory.T).T
    rhs = []
    for selected, original in zip(coeff, source_coeff):
        _, _, actual_phase = pauli_product(x, z, selected=np.flatnonzero(selected))
        _, _, desired_phase = pauli_product(source_x, source_z, selected=np.flatnonzero(original))
        if actual_phase % 2 or desired_phase % 2:
            raise ValueError("Non-Hermitian check product")
        rhs.append((actual_phase ^ desired_phase) // 2)
    m = len(source_x)
    # Only attachment-generator signs are free. Source and pure ancilla Z
    # checks are initially +1 under the data-code / |0> ancilla preparation.
    signs = np.zeros(len(x), np.uint8)
    signs[m:m+attachment_count] = solve(coeff[:, m:m+attachment_count],
                                        np.asarray(rhs, np.uint8))
    return signs


def validate_signed_group(code, source_x, source_z, x, z, signs, readout):
    if np.any(commutation_matrix(x, z)):
        raise ValueError("High-rate deformed checks do not commute")
    relations = matmul(nullspace(readout['logical_signatures'].T),
                       readout['auxiliary_cancellation_kernel'])
    memory = np.hstack((matmul(relations, x[:, :code.n]),
                        matmul(relations, z[:, :code.n])))
    original_coeff = solve(np.hstack((source_x, source_z)).T, memory.T).T
    for selected, original in zip(relations, original_coeff):
        _, _, actual = pauli_product(x, z, signs, np.flatnonzero(selected))
        _, _, expected = pauli_product(source_x, source_z, selected=np.flatnonzero(original))
        if actual != expected:
            raise ValueError("Deformed checks contain -I or contradict an original stabilizer sign")


def readout_witnesses(x, z, signs, target_x, target_z, target_signs):
    n = target_x.shape[1]
    tx = np.pad(target_x, ((0, 0), (0, x.shape[1]-n)))
    tz = np.pad(target_z, ((0, 0), (0, x.shape[1]-n)))
    witnesses = solve(np.hstack((x, z)).T, np.hstack((tx, tz)).T).T
    corrections = []
    for w, sx, sz, ss in zip(witnesses, tx, tz, target_signs):
        px, pz, phase = pauli_product(x, z, signs, np.flatnonzero(w))
        if phase % 2 or not np.array_equal(px, sx) or not np.array_equal(pz, sz):
            raise ValueError("Invalid target measurement witness")
        corrections.append(phase // 2 ^ int(ss))
    return witnesses, np.asarray(corrections, np.uint8)
