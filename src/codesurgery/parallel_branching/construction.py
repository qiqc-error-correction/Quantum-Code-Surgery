"""Two-layer branching stickers for arbitrary signed stabilizer checks.

This is the matrix form of the locally compatible branching construction in
Lemma 2.10 of arXiv:2503.05003v3. The local Pauli axis is chosen once on each
site in the union. Branching introduces no logical measurement: targets move
to leaf representatives, while the full logical space has an explicit lift.
"""
from __future__ import annotations

import numpy as np

from ..gf2 import binary_matrix, binary_vector


def _signed_check_rank(x, z, signs):
    """Validate commutation and redundant-generator phases with packed GF(2)."""
    n = x.shape[1]
    basis = {}
    for xr, zr, sign in zip(x, z, signs):
        xb = sum(int(bit) << j for j, bit in enumerate(xr))
        zb = sum(int(bit) << j for j, bit in enumerate(zr))
        phase_sign = int(sign)
        for bx, bz, _ in basis.values():
            if ((xb & bz) ^ (zb & bx)).bit_count() & 1:
                raise ValueError("Stabilizer checks do not commute")
        while xb or zb:
            pivot = (xb | (zb << n)).bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (xb, zb, phase_sign)
                break
            bx, bz, bs = basis[pivot]
            nx, nz = xb ^ bx, zb ^ bz
            phase = (2 * phase_sign + 2 * bs + (xb & zb).bit_count()
                     + (bx & bz).bit_count() - (nx & nz).bit_count()
                     + 2 * (zb & bx).bit_count()) % 4
            if phase & 1:
                raise ValueError("Non-Hermitian stabilizer relation")
            xb, zb, phase_sign = nx, nz, phase // 2
        else:
            if phase_sign:
                raise ValueError("Inconsistent stabilizer signs: the group contains -I")
    return len(basis)


def append_branch(check_x, check_z, check_signs, local_x, local_z, *,
                  leaf_pauli="Z", max_entries=50_000_000):
    """Append one sticker; return full symplectic checks and its local maps.

    Columns are ``(memory, a, b)`` with ``a`` in bijection with incident old
    checks and ``b`` in bijection with the support of the local Pauli union.
    Rows are ``(old, junction, cap)``. ``F`` is the retained incident-check
    restriction and ``T`` its selector, so ``T @ F`` is the full restriction.

    With Z leaves, junction j is P_j Z_a(F[:,j]) Z_bj and cap s is
    X_as X_b(F[s,:]). X leaves exchange X/Z on new ancillas only; no new
    ancilla carries Y, so this exchange creates no sign corrections.
    ``max_entries`` bounds returned dense-array storage, checked before its
    allocation. Inputs are validated and copied, and never modified.
    """
    if (isinstance(max_entries, (bool, np.bool_))
            or not isinstance(max_entries, (int, np.integer)) or max_entries < 1):
        raise ValueError("max_entries must be a positive integer")
    if not isinstance(leaf_pauli, str) or leaf_pauli.upper() not in {"X", "Z"}:
        raise ValueError("leaf_pauli must be X or Z")
    leaf_pauli = leaf_pauli.upper()
    x = binary_matrix(check_x, name="check_x")
    z = binary_matrix(check_z, name="check_z")
    if x.shape != z.shape or x.shape[1] == 0:
        raise ValueError("check_x/check_z must have equal shapes and positive width")
    m, n = x.shape
    signs = binary_vector(check_signs, length=m, name="check_signs")
    px = binary_vector(local_x, length=n, name="local_x")
    pz = binary_vector(local_z, length=n, name="local_z")
    support = np.flatnonzero(px | pz)
    v = len(support)
    if v == 0:
        raise ValueError("Cannot branch an empty local Pauli union")
    old_rank = _signed_check_rank(x, z, signs)

    # Find incident rows without allocating the m-by-v full restriction.
    incident = np.zeros(m, dtype=bool)
    for q in support:
        incident |= ((x[:, q] * pz[q]) ^ (z[:, q] * px[q])).astype(bool)
    incident_rows = np.flatnonzero(incident)
    r = len(incident_rows)
    new_n, new_m = n + r + v, m + r + v
    # Checks, signs, F, T, local axes, and all returned index vectors.
    entries = (2 * new_m * new_n + new_m + r * v + m * r + 2 * n
               + v + r + r + v + v + r)
    if entries > int(max_entries):
        raise ValueError(f"Branching needs {entries} dense entries, above max_entries={max_entries}")

    f = np.zeros((r, v), dtype=np.uint8)
    for j, q in enumerate(support):
        f[:, j] = (x[incident_rows, q] * pz[q]) ^ (z[incident_rows, q] * px[q])
    t = np.zeros((m, r), dtype=np.uint8)
    t[incident_rows, np.arange(r)] = 1
    out_x, out_z = np.zeros((new_m, new_n), np.uint8), np.zeros((new_m, new_n), np.uint8)
    out_x[:m, :n], out_z[:m, :n] = x, z
    dual_columns = np.arange(n, n + r)
    leaf_columns = np.arange(n + r, new_n)
    junction_rows = np.arange(m, m + v)
    cap_rows = np.arange(m + v, new_m)
    # For the dual sticker fill the swapped ancilla axes directly, avoiding
    # an additional full-array copy just to implement local Hadamards.
    anti_axis = out_x if leaf_pauli == "Z" else out_z
    leaf_axis = out_z if leaf_pauli == "Z" else out_x
    anti_axis[incident_rows, dual_columns] = 1
    out_x[junction_rows, support] = px[support]
    out_z[junction_rows, support] = pz[support]
    leaf_axis[m:m + v, n:n + r] = f.T
    leaf_axis[junction_rows, leaf_columns] = 1
    anti_axis[cap_rows, dual_columns] = 1
    anti_axis[m + v:, n + r:] = f
    out_signs = np.zeros(new_m, dtype=np.uint8)
    out_signs[:m] = signs
    return dict(check_x=out_x, check_z=out_z, check_signs=out_signs,
                old_n=n, old_checks=m, support=support, local_x=px, local_z=pz,
                F=f, T=t, incident_rows=incident_rows, dual_columns=dual_columns,
                leaf_columns=leaf_columns, junction_rows=junction_rows, cap_rows=cap_rows,
                leaf_pauli=leaf_pauli, old_rank=old_rank,
                rank_increment=r + v, dense_entries=entries)


def _pauli_rows(x, z, n):
    vector = np.asarray(x).ndim == 1
    if vector:
        xx = binary_vector(x, length=n, name="Pauli x").reshape(1, n)
        zz = binary_vector(z, length=n, name="Pauli z").reshape(1, n)
    else:
        xx = binary_matrix(x, n_cols=n, name="Pauli x")
        zz = binary_matrix(z, n_cols=n, name="Pauli z")
    if xx.shape != zz.shape:
        raise ValueError("Pauli x/z arrays must have equal shapes")
    return xx, zz, vector


def lift_paulis(x, z, signs, branch):
    """Lift original Paulis; vectors or row matrices retain their input shape.

    Add X (Z for X leaves) to leaf j when the original Pauli anticommutes
    locally with P_j. These new factors are on separate qubits and do not
    change the literal-Pauli sign. In particular, locally compatible targets
    have no extra support under the lift, before being cleaned to the leaves.
    """
    xx, zz, vector = _pauli_rows(x, z, int(branch["old_n"]))
    if vector and np.asarray(signs).ndim == 0:
        ss = binary_vector([signs], length=1, name="Pauli signs")
    else:
        ss = binary_vector(signs, length=len(xx), name="Pauli signs")
    n, new_n = int(branch["old_n"]), branch["check_x"].shape[1]
    sx = np.zeros((len(xx), new_n), dtype=np.uint8)
    sz = np.zeros_like(sx)
    sx[:, :n], sz[:, :n] = xx, zz
    support = branch["support"]
    r = ((xx[:, support] * branch["local_z"][support])
         ^ (zz[:, support] * branch["local_x"][support]))
    (sx if branch["leaf_pauli"] == "Z" else sz)[:, branch["leaf_columns"]] = r
    return (sx[0], sz[0], int(ss[0])) if vector else (sx, sz, ss)


def retract_paulis(x, z, branch):
    """Binary retraction to the old code; signs intentionally are not mapped.

    A leaf Z factor (X for X leaves) maps to its one-site original P_j.
    This map descends to a logical isomorphism and cannot increase Pauli
    weight. It is not asserted to be a physical Clifford on arbitrary Paulis.
    """
    xx, zz, vector = _pauli_rows(x, z, branch["check_x"].shape[1])
    n, support = int(branch["old_n"]), branch["support"]
    rx, rz = xx[:, :n].copy(), zz[:, :n].copy()
    leaf = (zz if branch["leaf_pauli"] == "Z" else xx)[:, branch["leaf_columns"]]
    rx[:, support] ^= leaf * branch["local_x"][support]
    rz[:, support] ^= leaf * branch["local_z"][support]
    return (rx[0], rz[0]) if vector else (rx, rz)


__all__ = ["append_branch", "lift_paulis", "retract_paulis"]
