"""Paired CSS logical operators in Zhang--Li's simultaneous standard form.

For CSS *stabilizer* codes, ``k = n - rank(H_X) - rank(H_Z)``. This module
constructs representatives in Appendix A's form

    J_X^S = [I_k | 0_(k,r_X) | J_X']
    J_Z^S = [I_k | J_Z'       | 0_(k,r_Z)].

The construction uses stabilizer row operations and physical column
permutations; it does not minimize operator weights. Returned ``jx``/``jz``
are in the original physical order and obey ``jx @ jz.T = I_k`` in GF(2).
Phases are omitted, with +1 stabilizer signs assumed.

Reference: Guo Zhang and Ying Li, arXiv:2408.01339v3, Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import binary_matrix, inverse, matmul, rref, row_basis


@dataclass(frozen=True)
class LogicalBasis:
    """Logical generators and reproducible standard-form metadata.

    ``column_permutation[j]`` is the original physical column placed in
    standard position ``j``: ``jx_standard == jx[:, column_permutation]``.
    Blocks are ordered ``[logical_columns, x_pivots, z_pivots]``. Pivot lists
    contain indices in the *original* physical order. ``hx_standard`` and
    ``hz_standard`` are independent stabilizer bases after row reduction,
    not permutations of the original (possibly redundant) input rows.
    """

    jx: np.ndarray
    jz: np.ndarray
    column_permutation: np.ndarray
    jx_standard: np.ndarray
    jz_standard: np.ndarray
    hx_standard: np.ndarray
    hz_standard: np.ndarray
    x_pivots: tuple[int, ...]
    z_pivots: tuple[int, ...]
    logical_columns: tuple[int, ...]

    @property
    def k(self) -> int:
        return self.jx.shape[0]

    @property
    def n(self) -> int:
        return self.jx.shape[1]

    @property
    def inverse_column_permutation(self) -> np.ndarray:
        """Map an original physical position to its standard position."""
        return np.argsort(self.column_permutation)


def standard_form_logicals(hx: object, hz: object) -> LogicalBasis:
    """Construct a deterministic paired logical basis for a CSS stabilizer code.

    Redundant checks and zero-row matrices are accepted. Use an explicit
    ``np.zeros((0, n), dtype=np.uint8)`` if both check sets are empty but the
    code has ``n > 0`` qubits. If only one argument is ``[]``, the width of
    the other argument supplies its missing width.

    Algorithm: pick ``r_X`` pivot columns ``P_X`` of ``H_X``. Pick ``r_Z``
    pivot columns ``P_Z`` of ``H_Z`` outside ``P_X``. Commutation implies
    the latter restriction retains full row rank. Put the remaining ``k``
    columns first. Normalizing the two pivot blocks gives

        H_X^S = [A | I_(r_X) | B],
        H_Z^S = [C | D       | I_(r_Z)].

    Set ``J_X^S = [I_k | 0 | C.T]`` and
    ``J_Z^S = [I_k | A.T | 0]``. Their tails are disjoint, so their pairing
    is exactly ``I_k``; they commute with the opposite checks by direct
    cancellation. This simultaneously proves their independence modulo
    stabilizers. Finally map physical columns back to the caller's order.
    """
    x_raw = binary_matrix(hx, name="H_X")
    z_raw = binary_matrix(hz, name="H_Z")
    # [] has no width: infer it only when the other input supplies one.
    if x_raw.shape == (0, 0) and z_raw.shape[1] != 0:
        x_raw = np.zeros((0, z_raw.shape[1]), dtype=np.uint8)
    if z_raw.shape == (0, 0) and x_raw.shape[1] != 0:
        z_raw = np.zeros((0, x_raw.shape[1]), dtype=np.uint8)
    if x_raw.shape[1] != z_raw.shape[1]:
        raise ValueError("H_X and H_Z must have the same number of columns")
    if np.any(matmul(x_raw, z_raw.T)):
        raise ValueError("CSS checks must commute: H_X @ H_Z.T = 0 over GF(2)")

    n = x_raw.shape[1]
    x_reduced, x_pivots = rref(x_raw)
    r_x = len(x_pivots)
    x_basis = x_reduced[:r_x].copy()
    z_basis = row_basis(z_raw)
    r_z = z_basis.shape[0]
    x_pivot_set = set(x_pivots)
    outside_x = [col for col in range(n) if col not in x_pivot_set]
    _, restricted_z_pivots = rref(z_basis[:, outside_x])
    if len(restricted_z_pivots) != r_z:
        # This cannot occur for commuting CSS checks; keep an explicit guard
        # in case future changes violate the rank-preservation argument.
        raise RuntimeError("H_Z lost rank outside the H_X pivot columns")
    z_pivots = tuple(outside_x[col] for col in restricted_z_pivots)
    z_pivot_set = set(z_pivots)
    logical_columns = tuple(col for col in outside_x if col not in z_pivot_set)
    k = len(logical_columns)
    permutation = np.array(logical_columns + x_pivots + z_pivots, dtype=int)

    # H_X has already been normalized by RREF. H_Z must be normalized on
    # the selected pivots, which may differ from its original RREF pivots.
    z_basis = matmul(inverse(z_basis[:, list(z_pivots)]), z_basis)
    x_standard = x_basis[:, permutation]
    z_standard = z_basis[:, permutation]
    x_tail = z_standard[:, :k].T
    z_tail = x_standard[:, :k].T
    jx_standard = np.concatenate(
        (np.eye(k, dtype=np.uint8), np.zeros((k, r_x), dtype=np.uint8), x_tail),
        axis=1,
    )
    jz_standard = np.concatenate(
        (np.eye(k, dtype=np.uint8), z_tail, np.zeros((k, r_z), dtype=np.uint8)),
        axis=1,
    )
    jx = np.empty((k, n), dtype=np.uint8)
    jz = np.empty((k, n), dtype=np.uint8)
    jx[:, permutation] = jx_standard
    jz[:, permutation] = jz_standard
    return LogicalBasis(
        jx=jx,
        jz=jz,
        column_permutation=permutation,
        jx_standard=jx_standard,
        jz_standard=jz_standard,
        hx_standard=x_standard,
        hz_standard=z_standard,
        x_pivots=x_pivots,
        z_pivots=z_pivots,
        logical_columns=logical_columns,
    )
