"""Small, deterministic linear-algebra helpers over GF(2).

Matrices use rows for check/operator vectors. Entries must already be binary;
invalid values are rejected rather than silently reduced modulo two. All
returned binary arrays have dtype ``numpy.uint8`` and own their data.
"""

from __future__ import annotations

import numpy as np


def _binary_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "buif":
        raise ValueError(f"{name} must contain numeric binary entries (0 or 1)")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return np.array(array, dtype=np.uint8, copy=True)


def binary_matrix(
    value: object, *, n_cols: int | None = None, name: str = "matrix"
) -> np.ndarray:
    """Validate/copy a binary 2-D matrix; ``[]`` denotes zero rows.

    For an empty Python list use ``n_cols=n`` to retain the physical width.
    Explicit arrays of shape ``(0, n)`` retain their width automatically.
    Nonempty one-dimensional inputs are rejected, even when they contain one
    row; wrap them in a list or use :func:`binary_vector`.
    """
    if n_cols is not None and (
        isinstance(n_cols, (bool, np.bool_))
        or not isinstance(n_cols, (int, np.integer))
        or n_cols < 0
    ):
        raise ValueError("n_cols must be a nonnegative integer")
    array = _binary_array(value, name=name)
    if array.ndim == 1 and array.size == 0:
        array = array.reshape(0, 0 if n_cols is None else n_cols)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if n_cols is not None and array.shape[1] != n_cols:
        raise ValueError(f"{name} must have {n_cols} columns, got {array.shape[1]}")
    return array


def binary_vector(
    value: object, *, length: int | None = None, name: str = "vector"
) -> np.ndarray:
    """Validate and copy a one-dimensional binary vector."""
    array = _binary_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and array.size != length:
        raise ValueError(f"{name} must have length {length}, got {array.size}")
    return array


def matmul(left: object, right: object) -> np.ndarray:
    """Matrix/vector multiplication in GF(2), with NumPy shape conventions."""
    a = _binary_array(left, name="left operand")
    b = _binary_array(right, name="right operand")
    if a.ndim not in (1, 2) or b.ndim not in (1, 2):
        raise ValueError("operands must be vectors or two-dimensional matrices")
    # uint8 overflow is harmless: removing multiples of 256 preserves parity.
    return np.asarray(np.matmul(a, b) & 1, dtype=np.uint8)


def rref(value: object) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return reduced row-echelon form and ascending pivot-column indices.

    Pivot selection is deterministic: scan columns from left to right and
    choose the first available row. Zero rows remain at the bottom. Inputs
    with redundant or zero checks are supported.
    """
    result = binary_matrix(value)
    n_rows, n_cols = result.shape
    row = 0
    pivots: list[int] = []
    for col in range(n_cols):
        candidates = np.flatnonzero(result[row:, col])
        if candidates.size == 0:
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            result[[row, pivot_row]] = result[[pivot_row, row]]
        targets = np.flatnonzero(result[:, col])
        targets = targets[targets != row]
        result[targets] ^= result[row]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    return result, tuple(pivots)


def rank(value: object) -> int:
    """Rank over GF(2)."""
    return len(rref(value)[1])


def row_basis(value: object) -> np.ndarray:
    """Return the nonzero RREF rows, an independent basis of the row space."""
    reduced, pivots = rref(value)
    return reduced[: len(pivots)].copy()


def nullspace(value: object) -> np.ndarray:
    """Return a row basis of ``{v: A @ v.T = 0}``.

    The returned shape is ``(n - rank(A), n)``. Free columns are taken in
    ascending order, making the basis deterministic.
    """
    reduced, pivots = rref(value)
    n_cols = reduced.shape[1]
    pivot_set = set(pivots)
    free = [col for col in range(n_cols) if col not in pivot_set]
    result = np.zeros((len(free), n_cols), dtype=np.uint8)
    for row, col in enumerate(free):
        result[row, col] = 1
        if pivots:
            result[row, list(pivots)] = reduced[: len(pivots), col]
    return result


def in_rowspace(vectors: object, basis: object) -> bool:
    """Whether every supplied vector belongs to ``rowspace(basis)``.

    ``vectors`` can be a single one-dimensional vector or a row matrix.
    Redundant basis rows are allowed. The empty collection is contained in
    every row space.
    """
    matrix = binary_matrix(basis, name="basis")
    candidates = _binary_array(vectors, name="vectors")
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    candidates = binary_matrix(candidates, n_cols=matrix.shape[1], name="vectors")
    reduced, pivots = rref(matrix)
    for row, col in enumerate(pivots):
        targets = np.flatnonzero(candidates[:, col])
        candidates[targets] ^= reduced[row]
    return not bool(np.any(candidates))


def solve(matrix: object, rhs: object) -> np.ndarray:
    """Solve ``A @ x = b`` in GF(2), setting all free variables to zero.

    ``rhs`` can be a vector or a matrix of right-hand-side columns. Raise
    ``ValueError`` if any right-hand side is inconsistent. The result has
    shape ``(A.shape[1],)`` or ``(A.shape[1], rhs.shape[1])``, respectively.
    """
    a = binary_matrix(matrix, name="matrix")
    b = _binary_array(rhs, name="rhs")
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = b[:, None]
    if b.ndim != 2 or b.shape[0] != a.shape[0]:
        raise ValueError("rhs must have one entry/row for each row of matrix")
    joined = np.concatenate((a, b), axis=1)
    row = 0
    pivots: list[int] = []
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(joined[row:, col])
        if candidates.size == 0:
            continue
        pivot_row = row + int(candidates[0])
        if pivot_row != row:
            joined[[row, pivot_row]] = joined[[pivot_row, row]]
        targets = np.flatnonzero(joined[:, col])
        targets = targets[targets != row]
        joined[targets] ^= joined[row]
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    if np.any(joined[row:, a.shape[1] :]):
        raise ValueError("linear system is inconsistent over GF(2)")
    result = np.zeros((a.shape[1], b.shape[1]), dtype=np.uint8)
    if pivots:
        result[list(pivots)] = joined[:row, a.shape[1] :]
    return result[:, 0] if vector_rhs else result


def inverse(value: object) -> np.ndarray:
    """Inverse of a nonsingular square matrix over GF(2)."""
    matrix = binary_matrix(value)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse requires a square matrix")
    if rank(matrix) != matrix.shape[0]:
        raise ValueError("matrix is singular over GF(2)")
    return solve(matrix, np.eye(matrix.shape[0], dtype=np.uint8))


# These spellings make conventions explicit at external API boundaries.
gf2_rank = rank
kernel = nullspace
