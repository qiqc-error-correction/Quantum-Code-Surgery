"""Prepare signed logical targets without changing their logical classes.

This module prepares branching, which copies representatives and performs no
logical measurement. Reducible targets and repeated logical classes are valid.
Coordinates are physical ``(x | z)``; logical signatures are ``(a | b)`` in the
selected paired CSS basis. Source stabilizers have the + signs of ``CSSCode``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..gadget_common import logical_basis_for
from ..gf2 import matmul, rank, solve
from ..models import CSSCode
from ..symplectic import commutation_matrix, parse_pauli, pauli_product


def qubitwise_commuting(x, z):
    """Whether every pair has the same Pauli, or identity, at every site."""
    for i in range(len(x)):
        if np.any((x[i] & z[i + 1:]) ^ (z[i] & x[i + 1:])):
            return False
    return True


def _pauli_list(value, name):
    if isinstance(value, str):
        result = [value]
    else:
        if isinstance(value, Mapping):
            raise ValueError(f"{name} must be a string or iterable of Pauli strings")
        try:
            result = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a string or iterable of Pauli strings") from exc
    if not result:
        raise ValueError("At least one nontrivial logical target is required")
    if any(not isinstance(item, str) for item in result):
        raise ValueError(f"{name} must contain only Pauli strings")
    return result


def _logical_representative(a, b, sign, jx, jz):
    """Realize i^(a.b) Xbar(a) Zbar(b), including its physical Pauli sign."""
    x, z = matmul(a, jx), matmul(b, jz)
    phase = (int(np.sum(a * b)) - int(np.sum(x * z))) % 4
    if phase % 2:
        raise RuntimeError("The paired CSS logical basis produced a non-Hermitian Pauli")
    return x, z, int(sign) ^ (phase // 2)


def _indices(value, k):
    if value is None:
        result = list(range(k))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        result = [value]
    else:
        try:
            result = list(value)
        except TypeError as exc:
            raise ValueError("logical_indices must contain integers in [0,k)") from exc
    if not result:
        raise ValueError("At least one nontrivial logical target is required")
    for index in result:
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or not 0 <= index < k):
            raise ValueError("logical_indices must contain integers in [0,k)")
    return [int(index) for index in result]


def prepare_targets(code, targets=None, *, logical_paulis=None, pauli="Z",
                    logical_indices=None, logical_basis=None, representatives="auto"):
    """Prepare a CSS code and a nonempty list of branching targets.

    Choose one input mode: physical signed IXYZ strings in ``targets``;
    logical signed IXYZ strings of length k in ``logical_paulis``; or
    ``pauli`` plus zero-based ``logical_indices`` (default: Z on every logical
    qubit). A single string/index is accepted as a one-element collection.
    A supplied basis must be a dict with exactly k paired pure ``jx``/``jz``
    rows. Otherwise Zhang--Li standard-form representatives are used.

    ``preserve`` requires the supplied physical representatives to be QWC.
    ``auto`` preserves them exactly if QWC, otherwise tries the SAME logical
    classes in the selected basis. ``canonical`` always tries that rewrite.
    None of these modes changes the chosen basis to make targets compatible.
    Failure of the resulting QWC test raises ValueError; global commutation
    alone does not justify simultaneous branching.

    Each row w of ``canonicalization_witnesses`` satisfies the exact signed
    Pauli identity ``prepared = original * product(source_checks[w])``.
    Source-check order is all original HX rows followed by all HZ rows,
    retaining redundant/zero rows. A zero witness denotes exact preservation.
    Duplicate or linearly dependent targets are allowed; their logical rank
    is reported separately from their number.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode; raw stabilizer targets are not supported here")
    if representatives not in ("auto", "preserve", "canonical"):
        raise ValueError("representatives must be 'auto', 'preserve', or 'canonical'")
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Y", "Z"):
        raise ValueError("pauli must be X, Y, or Z")
    pauli = pauli.upper()
    if targets is not None and logical_paulis is not None:
        raise ValueError("Choose exactly one of targets and logical_paulis")
    if (targets is not None or logical_paulis is not None) and (
            logical_indices is not None or pauli != "Z"):
        raise ValueError("Do not combine targets/logical_paulis with pauli or logical_indices")

    # Validate/copy the original sparse checks; never replace them by a
    # row-reduced stabilizer basis just because the logical basis uses RREF.
    source = CSSCode(code.hx, code.hz, code.distance, code.name)
    basis = logical_basis_for(source, logical_basis)
    jx, jz = basis["jx"].copy(), basis["jz"].copy()
    check_x = np.vstack((source.hx, np.zeros_like(source.hz)))
    check_z = np.vstack((np.zeros_like(source.hx), source.hz))
    check_signs = np.zeros(len(check_x), dtype=np.uint8)

    if targets is not None:
        parsed = [parse_pauli(item, source.n) for item in _pauli_list(targets, "targets")]
        input_mode = "physical"
    else:
        if logical_paulis is not None:
            logical = [parse_pauli(item, source.k)
                       for item in _pauli_list(logical_paulis, "logical_paulis")]
            input_mode = "logical_paulis"
        else:
            logical = []
            for index in _indices(logical_indices, source.k):
                a = np.zeros(source.k, dtype=np.uint8)
                b = np.zeros(source.k, dtype=np.uint8)
                a[index] = pauli in ("X", "Y")
                b[index] = pauli in ("Z", "Y")
                logical.append((a, b, 0))
            input_mode = "logical_indices"
        parsed = [_logical_representative(a, b, sign, jx, jz) for a, b, sign in logical]

    original_x = np.stack([item[0] for item in parsed])
    original_z = np.stack([item[1] for item in parsed])
    original_signs = np.asarray([item[2] for item in parsed], dtype=np.uint8)
    if np.any(matmul(original_x, source.hz.T)) or np.any(matmul(original_z, source.hx.T)):
        raise ValueError("Every target must commute with all source stabilizers (normalizer)")
    signatures = np.hstack((matmul(original_x, jz.T), matmul(original_z, jx.T)))
    if np.any(~np.any(signatures, axis=1)):
        raise ValueError("Every target must be a nontrivial logical Pauli, not identity or a stabilizer")
    if np.any(commutation_matrix(original_x, original_z)):
        raise ValueError("Branching targets must commute pairwise")

    qwc_before = qubitwise_commuting(original_x, original_z)
    canonicalize = representatives == "canonical" or (representatives == "auto" and not qwc_before)
    x, z, signs = original_x.copy(), original_z.copy(), original_signs.copy()
    witnesses = np.zeros((len(x), len(check_x)), dtype=np.uint8)
    if canonicalize:
        x = matmul(signatures[:, :source.k], jx)
        z = matmul(signatures[:, source.k:], jz)
        difference = np.hstack((original_x ^ x, original_z ^ z))
        witnesses = solve(np.hstack((check_x, check_z)).T, difference.T).T
        # Binary equivalence is insufficient: multiplying commuting Paulis
        # can introduce a minus sign, even with all source-check signs +.
        for i, selected in enumerate(witnesses):
            product_x = np.vstack((original_x[i:i + 1], check_x))
            product_z = np.vstack((original_z[i:i + 1], check_z))
            product_signs = np.concatenate((original_signs[i:i + 1], check_signs))
            px, pz, phase = pauli_product(
                product_x, product_z, product_signs,
                [0, *(np.flatnonzero(selected) + 1).tolist()],
            )
            if phase % 2 or not np.array_equal(px, x[i]) or not np.array_equal(pz, z[i]):
                raise RuntimeError("Invalid signed stabilizer-equivalence witness")
            signs[i] = phase // 2

    if not qubitwise_commuting(x, z):
        if representatives == "preserve":
            raise ValueError("The supplied representatives are not qubit-wise commuting (QWC); "
                             "use representatives='auto' to try the same classes in the selected basis")
        raise ValueError("The same logical classes in the selected basis are not qubit-wise commuting "
                         "(QWC); arbitrary commuting targets are not covered by simultaneous branching. "
                         "Supply suitable representatives/basis or use a separately validated sequential protocol")

    return {
        "source_check_x": check_x,
        "source_check_z": check_z,
        "source_check_signs": check_signs,
        "source_jx_x": jx,
        "source_jx_z": np.zeros_like(jx),
        "source_jz_x": np.zeros_like(jz),
        "source_jz_z": jz,
        "source_jx_signs": np.zeros(source.k, dtype=np.uint8),
        "source_jz_signs": np.zeros(source.k, dtype=np.uint8),
        "target_x": x,
        "target_z": z,
        "target_signs": signs,
        "original_target_x": original_x,
        "original_target_z": original_z,
        "original_target_signs": original_signs,
        "logical_signatures": signatures,
        "canonicalization_witnesses": witnesses,
        "basis_source": basis["origin"],
        "representation_action": "canonicalized" if canonicalize else "preserved",
        "target_logical_rank": rank(signatures),
        "input_mode": input_mode,
        "qwc_before": qwc_before,
        "qwc_after": True,
    }
