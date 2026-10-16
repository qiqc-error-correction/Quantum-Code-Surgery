"""Zhang--Li naked, dressed, and finely devised LDPC glue constructions.

Reference: arXiv:2408.01339v3, Appendix G, Algorithms 1--5. Algorithm 5's
check-duplication loop is applied to dressing checks C_D, as required by
Eq. (109) and the stated degree bound; its printed C_N index would leave
the dense dressing rows unchanged. Original naked rows are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..gf2 import (binary_matrix, in_rowspace, inverse, matmul, nullspace,
                   rank, rref, row_basis, solve)
from ..models import CSSCode


@dataclass
class GlueCode:
    """Compatible classical glue code, with row-vector operator conventions."""

    h_g: np.ndarray
    s: np.ndarray
    t: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def H_G(self):
        return self.h_g

    @property
    def S(self):
        return self.s

    @property
    def T(self):
        return self.t


def _weight_max(matrix):
    """Maximum row or column weight, including empty check sets."""
    return max(int(np.max(matrix.sum(axis=0), initial=0)),
               int(np.max(matrix.sum(axis=1), initial=0)))


def _guard_size(rows, columns, memory_n, original_rows, max_entries, stage):
    entries = rows * columns + columns * memory_n + original_rows * rows
    if entries > max_entries:
        raise ValueError(
            f"{stage} exceeds max_entries: H_G, S and T require {entries} "
            "dense entries; increase the limit explicitly"
        )


def _information(h_g, s, opposite, conjugate):
    k = nullspace(h_g)
    q = matmul(k, s)
    syndrome = matmul(q, opposite.T)
    if np.any(syndrome):
        raise RuntimeError("Glue kernel maps outside the memory normalizer")
    m = matmul(q, conjugate.T)
    return k, q, m


def _coefficient_basis(target_signatures, logical_k):
    """Use the paper's [I_q | P] form and unit complementary coordinates."""
    reduced, pivots = rref(target_signatures)
    completed = reduced[:len(pivots)].copy()
    for coordinate in range(logical_k):
        if coordinate in pivots:
            continue
        unit = np.zeros(logical_k, dtype=np.uint8)
        unit[coordinate] = 1
        completed = np.vstack((completed, unit))
    if completed.shape != (logical_k, logical_k):
        raise RuntimeError("Could not complete the target logical coefficient basis")
    return completed


def build_devised_glue(code, targets, *, pauli="Z", ldpc=True,
                       max_entries=50_000_000):
    """Construct a finely devised glue code for independent pure logicals.

    ``targets`` is a binary row matrix of physical operators. Its logical
    signatures must be nonzero and independent. Physical representatives,
    including stabilizer factors, are preserved exactly. For Z measurements
    the compatibility relation is H_X S^T = T H_G; swap X/Z for X targets.

    With ``ldpc=True``, bit and check duplication make both row and column
    weights at most max(w_max(H_opposite) + 1, 3), while preserving the
    mapped glue-kernel space. The resource count for arbitrary supplied
    representatives is reported from the actual dressing matrix; the
    paper's optimized O((k_N-q)(q+1)) sparsity count is certified when the
    supplied physical representatives belong to the standard logical span.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
        raise ValueError("devised glue pauli must be X or Z")
    pauli = pauli.upper()
    if not isinstance(ldpc, (bool, np.bool_)):
        raise ValueError("ldpc must be a boolean")
    if (isinstance(max_entries, (bool, np.bool_))
            or not isinstance(max_entries, (int, np.integer))
            or max_entries < 1):
        raise ValueError("max_entries must be a positive integer")
    if np.asarray(targets).size > max_entries:
        raise ValueError("Target matrix exceeds max_entries")
    targets = binary_matrix(targets, n_cols=code.n, name="targets")
    if not len(targets):
        raise ValueError("At least one logical target is required")
    opposite, same = ((code.hx, code.hz) if pauli == "Z"
                      else (code.hz, code.hx))
    if np.any(matmul(targets, opposite.T)):
        raise ValueError("Every target must commute with the memory checks")
    if code.n * code.k > max_entries:
        raise ValueError("The complete logical basis exceeds max_entries")
    logicals = code.standard_logicals()
    conjugate = logicals.jx if pauli == "Z" else logicals.jz
    logical_generators = logicals.jz if pauli == "Z" else logicals.jx
    target_signatures = matmul(targets, conjugate.T)
    logical_count = rank(target_signatures)
    if logical_count != len(targets):
        raise ValueError("Target logical signatures must be nonzero and independent")

    # Algorithm 3: support-restricted checks, with selector pasting matrices.
    support = np.flatnonzero(np.any(targets, axis=0))
    n_n = len(support)
    incident = np.flatnonzero(np.any(opposite[:, support], axis=1))
    r_n = len(incident)
    _guard_size(r_n, n_n, code.n, len(opposite), max_entries, "Naked glue")
    h_n = opposite[np.ix_(incident, support)].copy()
    s_n = np.zeros((n_n, code.n), dtype=np.uint8)
    s_n[np.arange(n_n), support] = 1
    t_n = np.zeros((len(opposite), r_n), dtype=np.uint8)
    t_n[incident, np.arange(r_n)] = 1
    kernel_dimension = n_n - rank(h_n)
    if kernel_dimension * n_n > max_entries:
        raise ValueError("Naked glue kernel basis exceeds max_entries")
    k_before, q_before, m_before = _information(h_n, s_n, opposite, conjugate)
    k_n = rank(m_before)
    redundancy = k_n - logical_count
    if redundancy < 0:
        raise RuntimeError("Naked glue does not contain every requested logical")

    # Algorithm 4: retain all support-contained stabilizers (G0) and exact
    # target representatives (G1), and kill only the complementary classes.
    g0 = matmul(nullspace(m_before.T), k_before)
    coefficients = _coefficient_basis(target_signatures, code.k)
    active_coefficients = coefficients[:logical_count]
    active_basis_change = solve(target_signatures.T, active_coefficients.T).T
    active_targets = matmul(active_basis_change, targets)
    g1 = matmul(active_targets, s_n.T)
    paired_conjugates = matmul(inverse(coefficients).T, conjugate)
    active_conjugates = paired_conjugates[:logical_count]
    complementary_conjugates = paired_conjugates[logical_count:]
    complementary_logicals = matmul(coefficients[logical_count:],
                                    logical_generators)
    if not np.array_equal(matmul(active_conjugates, active_targets.T),
                          np.eye(logical_count, dtype=np.uint8)):
        raise RuntimeError("Target logical pairing failed")
    if np.any(matmul(complementary_conjugates, targets.T)):
        raise RuntimeError("Complementary logicals do not commute with targets")

    span = np.vstack((g0, g1))
    if rank(span) != len(g0) + len(g1):
        raise RuntimeError("Target and stabilizer kernel bases are not independent")
    active_components = matmul(q_before, active_conjugates.T)
    adjusted = k_before ^ matmul(active_components, g1)
    complement_rows = []
    for candidate in adjusted:
        if not in_rowspace(candidate, span):
            complement_rows.append(candidate.copy())
            span = np.vstack((span, candidate))
    g2 = (np.stack(complement_rows) if complement_rows else
          np.zeros((0, n_n), dtype=np.uint8))
    if len(g2) != redundancy or rank(span) != kernel_dimension:
        raise RuntimeError("Could not complete the restricted kernel basis")
    g2_image = matmul(g2, s_n)
    if np.any(matmul(g2_image, active_conjugates.T)):
        raise RuntimeError("Complement kernel has an active target component")
    u = matmul(g2_image, complementary_conjugates.T)
    if rank(u) != redundancy:
        raise RuntimeError("Complement logical coefficient matrix is not full row rank")
    # The last step of Appendix G chooses U=[I|Q] up to permutation. This
    # makes its right inverse select complementary conjugate rows, rather
    # than combining them into unnecessarily dense dressing checks.
    u_standard = row_basis(u)
    complement_basis_change = solve(u.T, u_standard.T).T
    g2 = matmul(complement_basis_change, g2)
    g2_image = matmul(g2, s_n)
    u = u_standard
    right_inverse = solve(u, np.eye(redundancy, dtype=np.uint8))
    d = matmul(matmul(right_inverse.T, complementary_conjugates), s_n.T)
    # These are the defining Algorithm 4 identities, independent of an
    # arbitrary choice of restricted-kernel or logical-complement basis.
    if np.any(matmul(d, g0.T)) or np.any(matmul(d, g1.T)):
        raise RuntimeError("Dressing removed a stabilizer or requested target")
    if not np.array_equal(matmul(d, g2.T),
                          np.eye(redundancy, dtype=np.uint8)):
        raise RuntimeError("Dressing did not isolate every redundant logical class")
    residual = g2_image ^ matmul(u, complementary_logicals)
    stabilizer_coefficients = solve(same.T, residual.T).T

    _guard_size(r_n + redundancy, n_n, code.n, len(opposite),
                max_entries, "Dressed glue")
    h_d = np.vstack((h_n, d))
    h_g = h_d.copy()
    s = s_n.copy()
    t = np.pad(t_n, ((0, 0), (0, redundancy)))
    operation_trace = []

    def duplicate_bit(bit, moved_checks):
        nonlocal h_g, s, t
        rows, columns = h_g.shape
        _guard_size(rows + 1, columns + 1, code.n, len(opposite),
                    max_entries, "Bit duplication")
        updated = np.pad(h_g, ((0, 1), (0, 1)))
        updated[moved_checks, bit] = 0
        updated[moved_checks, columns] = 1
        updated[rows, [bit, columns]] = 1
        h_g = updated
        s = np.pad(s, ((0, 1), (0, 0)))
        t = np.pad(t, ((0, 0), (0, 1)))
        operation_trace.append({
            "operation": "bit_duplication", "source_bit": int(bit),
            "moved_checks": [int(a) for a in moved_checks],
            "new_bit": columns, "new_check": rows,
            "new_bit_value": "copy of source_bit",
        })

    def duplicate_check(check, moved_bits):
        nonlocal h_g, s, t
        rows, columns = h_g.shape
        _guard_size(rows + 1, columns + 1, code.n, len(opposite),
                    max_entries, "Check duplication")
        updated = np.pad(h_g, ((0, 1), (0, 1)))
        updated[check, moved_bits] = 0
        updated[[check, rows], columns] = 1
        updated[rows, moved_bits] = 1
        h_g = updated
        s = np.pad(s, ((0, 1), (0, 0)))
        t = np.pad(t, ((0, 0), (0, 1)))
        operation_trace.append({
            "operation": "check_duplication", "source_check": int(check),
            "moved_bits": [int(u_bit) for u_bit in moved_bits],
            "new_bit": columns, "new_check": rows,
            "new_bit_value": "parity of moved_bits",
        })

    if ldpc:
        # Algorithm 5: only non-naked incidences are moved, so H_N stays
        # exactly in the leading block and the selector T remains valid.
        for bit in range(n_n):
            while True:
                extra = np.flatnonzero(h_g[r_n:, bit]) + r_n
                if len(extra) <= 1:
                    break
                duplicate_bit(bit, extra[:2])
        # C_D, rather than the printed C_N, is required by Eq. (109).
        for check in range(r_n, r_n + redundancy):
            while np.count_nonzero(h_g[check]) > 1:
                duplicate_check(check, np.flatnonzero(h_g[check])[:2])

    if not np.array_equal(h_g[:r_n, :n_n], h_n) or np.any(h_g[:r_n, n_n:]):
        raise RuntimeError("LDPC conversion changed a naked glue check")
    if not np.array_equal(matmul(opposite, s.T), matmul(t, h_g)):
        raise RuntimeError("Final glue pasting compatibility failed")
    k_after, q_after, m_after = _information(h_g, s, opposite, conjugate)
    desired = np.vstack((matmul(g0, s_n), targets))
    if (not in_rowspace(q_after, desired)
            or not in_rowspace(desired, q_after)):
        raise RuntimeError("Graph duplication changed the finely devised mapped kernel")
    if (rank(m_after) != logical_count
            or rank(np.vstack((m_after, target_signatures))) != logical_count):
        raise RuntimeError("Final glue measures logical classes outside the target space")
    final_bound = max(_weight_max(opposite) + 1, 3)
    final_weight = _weight_max(h_g)
    if ldpc and final_weight > final_bound:
        raise RuntimeError("Graph duplication did not attain the LDPC degree bound")
    if _weight_max(s) != 1 or _weight_max(t) > 1:
        raise RuntimeError("Pasting selectors lost their weight-one bound")
    sparse_budget = redundancy * (logical_count + 1)
    dressing_nonzeros = int(np.count_nonzero(d))
    canonical_representatives = in_rowspace(targets, logical_generators)
    optimized_size_bound = (
        canonical_representatives and dressing_nonzeros <= sparse_budget
        and h_g.shape[1] <= n_n + 2 * sparse_budget
        and h_g.shape[0] <= r_n + 2 * sparse_budget
    )
    if canonical_representatives and dressing_nonzeros > sparse_budget:
        raise RuntimeError("Standard-form dressing failed the paper's nonzero bound")
    metadata = {
        "construction": "Zhang--Li devised sticking, Appendix G Algorithms 3--5",
        "pauli": pauli, "ldpc": bool(ldpc), "targets": targets,
        "target_logical_signatures": target_signatures,
        "naked_support": support, "naked_check_indices": incident,
        "H_N": h_n, "S_N": s_n, "T_N": t_n,
        "K_before": k_before, "Q_before": q_before, "M_before": m_before,
        "G0": g0, "G1": g1, "G2": g2,
        "active_target_basis": active_targets,
        "active_basis_change": active_basis_change,
        "complement_basis_change": complement_basis_change,
        "logical_coefficient_basis": coefficients,
        "paired_conjugate_generators": paired_conjugates,
        "complementary_logical_generators": complementary_logicals,
        "U": u, "U_right_inverse": right_inverse,
        "V": stabilizer_coefficients, "D": d, "H_D": h_d,
        "K_after": k_after, "Q_after": q_after, "M_after": m_after,
        "operation_trace": operation_trace,
        "check_duplication_index_note": "C_D used according to Eq. (109); printed Algorithm 5 uses C_N",
        "n_N": n_n, "r_N": r_n, "k_N": k_n, "q": logical_count,
        "redundancy_number": redundancy,
        "n_G": h_g.shape[1], "r_G": h_g.shape[0],
        "added_bits": h_g.shape[1] - n_n,
        "added_checks": h_g.shape[0] - (r_n + redundancy),
        "dressing_nonzeros": dressing_nonzeros,
        "optimized_standard_form_dressing_nonzero_bound": sparse_budget,
        "dressing_satisfies_optimized_nonzero_bound": dressing_nonzeros <= sparse_budget,
        "canonical_standard_form_representatives": canonical_representatives,
        "optimized_standard_form_size_bound_claimed": bool(optimized_size_bound),
        "resource_count_note": "Optimized Appendix G bounds are certified for canonical standard-form representatives; otherwise actual counts are reported",
        "max_row_weight": int(np.max(h_g.sum(axis=1), initial=0)),
        "max_column_weight": int(np.max(h_g.sum(axis=0), initial=0)),
        "w_max": final_weight, "ldpc_weight_bound": final_bound,
        "ldpc_bound_satisfied": final_weight <= final_bound,
        "pasting_S_weight_max": _weight_max(s),
        "pasting_T_weight_max": _weight_max(t),
        "pasting_S_hamming_norm": 1,
        "logical_signature_rank_before": k_n,
        "logical_signature_rank_after": rank(m_after),
        "mapped_kernel_preserved_by_duplication": True,
        "finely_devised": True,
    }
    return GlueCode(h_g, s, t, metadata)
