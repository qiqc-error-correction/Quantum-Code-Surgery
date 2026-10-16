"""Check and augment the parity expansion needed by the measurement gadget.

Columns are vertices (target-support data qubits), and rows are binary
even-cardinality hyperedges (ancilla qubits).  For a restriction matrix ``F``
we use ``h(F) = min wt(F v) / wt(v)``, over nonempty ``v`` with
``wt(v) <= floor(w / 2)``.  This is a parity boundary, including when rows
have weight greater than two; ordinary hypergraph boundary counting would
give the wrong certificate.

Small supports are exhaustively checked.  Larger supports can be certified
by adjoining a star, which uses only ``w - 1`` rows but has an unbounded
degree center.  Consequently this module certifies ``h >= 1`` and reports
degrees, but does not claim that its augmentation is an LDPC construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..gf2 import binary_matrix


_BYTE_POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_limit(exact_limit: object) -> int:
    value = _positive_integer(exact_limit, "exact_limit")
    if value > 24:
        raise ValueError("exact_limit must be at most 24; exact cut enumeration is exponential")
    return value


def _validated_matrix(value: object, *, name: str, max_entries: int,
                      n_cols: int | None = None) -> np.ndarray:
    # Check the allocation budget before binary_matrix creates its owned copy.
    array = np.asarray(value)
    if array.size > max_entries:
        raise ValueError(f"{name} exceeds max_entries={max_entries}")
    matrix = binary_matrix(value, n_cols=n_cols, name=name)
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one support column")
    if np.any(matrix.sum(axis=1, dtype=np.uint64) & 1):
        raise ValueError(f"every row of {name} must have even weight (F @ 1 = 0)")
    return matrix


def _vacuous_report(w: int) -> dict[str, Any]:
    return {
        "support_size": w,
        "h": None,
        "h_lower_bound": 1.0,
        "exact": True,
        "vacuous": True,
        "distance_condition_met": True,
        "certificate": "no nonempty subset of size at most floor(w/2)",
        "worst_cut": [],
        "worst_cut_vector": [0] * w,
        "boundary_weight": 0,
        "cut_size": 0,
        "eligible_cut_count": 0,
    }


def _unchecked_report(w: int) -> dict[str, Any]:
    return {
        "support_size": w,
        "h": None,
        "h_lower_bound": 0.0,
        "exact": False,
        "vacuous": False,
        "distance_condition_met": False,
        "certificate": "not checked: support exceeds exact enumeration limit",
        "worst_cut": None,
        "worst_cut_vector": None,
        "boundary_weight": None,
        "cut_size": None,
        "eligible_cut_count": None,
    }


def _enumerate_boundaries(matrix: np.ndarray, *, max_entries: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eligible subset masks, sizes and parity-boundary weights.

    Work is vectorized in chunks of 64 check rows.  All check rows, including
    redundant rows, count separately because they represent different gadget
    ancilla qubits.  No row-space reduction is permitted here.
    """
    n_rows, w = matrix.shape
    n_masks = 1 << w
    # The largest temporary is the byte-popcount lookup (8 entries per mask),
    # with syndromes, masks, sizes and counts also present.  This conservative
    # budget applies to work arrays as well as the returned dense matrix.
    if 12 * n_masks > max_entries:
        raise ValueError(
            "exact Cheeger enumeration exceeds max_entries; lower exact_limit "
            "to use certified star augmentation, or increase max_entries"
        )
    masks = np.arange(n_masks, dtype=np.uint32)
    sizes = np.zeros(n_masks, dtype=np.uint8)
    for col in range(w):
        step = 1 << col
        sizes[step: 2 * step] = sizes[:step] + 1
    counts = np.zeros(n_masks, dtype=np.uint32)
    for start in range(0, n_rows, 64):
        rows = matrix[start: start + 64]
        syndromes = np.zeros(n_masks, dtype=np.uint64)
        for col in range(w):
            bits = 0
            for row in np.flatnonzero(rows[:, col]):
                bits |= 1 << int(row)
            step = 1 << col
            syndromes[step: 2 * step] = syndromes[:step] ^ np.uint64(bits)
        counts += _BYTE_POPCOUNT[syndromes.view(np.uint8).reshape(-1, 8)].sum(
            axis=1, dtype=np.uint32
        )
    eligible = (sizes > 0) & (sizes <= w // 2)
    return masks[eligible], sizes[eligible], counts[eligible]


def _report_from_counts(w: int, masks: np.ndarray, sizes: np.ndarray,
                        counts: np.ndarray) -> dict[str, Any]:
    """Choose a worst cut using exact integer ratio comparisons."""
    best_mask = 0
    best_boundary = 0
    best_size = 1
    first = True
    for size in range(1, w // 2 + 1):
        positions = np.flatnonzero(sizes == size)
        boundary = int(counts[positions].min())
        candidates = positions[counts[positions] == boundary]
        mask = int(masks[int(candidates[0])])
        comparison = boundary * best_size - best_boundary * size
        if first or comparison < 0 or (comparison == 0 and mask < best_mask):
            first = False
            best_mask, best_boundary, best_size = mask, boundary, size
    cut = [col for col in range(w) if best_mask & (1 << col)]
    h = best_boundary / best_size
    return {
        "support_size": w,
        "h": h,
        "h_lower_bound": h,
        "exact": True,
        "vacuous": False,
        "distance_condition_met": best_boundary >= best_size,
        "certificate": "exhaustive parity-boundary enumeration",
        "worst_cut": cut,
        "worst_cut_vector": [int(col in cut) for col in range(w)],
        "boundary_weight": best_boundary,
        "cut_size": best_size,
        "eligible_cut_count": int(len(masks)),
    }


def exact_cheeger(F: object, *, exact_limit: int = 20,
                  max_entries: int = 50_000_000) -> dict[str, Any]:
    """Return an exact, JSON-compatible report for the parity Cheeger constant.

    ``worst_cut`` uses zero-based support-column indices.  ``h=None`` for the
    vacuous single-column case represents an empty minimization (infinite
    expansion), rather than putting nonstandard JSON Infinity in a report.
    Above ``exact_limit`` this function raises; it never substitutes sampling
    for an exact certificate.
    """
    budget = _positive_integer(max_entries, "max_entries")
    limit = _validate_limit(exact_limit)
    matrix = _validated_matrix(F, name="F", max_entries=budget)
    w = matrix.shape[1]
    if w == 1:
        return _vacuous_report(w)
    if w > limit:
        raise ValueError(f"support size {w} exceeds exact_limit={limit}")
    masks, sizes, counts = _enumerate_boundaries(matrix, max_entries=budget)
    return _report_from_counts(w, masks, sizes, counts)


def _degree_report(matrix: np.ndarray) -> dict[str, int]:
    row_weights = matrix.sum(axis=1, dtype=np.uint64)
    col_weights = matrix.sum(axis=0, dtype=np.uint64)
    return {
        "max_row_weight": int(row_weights.max()) if row_weights.size else 0,
        "max_column_weight": int(col_weights.max()) if col_weights.size else 0,
        "n_rows": int(matrix.shape[0]),
        "n_columns": int(matrix.shape[1]),
    }


def expand_restriction(F: object, *, mode: str = "auto", extra_rows: object = None,
                       exact_limit: int = 20, max_entries: int = 50_000_000
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Adjoin even-weight rows, preserving all original restriction rows.

    Modes:

    * ``none``: only append caller-supplied ``extra_rows``; enumerate if small.
    * ``auto``: repair exact worst cuts with unused weight-two edges when
      small; above the exact limit append a certified star.
    * ``star``: append every edge of a star on the support vertices.
    * ``complete``: append every edge of the complete graph.

    A star gives ``h >= 1`` with ``w-1`` added rows.  A complete graph gives
    ``h >= ceil(w/2)`` with ``w(w-1)/2`` added rows.  These are certified lower
    bounds for the augmented matrix because every preexisting row contributes
    a nonnegative boundary.  Neither construction bounds the column degree.

    Supplementary rows are independent ancilla rows, so duplicates are valid
    and are retained.  ``added_edges`` contains only automatically generated
    edges; ``user_added_rows`` reports the number supplied by the caller.
    """
    budget = _positive_integer(max_entries, "max_entries")
    limit = _validate_limit(exact_limit)
    if mode not in {"none", "auto", "star", "complete"}:
        raise ValueError("mode must be 'none', 'auto', 'star', or 'complete'")
    base = _validated_matrix(F, name="F", max_entries=budget)
    n_base, w = base.shape
    user = np.zeros((0, w), dtype=np.uint8) if extra_rows is None else _validated_matrix(
        extra_rows, name="extra_rows", n_cols=w, max_entries=budget
    )
    n_initial = n_base + len(user)
    if n_initial * w > budget:
        raise ValueError("F plus extra_rows exceeds max_entries")
    matrix = np.concatenate((base, user), axis=0)
    enumerable = w <= limit and (w == 1 or 12 * (1 << w) <= budget)
    masks = sizes = counts = None
    if w == 1:
        initial = _vacuous_report(w)
    elif enumerable:
        masks, sizes, counts = _enumerate_boundaries(matrix, max_entries=budget)
        initial = _report_from_counts(w, masks, sizes, counts)
    else:
        initial = _unchecked_report(w)
        if w <= limit:
            initial["certificate"] = "not checked: exact enumeration exceeds max_entries"

    edges: list[tuple[int, int]] = []
    method = mode
    final = initial.copy()
    if mode == "auto" and not enumerable:
        method = "star"

    if method == "auto" and not initial["distance_condition_met"]:
        assert masks is not None and sizes is not None and counts is not None
        degree = matrix.sum(axis=0, dtype=np.uint64)
        existing_edges: set[tuple[int, int]] = set()
        for row in matrix:
            endpoints = np.flatnonzero(row)
            if len(endpoints) == 2:
                existing_edges.add((int(endpoints[0]), int(endpoints[1])))
        while not final["distance_condition_met"]:
            inside = set(final["worst_cut"])
            # A bad cut cannot already contain every crossing simple edge:
            # the complete graph alone has at least |S| crossing edges.
            # Consequently unused-edge repair terminates after at most C(w,2)
            # additions and never needs parallel generated edges.
            candidates = [
                (min(u, v), max(u, v))
                for u in inside for v in range(w) if v not in inside
                and (min(u, v), max(u, v)) not in existing_edges
            ]
            if not candidates:  # Should be impossible, retained as an invariant check.
                raise RuntimeError("bad cut has no unused crossing edge")
            edge = min(candidates, key=lambda uv: (
                max(int(degree[uv[0]]), int(degree[uv[1]])),
                int(degree[uv[0]]) + int(degree[uv[1]]), uv[0], uv[1]
            ))
            if (n_initial + len(edges) + 1) * w > budget:
                raise ValueError("expansion augmentation exceeds max_entries")
            edges.append(edge)
            existing_edges.add(edge)
            degree[list(edge)] += 1
            crossing = ((masks >> edge[0]) ^ (masks >> edge[1])) & 1
            counts += crossing.astype(np.uint32)
            final = _report_from_counts(w, masks, sizes, counts)

    elif method in {"star", "complete"} and w > 1:
        n_edges = w - 1 if method == "star" else w * (w - 1) // 2
        if (n_initial + n_edges) * w > budget:
            raise ValueError(f"{method} augmentation exceeds max_entries")
        if method == "star":
            # Lower initial-degree center reduces the final maximum degree.
            center = int(np.argmin(matrix.sum(axis=0, dtype=np.uint64)))
            edges = [(min(center, v), max(center, v)) for v in range(w) if v != center]
        else:
            edges = [(u, v) for u in range(w) for v in range(u + 1, w)]
        if enumerable:
            assert masks is not None and sizes is not None and counts is not None
            for u, v in edges:
                counts += (((masks >> u) ^ (masks >> v)) & 1).astype(np.uint32)
            final = _report_from_counts(w, masks, sizes, counts)
        else:
            final = initial.copy()
            final.update({
                "h_lower_bound": 1.0 if method == "star" else float((w + 1) // 2),
                "distance_condition_met": True,
                "certificate": "star subgraph" if method == "star" else "complete graph subgraph",
            })

    if edges:
        supplement = np.zeros((len(edges), w), dtype=np.uint8)
        for row, (u, v) in enumerate(edges):
            supplement[row, [u, v]] = 1
        matrix = np.concatenate((matrix, supplement), axis=0)
    metadata = {
        "mode": mode,
        "augmentation_method": method,
        "exact_limit": limit,
        "h": final["h"],
        "h_lower_bound": final["h_lower_bound"],
        "distance_condition_met": final["distance_condition_met"],
        "exact": final["exact"],
        "certificate": final["certificate"],
        "base_rows": n_base,
        "user_added_rows": int(len(user)),
        "augmentation_rows": len(edges),
        "added_edges": [list(edge) for edge in edges],
        "initial_cheeger": initial,
        "final_cheeger": final,
        **_degree_report(matrix),
        "ldpc_degree_guarantee": False,
    }
    return matrix, metadata


__all__ = ["exact_cheeger", "expand_restriction"]
