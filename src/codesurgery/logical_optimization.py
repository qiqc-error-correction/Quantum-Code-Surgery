"""Reduce the physical weight of one fixed logical Pauli modulo stabilizers.

The search is restricted to the affine coset of the supplied operator. It is
not a code-distance search. Mixed Paulis use ``wt(x OR z)``; multiplying CSS
checks may change the literal Pauli sign, which is retained exactly.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from .gf2 import binary_vector
from .gadget_common import prepare, signature
from .symplectic import pauli_product


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _options(method, threshold, max_trials, seed, time_limit):
    if method not in ("auto", "heuristic", "milp"):
        raise ValueError("method must be auto, heuristic, or milp")
    if isinstance(threshold, str):
        if threshold != "2d":
            raise ValueError("threshold must be '2d', a nonnegative integer, or None")
    elif threshold is not None:
        threshold = _integer(threshold, "threshold")
    max_trials, seed = _integer(max_trials, "max_trials"), _integer(seed, "seed")
    if time_limit is not None:
        if (isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, Real)
                or not np.isfinite(time_limit) or time_limit < 0):
            raise ValueError("time_limit must be finite and nonnegative, or None")
        time_limit = float(time_limit)
    return threshold, max_trials, seed, time_limit


def _scipy():
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise ImportError("MILP logical-weight reduction needs scipy>=1.9; install "
                          "codesurgery[distance] or select method='heuristic'") from error
    return Bounds, LinearConstraint, milp, coo_matrix


def _bits(row):
    return sum(1 << int(q) for q in np.flatnonzero(row))


def _vector(bits, size):
    return np.fromiter(((bits >> q) & 1 for q in range(size)), dtype=np.uint8, count=size)


def _rref(rows, order):
    """Eliminate packed rows, retaining original-generator coefficient masks."""
    rows = list(rows)
    pivot = 0
    pivots = []
    for column in order:
        bit = 1 << int(column)
        found = next((j for j in range(pivot, len(rows)) if rows[j][0] & bit), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        value, witness = rows[pivot]
        for j in range(len(rows)):
            if j != pivot and rows[j][0] & bit:
                rows[j] = (rows[j][0] ^ value, rows[j][1] ^ witness)
        pivots.append(int(column))
        pivot += 1
        if pivot == len(rows):
            break
    return rows[:pivot], pivots


def _greedy(value, witness, moves, weight):
    """Strict descent with single checks and pairs; no local-optimality claim."""
    current = weight(value)
    while True:
        best, change = current, None
        for move, coeff in moves:
            candidate = weight(value ^ move)
            if candidate < best:
                best, change = candidate, (move, coeff)
        if change is None:
            for i, (left, a) in enumerate(moves):
                for right, b in moves[i + 1:]:
                    candidate = weight(value ^ left ^ right)
                    if candidate < best:
                        best, change = candidate, (left ^ right, a ^ b)
        if change is None:
            return value, witness
        value ^= change[0]
        witness ^= change[1]
        current = best


def _heuristic(original, generators, independent, width, weight, trials, seed, lower):
    best, coefficients = original, 0
    stats = dict(trials_requested=trials, trials_completed=0, seed=seed,
                 guarantee="validated upper bound only", improvements=0)
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        if trial == 0:
            candidate, witness, moves = original, 0, generators
        else:
            rows, pivots = _rref(independent, rng.permutation(width))
            candidate, witness = original, 0
            for (row, coefficient), pivot in zip(rows, pivots):
                if (candidate >> pivot) & 1:
                    candidate ^= row
                    witness ^= coefficient
            moves = generators + rows
        candidate, witness = _greedy(candidate, witness, moves, weight)
        if weight(candidate) < weight(best):
            best, coefficients = candidate, witness
            stats["improvements"] += 1
        stats["trials_completed"] += 1
        if weight(best) <= lower:
            break
    return best, coefficients, stats


def _exhaustive(original, independent, weight, best, coefficients):
    value, witness, previous = original, 0, 0
    count = 1 << len(independent)
    for index in range(1, count):
        gray = index ^ (index >> 1)
        changed = (gray ^ previous).bit_length() - 1
        value ^= independent[changed][0]
        witness ^= independent[changed][1]
        previous = gray
        if weight(value) < weight(best):
            best, coefficients = value, witness
    return best, coefficients, count


def _milp(original, independent, n, pure, best, coefficients, weight, time_limit):
    Bounds, LinearConstraint, milp, coo_matrix = _scipy()
    width = n if pure else 2 * n
    rank = len(independent)
    # Binary stabilizer coefficients and physical bits, then integer parity
    # slack. Mixed targets additionally have one binary union-support bit/qubit.
    bit_start, parity_start = rank, rank + width
    t_start = parity_start + width
    size = t_start if pure else t_start + n
    lower, upper = np.zeros(size), np.ones(size)
    objective = np.zeros(size)
    objective[bit_start:bit_start + n] = 1 if pure else 0
    if not pure:
        objective[t_start:] = 1
    row_idx, col_idx, data, low, high = [], [], [], [], []

    def constraint(terms, lb, ub):
        row = len(low)
        for column, coefficient in terms:
            row_idx.append(row)
            col_idx.append(column)
            data.append(coefficient)
        low.append(lb)
        high.append(ub)

    for q in range(width):
        terms = [(j, 1) for j, (row, _) in enumerate(independent) if (row >> q) & 1]
        bit = (original >> q) & 1
        upper[parity_start + q] = (len(terms) + bit) // 2
        constraint(terms + [(bit_start + q, -1), (parity_start + q, -2)], -bit, -bit)
    if not pure:
        for q in range(n):
            constraint([(t_start + q, 1), (bit_start + q, -1)], 0, np.inf)
            constraint([(t_start + q, 1), (bit_start + n + q, -1)], 0, np.inf)
            constraint([(t_start + q, 1), (bit_start + q, -1),
                        (bit_start + n + q, -1)], -np.inf, 0)
    constraint([(j, 1) for j in np.flatnonzero(objective)], -np.inf, weight(best))
    matrix = coo_matrix((data, (row_idx, col_idx)), shape=(len(low), size), dtype=float).tocsc()
    low, high = np.asarray(low), np.asarray(high)
    info = dict(solver="scipy.optimize.milp/HiGHS", status=None, proof=None,
                witness_validated=False, time_limit=time_limit,
                lower_bound_policy="optimal status and validated integral witness only")
    options = dict(mip_rel_gap=0.0, presolve=True)
    if time_limit is not None:
        options["time_limit"] = time_limit
    try:
        result = milp(c=objective, integrality=np.ones(size, dtype=np.uint8),
                      bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(matrix, low, high), options=options)
    except Exception as error:
        info["message"] = f"{type(error).__name__}: {error}"
        return best, coefficients, False, info
    status = getattr(result, "status", None)
    info["status"] = int(status) if isinstance(status, (int, np.integer)) else None
    info["message"] = str(getattr(result, "message", ""))
    for name in ("mip_gap", "mip_dual_bound", "mip_node_count"):
        value = getattr(result, name, None)
        if isinstance(value, Real) and np.isfinite(value):
            info[name] = int(value) if name == "mip_node_count" else float(value)
    candidate = getattr(result, "x", None)
    valid, proposed, witness = False, original, 0
    if candidate is not None:
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape == (size,) and np.all(np.isfinite(candidate)):
            rounded = np.rint(candidate)
            action = matrix @ rounded
            valid = bool(np.all(np.abs(candidate - rounded) <= 1e-6)
                         and np.all(rounded >= lower) and np.all(rounded <= upper)
                         and np.all(action >= low) and np.all(action <= high))
            if valid:
                for j, (row, coefficient) in enumerate(independent):
                    if rounded[j]:
                        proposed ^= row
                        witness ^= coefficient
                valid = (proposed == _bits(rounded[bit_start:bit_start + width])
                         and weight(proposed) <= weight(best)
                         and int(np.dot(objective, rounded)) == weight(proposed))
    info["witness_validated"] = bool(valid)
    if valid and weight(proposed) < weight(best):
        best, coefficients = proposed, witness
    fun = getattr(result, "fun", None)
    proved = bool(status == 0 and valid and isinstance(fun, Real) and np.isfinite(fun)
                  and abs(float(fun) - weight(proposed)) <= 1e-6
                  and info.get("mip_gap", 0.0) <= 1e-9)
    if proved:
        info["proof"] = "milp_optimal_affine_stabilizer_coset"
    return best, coefficients, proved, info


def verify_logical_optimization(code, optimized_target, basis, report):
    """Recheck coefficient witness, exact phase, fixed signature and weights.

    This verifier does not trust cached Boolean assertions in ``report``.
    Stabilizers are the original positive CSS generators, including redundant
    rows. The return value is True; invalid or tampered reports raise ValueError.
    """
    if not isinstance(report, dict):
        raise ValueError("logical optimization report must be a dict")
    try:
        _, original, _ = prepare(code, dict(x=report["original_x"], z=report["original_z"],
                                            sign=report["original_sign"]), logical_basis=basis)
        _, optimized, _ = prepare(code, dict(x=report["x"], z=report["z"], sign=report["sign"]),
                                  logical_basis=basis)
        a = binary_vector(report["hx_coefficients"], length=len(code.hx), name="hx_coefficients")
        b = binary_vector(report["hz_coefficients"], length=len(code.hz), name="hz_coefficients")
        xs = np.vstack([original[0], code.hx, np.zeros_like(code.hz)])
        zs = np.vstack([original[1], np.zeros_like(code.hx), code.hz])
        signs = np.zeros(len(xs), dtype=np.uint8)
        signs[0] = original[2]
        selected = np.flatnonzero(np.concatenate(([1], a, b)))
        x, z, phase = pauli_product(xs, zs, signs, selected=selected)
        before, after = signature(*original[:2], basis), signature(*optimized[:2], basis)
        expected = (x, z, phase // 2)
        if phase % 2 or any(not np.array_equal(a, b) for a, b in zip(expected, optimized)):
            raise ValueError("logical optimization coefficient witness or Pauli sign is invalid")
        if any(not np.array_equal(a, b) for a, b in zip(optimized, optimized_target)):
            raise ValueError("logical optimization report does not match the gadget target")
        if (not np.array_equal(before, after)
                or not np.array_equal(before, report["original_signature"])
                or not np.array_equal(after, report["optimized_signature"])):
            raise ValueError("logical optimization changed or misstated the logical signature")
        original_weight = int(np.count_nonzero(original[0] | original[1]))
        optimized_weight = int(np.count_nonzero(optimized[0] | optimized[1]))
        if (report["before_weight"] != original_weight or report["after_weight"] != optimized_weight
                or optimized_weight > original_weight):
            raise ValueError("logical optimization weight report is invalid")
        if not np.any(original[0]) and np.any(a):
            raise ValueError("pure-Z optimization may only multiply Z stabilizers")
        if not np.any(original[1]) and np.any(b):
            raise ValueError("pure-X optimization may only multiply X stabilizers")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Malformed logical optimization report") from error
    return True


def optimize_prepared_logical(source, target, basis, *, method="auto", threshold="2d",
                              max_trials=128, seed=0, time_limit=10.0):
    """Internal prepared-input entry point; return ``(signed_target, report)``."""
    threshold, max_trials, seed, time_limit = _options(method, threshold, max_trials, seed, time_limit)
    threshold_rule = "2d" if threshold == "2d" else "always" if threshold is None else "explicit"
    threshold_distance = int(source.distance) if threshold_rule == "2d" and source.distance is not None else None
    unknown_distance = threshold_rule == "2d" and threshold_distance is None
    if threshold_rule == "2d":
        threshold = None if unknown_distance else 2 * threshold_distance
    if method == "milp":
        _scipy()  # Explicit selection fails clearly even when a later shortcut applies.
    x, z, sign = target
    n, rx, rz = source.n, len(source.hx), len(source.hz)
    pure = not np.any(x) or not np.any(z)
    kind = "Z" if not np.any(x) else "X" if not np.any(z) else "mixed"
    mask = (1 << n) - 1
    if kind == "Z":
        original = _bits(z)
        generators = [(_bits(row), 1 << (rx + j)) for j, row in enumerate(source.hz)]
    elif kind == "X":
        original = _bits(x)
        generators = [(_bits(row), 1 << j) for j, row in enumerate(source.hx)]
    else:
        original = _bits(x) | (_bits(z) << n)
        generators = ([(_bits(row), 1 << j) for j, row in enumerate(source.hx)]
                      + [(_bits(row) << n, 1 << (rx + j)) for j, row in enumerate(source.hz)])
    generators = [(row, coefficient) for row, coefficient in generators if row]
    width = n if pure else 2 * n
    weight = int.bit_count if pure else lambda row: ((row & mask) | (row >> n)).bit_count()
    independent, _ = _rref(generators, range(width))
    best, coefficients = original, 0
    lower = int(source.distance) if source.distance is not None else 1
    lower_source = "declared_memory_distance" if source.distance is not None else "nontrivial_logical_weight_at_least_one"
    skipped = unknown_distance or (threshold is not None and weight(original) <= threshold)
    report = dict(method=method, objective="Pauli support weight: count(x OR z)",
                  coset="original Pauli times original positive CSS stabilizers",
                  search_kind=kind, threshold=threshold, threshold_rule=threshold_rule,
                  threshold_distance=threshold_distance, heuristic=None, milp=None,
                  exhaustive=None, basis_origin=basis.get("origin"), optimal=False,
                  status="skipped_unknown_distance" if unknown_distance else
                         "skipped_threshold" if skipped else "bounded")
    if not skipped and weight(best) > lower:
        if method != "milp":
            best, coefficients, report["heuristic"] = _heuristic(
                original, generators, independent, width, weight, max_trials, seed, lower)
        if weight(best) < lower:
            raise ValueError("Reduced logical weight contradicts the declared memory-code distance")
        if method == "auto" and weight(best) > lower and len(independent) <= 18:
            best, coefficients, visited = _exhaustive(original, independent, weight, best, coefficients)
            report["exhaustive"] = dict(stabilizer_rank=len(independent), representatives_checked=visited)
            lower, lower_source = weight(best), "exhaustive_affine_stabilizer_coset"
        elif method in ("auto", "milp") and weight(best) > lower:
            try:
                best, coefficients, proved, report["milp"] = _milp(
                    original, independent, n, pure, best, coefficients, weight, time_limit)
                if proved:
                    lower, lower_source = weight(best), "milp_optimal_affine_stabilizer_coset"
            except ImportError:
                if method == "milp":
                    raise
                report["milp"] = dict(status="unavailable", proof=None,
                                      message="Optional SciPy backend unavailable; retained heuristic incumbent")
    if source.distance is not None and weight(best) < source.distance:
        raise ValueError("Reduced logical weight contradicts the declared memory-code distance")
    a, b = _vector(coefficients, rx), _vector(coefficients >> rx, rz)
    xs = np.vstack([x, source.hx, np.zeros_like(source.hz)])
    zs = np.vstack([z, np.zeros_like(source.hx), source.hz])
    signs = np.zeros(len(xs), dtype=np.uint8)
    signs[0] = sign
    out_x, out_z, phase = pauli_product(xs, zs, signs,
                                      selected=np.flatnonzero(np.concatenate(([1], a, b))))
    if phase % 2:
        raise RuntimeError("Commuting logical/stabilizer product unexpectedly has imaginary phase")
    result = (out_x, out_z, phase // 2)
    reconstructed = _bits(out_z) if kind == "Z" else _bits(out_x) if kind == "X" else _bits(out_x) | (_bits(out_z) << n)
    if reconstructed != best:
        raise RuntimeError("Logical optimization coefficient witness failed reconstruction")
    report.update(x=out_x.tolist(), z=out_z.tolist(), sign=int(result[2]),
                  original_x=x.tolist(), original_z=z.tolist(), original_sign=int(sign),
                  before_weight=weight(original), after_weight=weight(best),
                  hx_coefficients=a.tolist(), hz_coefficients=b.tolist(),
                  original_signature=signature(x, z, basis).tolist(),
                  optimized_signature=signature(out_x, out_z, basis).tolist(),
                  optimal=bool(lower == weight(best)), proven_lower_bound=int(lower),
                  upper_bound=weight(best), lower_bound_source=lower_source,
                  equivalence_verified=True, signature_verified=True,
                  sign_convention="(-1)^sign times literal tensor I/X/Y/Z")
    if not skipped:
        report["status"] = "optimal" if report["optimal"] else "improved" if weight(best) < weight(original) else "bounded"
    verify_logical_optimization(source, result, basis, report)
    return result, report


def reduce_logical_weight(code, logical=None, *, pauli="Z", logical_index=0,
                          logical_basis=None, method="auto", threshold="2d",
                          max_trials=128, seed=0, time_limit=10.0):
    """Find a no-heavier representative of a specified logical observable.

    The default ``threshold='2d'`` attempts reduction only when the input
    weight is strictly greater than twice ``code.distance``; unknown distance
    skips this automatic rule and records ``skipped_unknown_distance``.
    An integer threshold overrides the rule, and None always attempts
    reduction. ``heuristic`` supplies an upper bound,
    ``auto`` also exhausts small cosets or tries optional SciPy/HiGHS MILP, and
    ``milp`` requires SciPy. ``time_limit`` bounds the MILP stage; ``max_trials``
    bounds deterministic seeded heuristic restarts. Optimality may also follow
    conditionally from the code's declared memory distance, explicitly recorded
    in ``lower_bound_source``. No output is allowed to exceed its input weight.

    Return a JSON-serializable report containing physical x/z/sign vectors and
    original-generator coefficients. These certify equality on the original
    code space, including a possible change in literal Pauli sign for Y/XZ.
    """
    _options(method, threshold, max_trials, seed, time_limit)
    source, target, basis = prepare(code, logical, pauli=pauli, logical_index=logical_index,
                                    logical_basis=logical_basis)
    return optimize_prepared_logical(source, target, basis, method=method, threshold=threshold,
                                     max_trials=max_trials, seed=seed, time_limit=time_limit)[1]


__all__ = ["reduce_logical_weight", "verify_logical_optimization"]
