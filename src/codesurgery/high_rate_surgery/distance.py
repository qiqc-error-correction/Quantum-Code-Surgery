"""Protected CSS subsystem identification and bounded dressed-distance search.

The subsystem splitting is explicit: gauge X generators have zero signature
against the original logical Z basis.  A merged stabilizer matrix alone does
not specify this splitting.  Distances here concern the protected subsystem,
not the classical measurement record or a circuit/spacetime distance.

Enumeration produces exact finite combinatorial certificates.  Gurobi bounds
are floating-point MIP certificates under the solver's tolerances; every
reported upper-bound witness is independently checked over GF(2).
"""

from __future__ import annotations

from itertools import combinations, product
from math import ceil
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from ..gf2 import binary_matrix, in_rowspace, inverse, nullspace, rank, row_basis, rref, solve


def _mm(a, b):
    return (a @ b) & 1


def _complement(candidates, base):
    """Select original candidate rows independent modulo the supplied base."""
    joined = np.vstack((base, candidates))
    _, pivots = rref(joined.T)
    indices = [i - len(base) for i in pivots if i >= len(base)]
    return candidates[indices].copy()


def _same_space(a, b):
    return rank(a) == rank(b) and in_rowspace(a, b)


def build_css_protected_basis(hx, hz, jx, jz, merged_hx, merged_hz, B, R,
                              target_signatures):
    """Identify the surviving data subsystem of a pure-X high-rate surgery.

    ``B`` is ancilla-qubits by attachment-checks, and ``R`` is data-qubits
    by attachment-checks. ``target_signatures`` contains pure-X coefficients
    in the supplied canonical ``jx,jz`` basis. All returned Pauli matrices
    contain only their pure X or Z component, with columns (data, ancilla).

    The pure-Z case is obtained by exchanging X and Z at the call boundary.
    Redundant data checks and ancilla gauge operators eliminated by data
    metachecks are handled by taking quotients in the actual merged code.
    """
    hx = binary_matrix(hx, name="hx")
    n = hx.shape[1]
    hz = binary_matrix(hz, n_cols=n, name="hz")
    jx = binary_matrix(jx, n_cols=n, name="jx")
    jz = binary_matrix(jz, n_cols=n, name="jz")
    mx = binary_matrix(merged_hx, name="merged_hx")
    N = mx.shape[1]
    mz = binary_matrix(merged_hz, n_cols=N, name="merged_hz")
    if N < n:
        raise ValueError("merged code cannot have fewer columns than the data code")
    b = binary_matrix(B, name="B")
    if b.shape[0] != N - n:
        raise ValueError("B must have one row per added ancilla qubit")
    r = binary_matrix(R, n_cols=b.shape[1], name="R")
    if r.shape[0] != n:
        raise ValueError("R must have one row per data qubit")
    k = n - rank(hx) - rank(hz)
    if (np.any(_mm(hx, hz.T)) or len(jx) != k or len(jz) != k
            or np.any(_mm(hz, jx.T)) or np.any(_mm(hx, jz.T))
            or not np.array_equal(_mm(jx, jz.T), np.eye(k, dtype=np.uint8))):
        raise ValueError("source checks and canonical logical bases are inconsistent")
    if np.any(_mm(mx, mz.T)):
        raise ValueError("merged checks do not commute")
    expected_x = np.vstack((np.pad(hx, ((0, 0), (0, N - n))),
                            np.hstack((r.T, b.T))))
    if not _same_space(mx, expected_x):
        raise ValueError("merged X checks do not match the supplied B and R")
    if not _same_space(mz[:, :n], hz):
        raise ValueError("merged Z checks do not preserve the source Z-check projection")
    targets = row_basis(binary_matrix(target_signatures, n_cols=k,
                                     name="target_signatures"))
    actual = row_basis(_mm(_mm(nullspace(b), r.T), jz.T))
    if not _same_space(actual, targets):
        raise ValueError("actual measured logical space differs from target_signatures")
    t = len(targets)
    complement = _complement(np.eye(k, dtype=np.uint8), targets)
    transform = np.vstack((targets, complement))
    dual = inverse(transform).T[t:]
    bare_x = np.pad(_mm(complement, jx), ((0, 0), (0, N - n)))

    ancilla_z = np.pad(nullspace(b.T), ((0, 0), (n, 0)))
    gauge_z = _complement(ancilla_z, mz)
    g = len(gauge_z)
    merged_k = N - rank(mx) - rank(mz)
    if merged_k - g != k - t:
        raise ValueError("ancilla gauge quotient does not leave the expected k - t subsystem")
    zero_signature = np.pad(jz, ((0, 0), (0, N - n)))
    equations = np.vstack((mz, gauge_z, zero_signature))
    rhs = np.vstack((np.zeros((len(mz), g), np.uint8), np.eye(g, dtype=np.uint8),
                     np.zeros((k, g), np.uint8)))
    try:
        gauge_x = solve(equations, rhs).T
        z_data = _mm(dual, jz)
        z_ancilla = solve(b.T, _mm(r.T, z_data.T)).T
    except ValueError as error:
        raise ValueError("cannot identify the protected subsystem in the specified gauge convention") from error
    bare_z = np.hstack((z_data, z_ancilla))
    # Gz_i pairs only with Gx_i, and has no data support.
    bare_z ^= _mm(_mm(bare_z, gauge_x.T), gauge_z)
    p = k - t
    valid = (not np.any(_mm(mz, bare_x.T))
             and not np.any(_mm(mx, bare_z.T))
             and not np.any(_mm(mz, gauge_x.T))
             and not np.any(_mm(mx, gauge_z.T))
             and not np.any(_mm(bare_x, gauge_z.T))
             and not np.any(_mm(bare_z, gauge_x.T))
             and np.array_equal(_mm(bare_x, bare_z.T), np.eye(p, dtype=np.uint8))
             and np.array_equal(_mm(gauge_x, gauge_z.T), np.eye(g, dtype=np.uint8)))
    if not valid:
        raise ValueError("protected subsystem failed its symplectic pairing audit")
    return dict(gauge_x=gauge_x, gauge_z=gauge_z, bare_x=bare_x, bare_z=bare_z,
                protected_logicals=p, gauge_qubits=g, merged_logicals=merged_k,
                target_rank=t, measured_signatures=actual,
                source_logical_transform=transform,
                convention="zero_original_logical_Z_signature",
                measurement_distance_certified=False)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bits(row):
    return sum(int(bit) << i for i, bit in enumerate(row))


def _candidate(value, h, detector):
    """Accept only an exactly binary, stabilizer-commuting protected logical."""
    if value is None:
        return None
    try:
        v = np.asarray(value, dtype=float)
        if v.shape != (h.shape[1],) or not np.all(np.isfinite(v)):
            return None
        rounded = np.rint(v)
        if np.any(np.abs(v - rounded) > 1e-6) or np.any((rounded != 0) & (rounded != 1)):
            return None
        v = rounded.astype(np.uint8)
        if np.any(_mm(h, v)) or not np.any(_mm(detector, v)):
            return None
        return v
    except (TypeError, ValueError, OverflowError):
        return None


def _physical_weight(v, physical_n=None):
    if physical_n is None:
        return int(np.asarray(v).sum())
    value = np.asarray(v, dtype=np.uint8)
    return int(np.count_nonzero(value[:physical_n] | value[physical_n:]))


def _remember(state, v, h, detector, physical_n=None):
    v = _candidate(v, h, detector)
    if v is None:
        return False
    weight = _physical_weight(v, physical_n)
    if state["upper_bound"] is None or weight < state["upper_bound"]:
        state.update(upper_bound=weight, witness=v.tolist(), witness_validated=True)
    sig = _bits(_mm(detector, v))
    old = state["_candidates"].get(sig)
    if old is None or weight < _physical_weight(old, physical_n):
        state["_candidates"][sig] = v.tolist()
    # Keep the witness list bounded even with many logical qubits.
    if len(state["_candidates"]) > 128:
        state["_candidates"] = dict(sorted(state["_candidates"].items(),
                                           key=lambda item: (_physical_weight(item[1], physical_n), item[0]))[:128])
    return True


def _finish(state):
    state["proven_lower_bound"] = state["lower_bound"]
    if state["lower_bound"] == state["upper_bound"]:
        state.update(status="exact", distance=state["upper_bound"])
    else:
        state.update(status="bounded" if state["upper_bound"] is not None else "unknown", distance=None)


def _heuristic(h, detector, state, rng, trials, deadline, physical_n=None):
    """Random column permutations change the chosen kernel basis; no lower bound."""
    n = h.shape[1]
    for trial in range(trials):
        if perf_counter() >= deadline:
            break
        permutation = np.arange(n) if trial == 0 else rng.permutation(n)
        basis = nullspace(h[:, permutation])
        restored = np.zeros_like(basis)
        restored[:, permutation] = basis
        for row in restored:
            if perf_counter() >= deadline:
                break
            _remember(state, row, h, detector, physical_n)
        state["heuristic_trials"] += 1


def _enumerate(h, detector, state, cutoff, deadline, budget):
    """Exhaust Hamming shells; only fully completed shells improve the bound."""
    n = h.shape[1]
    hcols = [_bits(h[:, q]) for q in range(n)]
    lcols = [_bits(detector[:, q]) for q in range(n)]
    maximum = min(cutoff, (state["upper_bound"] or (n + 1)) - 1)
    state["method"] = "enumeration"
    for weight in range(state["lower_bound"], maximum + 1):
        for support in combinations(range(n), weight):
            if perf_counter() >= deadline:
                state["stop_reason"] = "time_limit"
                return
            if budget[0] <= 0:
                state["stop_reason"] = "candidate_limit"
                return
            budget[0] -= 1
            state["enumerated_candidates"] += 1
            syndrome = logical = 0
            for q in support:
                syndrome ^= hcols[q]
                logical ^= lcols[q]
            if syndrome == 0 and logical:
                v = np.zeros(n, dtype=np.uint8)
                v[list(support)] = 1
                if not _remember(state, v, h, detector):
                    raise RuntimeError("enumeration produced an invalid logical witness")
                state.update(lower_bound=weight, stop_reason="enumeration_optimal",
                             lower_bound_certificate="exhausted_smaller_hamming_shells")
                return
        state.update(lower_bound=weight + 1,
                     lower_bound_certificate="exhausted_hamming_shells")
    state["stop_reason"] = ("enumeration_optimal" if state["lower_bound"] == state["upper_bound"]
                            else "max_weight")


def _solve_gurobi(h, detector, cutoff, deadline, initial, seed, physical_n=None):
    """Solve a CSS direction or paired symplectic coordinates before deadline."""
    try:
        import gurobipy as gp
    except (ImportError, OSError) as error:
        return dict(status="unavailable", error=f"{type(error).__name__}: {error}")
    env = model = None
    try:
        if perf_counter() >= deadline:
            return dict(status=9, message="deadline reached before model construction")
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("high_rate_css_dressed_distance", env=env)
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.Params.Seed = seed % 2_000_000_000
        n = h.shape[1]
        e = model.addVars(n, vtype=gp.GRB.BINARY, name="error")
        for i, row in enumerate(h):
            if i % 64 == 0 and perf_counter() >= deadline:
                return dict(status=9, message="deadline reached during check model construction")
            support = np.flatnonzero(row).tolist()
            u = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=len(support) // 2,
                             name=f"stabilizer_parity_{i}")
            model.addConstr(gp.quicksum(e[q] for q in support) == 2 * u)
        parity = model.addVars(len(detector), vtype=gp.GRB.BINARY, name="logical_signature")
        for i, row in enumerate(detector):
            if i % 64 == 0 and perf_counter() >= deadline:
                return dict(status=9, message="deadline reached during logical model construction")
            support = np.flatnonzero(row).tolist()
            u = model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=len(support) // 2,
                             name=f"logical_parity_{i}")
            model.addConstr(gp.quicksum(e[q] for q in support) == 2 * u + parity[i])
        model.addConstr(parity.sum() >= 1)
        if physical_n is None:
            objective = e.sum()
        else:
            if n != 2 * physical_n:
                raise ValueError("symplectic coordinates must have twice the physical width")
            occupied = model.addVars(physical_n, vtype=gp.GRB.BINARY, name="occupied_qubit")
            for q in range(physical_n):
                if q % 64 == 0 and perf_counter() >= deadline:
                    return dict(status=9, message="deadline reached during Pauli-weight model construction")
                model.addConstr(occupied[q] >= e[q])
                model.addConstr(occupied[q] >= e[physical_n + q])
                model.addConstr(occupied[q] <= e[q] + e[physical_n + q])
            objective = occupied.sum()
        model.addConstr(objective <= cutoff)
        model.setObjective(objective, gp.GRB.MINIMIZE)
        if initial is not None and _physical_weight(initial, physical_n) <= cutoff:
            for q, bit in enumerate(initial):
                e[q].Start = int(bit)
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return dict(status=9, message="deadline reached during model construction")
        model.Params.TimeLimit = remaining
        model.optimize()
        result = dict(status=int(model.Status), runtime_seconds=float(model.Runtime),
                      solution_count=int(model.SolCount), candidate=None, objective=None)
        if model.SolCount:
            result.update(candidate=[float(e[q].X) for q in range(n)], objective=float(model.ObjVal))
        try:
            result["objective_bound"] = float(model.ObjBoundC)
        except (AttributeError, gp.GurobiError):
            pass
        return result
    except Exception as error:
        return dict(status="solver_error", error=f"{type(error).__name__}: {error}")
    finally:
        if model is not None:
            model.dispose()
        if env is not None:
            env.dispose()


def _use_gurobi_result(result, h, detector, state, cutoff, physical_n=None):
    """Numerical/error statuses never provide a lower-bound certificate."""
    state["method"] = "gurobi"
    info = {key: (None if isinstance(value, Real) and not np.isfinite(value) else value)
            for key, value in result.items() if key != "candidate"}
    state["solver"] = info
    status = result.get("status")
    candidate = _candidate(result.get("candidate"), h, detector)
    if candidate is not None and _physical_weight(candidate, physical_n) <= cutoff:
        _remember(state, candidate, h, detector, physical_n)
    else:
        candidate = None
    bound = result.get("objective_bound")
    finite_bound = (isinstance(bound, Real) and not isinstance(bound, bool)
                    and np.isfinite(bound) and abs(bound) < 1e100)
    objective = result.get("objective")
    if status == 2 and candidate is not None and isinstance(objective, Real):
        if np.isfinite(objective) and abs(objective - _physical_weight(candidate, physical_n)) <= 1e-6:
            state.update(lower_bound=max(state["lower_bound"], _physical_weight(candidate, physical_n)),
                         lower_bound_certificate="gurobi_optimal_under_numerical_tolerances",
                         stop_reason="gurobi_optimal")
            return
    if (status == 3 and result.get("solution_count", 0) == 0
            and result.get("candidate") is None):
        proposed = cutoff + 1
        if state["upper_bound"] is None or proposed <= state["upper_bound"]:
            state.update(lower_bound=max(state["lower_bound"], proposed),
                         lower_bound_certificate="gurobi_infeasible_cutoff_under_numerical_tolerances",
                         stop_reason="max_weight")
            return
        info["bound_rejected"] = "infeasibility contradicts a validated witness"
    if status in (8, 9) and finite_bound:
        if state["upper_bound"] is not None and bound > state["upper_bound"] + 1e-6:
            info["bound_rejected"] = "objective bound contradicts a validated witness"
        else:
            proposed = min(cutoff + 1, ceil(float(bound) - 1e-6))
            state.update(lower_bound=max(state["lower_bound"], proposed),
                         lower_bound_certificate="gurobi_objective_bound_under_numerical_tolerances")
    state["stop_reason"] = {8: "node_limit", 9: "time_limit"}.get(status, str(status))


def css_dressed_distance(merged_hx, merged_hz, bare_x, bare_z, *, method="auto",
                         time_limit=60, max_candidates=100_000, max_weight=None,
                         seed=0, trials=16):
    """Bound protected CSS dressed distance with one shared wall-clock budget.

    ``auto`` prefers optional Gurobi and falls back to bounded exhaustive
    enumeration if it is unavailable. ``enum`` and ``heuristic`` are usable
    without Gurobi. ``max_candidates`` is shared by both enumeration searches.
    ``max_weight`` restricts certification/search, not the validity of an
    independently discovered upper-bound witness. Random kernel searches only
    improve upper bounds. No timeout or missing solver is treated as a proof.
    """
    started = perf_counter()
    if (isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, Real)
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be a positive finite number of seconds")
    if method not in ("auto", "gurobi", "enum", "heuristic"):
        raise ValueError("method must be auto, gurobi, enum, or heuristic")
    max_candidates = _integer(max_candidates, "max_candidates")
    trials = _integer(trials, "trials")
    seed = _integer(seed, "seed")
    hx = binary_matrix(merged_hx, name="merged_hx")
    n = hx.shape[1]
    hz = binary_matrix(merged_hz, n_cols=n, name="merged_hz")
    lx = binary_matrix(bare_x, n_cols=n, name="bare_x")
    lz = binary_matrix(bare_z, n_cols=n, name="bare_z")
    cutoff = n if max_weight is None else _integer(max_weight, "max_weight")
    cutoff = min(cutoff, n)
    p = len(lx)
    if (len(lz) != p or np.any(_mm(hx, hz.T)) or np.any(_mm(hz, lx.T))
            or np.any(_mm(hx, lz.T))
            or not np.array_equal(_mm(lx, lz.T), np.eye(p, dtype=np.uint8))):
        raise ValueError("checks and bare protected logical matrices are inconsistent")
    common = dict(scope="protected_css_dressed_static", protected_logicals=p,
                  measurement_distance_certified=False,
                  measurement_distance_note="No measurement-record, circuit, or spacetime certificate is supplied.",
                  time_limit=float(time_limit), max_weight=cutoff,
                  max_candidates=max_candidates, seed=seed, method_requested=method)
    if p == 0:
        return dict(common, status="no_protected_logicals", distance=None, lower_bound=None,
                    proven_lower_bound=None, upper_bound=None, x=None, z=None,
                    low_weight_x_witnesses=[], low_weight_z_witnesses=[],
                    distance_convention="infinite_only_for_the_empty_protected_subsystem",
                    runtime_seconds=perf_counter() - started)
    deadline = started + float(time_limit)
    rng = np.random.default_rng(seed)
    states = []
    for h, detector, initial in ((hz, lz, lx), (hx, lx, lz)):
        state = dict(lower_bound=1, upper_bound=None, distance=None, status="unknown",
                     witness=None, witness_validated=False, method="initial_bare_logical",
                     stop_reason="not_started", heuristic_trials=0, enumerated_candidates=0,
                     lower_bound_certificate="nonzero_protected_logical_has_positive_weight",
                     _candidates={})
        for row in initial:
            _remember(state, row, h, detector)
        states.append(state)
    budget = [max_candidates]
    for index, (h, detector, state) in enumerate(((hz, lz, states[0]), (hx, lx, states[1]))):
        now = perf_counter()
        # Reserve half the remaining budget for the other direction if needed.
        reserve = 2 if index == 0 and states[1]["upper_bound"] != 1 else 1
        direction_deadline = now + max(0.0, deadline - now) / reserve
        heuristic_deadline = min(direction_deadline, now + max(0.0, direction_deadline - now) * .10)
        if state["upper_bound"] != 1 and perf_counter() < direction_deadline:
            _heuristic(h, detector, state, rng, trials, heuristic_deadline)
        _finish(state)
        if state["status"] == "exact":
            state["stop_reason"] = "positive_weight_bound_matches_witness"
            continue
        if perf_counter() >= direction_deadline:
            state["stop_reason"] = "time_limit"
            continue
        if method == "heuristic":
            state.update(method="random_column_kernel_basis", stop_reason="heuristic_only")
            continue
        search_cutoff = min(cutoff, state["upper_bound"] - 1)
        if search_cutoff < state["lower_bound"]:
            state["stop_reason"] = "max_weight"
            continue
        if method in ("auto", "gurobi"):
            result = _solve_gurobi(h, detector, search_cutoff, direction_deadline,
                                   state["witness"], seed + index)
            _use_gurobi_result(result, h, detector, state, search_cutoff)
            _finish(state)
            fallback = result.get("status") in ("unavailable", "solver_error")
            if method == "auto" and fallback and perf_counter() < direction_deadline:
                _enumerate(h, detector, state, search_cutoff, direction_deadline, budget)
        else:
            _enumerate(h, detector, state, search_cutoff, direction_deadline, budget)
        _finish(state)
    lists = []
    for state in states:
        _finish(state)
        vectors = list(state.pop("_candidates").values())
        lists.append(sorted((v for v in vectors if sum(v) <= cutoff), key=lambda v: (sum(v), v)))
    x, z = states
    lower = min(x["lower_bound"], z["lower_bound"])
    upper = min(x["upper_bound"], z["upper_bound"])
    exact = lower == upper
    selected = x if x["upper_bound"] <= z["upper_bound"] else z
    return dict(common, x=x, z=z, lower_bound=lower, proven_lower_bound=lower,
                upper_bound=upper, distance=upper if exact else None,
                status="exact" if exact else "bounded", witness=selected["witness"],
                witness_type="X" if selected is x else "Z", witness_validated=True,
                low_weight_x_witnesses=lists[0], low_weight_z_witnesses=lists[1],
                enumerated_candidates=max_candidates - budget[0],
                runtime_seconds=perf_counter() - started)


def _enumerate_symplectic(h, detector, state, cutoff, deadline, budget, n):
    """Enumerate physical support shells and their three Pauli assignments."""
    hcols = [_bits(h[:, q]) for q in range(2 * n)]
    lcols = [_bits(detector[:, q]) for q in range(2 * n)]
    maximum = min(cutoff, (state["upper_bound"] or (n + 1)) - 1)
    state["method"] = "symplectic_enumeration"
    for weight in range(state["lower_bound"], maximum + 1):
        for support in combinations(range(n), weight):
            for assignments in product((1, 2, 3), repeat=weight):
                if perf_counter() >= deadline:
                    state["stop_reason"] = "time_limit"
                    return
                if budget[0] <= 0:
                    state["stop_reason"] = "candidate_limit"
                    return
                budget[0] -= 1
                state["enumerated_candidates"] += 1
                syndrome = logical = 0
                for q, pauli in zip(support, assignments):
                    if pauli & 1:
                        syndrome ^= hcols[q]
                        logical ^= lcols[q]
                    if pauli & 2:
                        syndrome ^= hcols[n + q]
                        logical ^= lcols[n + q]
                if syndrome == 0 and logical:
                    vector = np.zeros(2 * n, np.uint8)
                    for q, pauli in zip(support, assignments):
                        vector[q], vector[n + q] = pauli & 1, bool(pauli & 2)
                    if not _remember(state, vector, h, detector, n):
                        raise RuntimeError("symplectic enumeration produced an invalid logical witness")
                    state.update(lower_bound=weight, stop_reason="enumeration_optimal",
                                 lower_bound_certificate="exhausted_smaller_physical_Pauli_shells")
                    return
        state.update(lower_bound=weight + 1,
                     lower_bound_certificate="exhausted_physical_Pauli_shells")
    state["stop_reason"] = ("enumeration_optimal" if state["lower_bound"] == state["upper_bound"]
                            else "max_weight")


def _symplectic_witness(vector, n):
    if vector is None:
        return None
    x, z = vector[:n], vector[n:]
    return dict(x=x, z=z, weight=_physical_weight(vector, n),
                pauli="".join("IXZY"[int(a) + 2 * int(b)] for a, b in zip(x, z)))


def symplectic_gauge_free_distance(check_x, check_z, *, method="auto", time_limit=60,
                                   max_weight=None, seed=0, trials=16,
                                   max_candidates=100_000):
    """Bound a gauge-free merged code's protected static Pauli distance.

    The caller must establish that the merged code has no ancillary gauge
    qubits. Under that assumption ordinary stabilizer distance equals the
    protected distance. A general subsystem code must use an explicitly
    identified protected detector instead of this function.

    Write S=(check_x|check_z), H=(check_z|check_x), and K=ker(H).
    Kernel rows independent modulo S span all logical classes. Their
    symplectic pairing detects exactly K minus row(S): an error commuting
    with all of K belongs to K-perp = row(S). This tests the entire logical
    space without silently counting only one selected logical operator.

    Pauli weight is wt(x OR z), so a Y contributes one. Random kernel bases
    provide upper bounds only. All search stages share one deadline, and
    enumeration checks time and candidate budgets for every assignment.
    Gurobi model construction checks that same deadline; its solve receives
    only the time remaining. Linear-algebra primitives and external solver
    startup are atomic, so this is not a process-killing real-time limit.
    """
    started = perf_counter()
    if (isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, Real)
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be a positive finite number of seconds")
    if method not in ("auto", "gurobi", "enum", "heuristic"):
        raise ValueError("method must be auto, gurobi, enum, or heuristic")
    max_candidates = _integer(max_candidates, "max_candidates")
    seed, trials = _integer(seed, "seed"), _integer(trials, "trials")
    x = binary_matrix(check_x, name="check_x")
    n = x.shape[1]
    z = binary_matrix(check_z, n_cols=n, name="check_z")
    if x.shape != z.shape or np.any(_mm(x, z.T) ^ _mm(z, x.T)):
        raise ValueError("check_x and check_z must be equally shaped commuting checks")
    stabilizers = row_basis(np.hstack((x, z)))
    h = np.hstack((stabilizers[:, n:], stabilizers[:, :n]))
    k = n - len(stabilizers)
    cutoff = min(n, n if max_weight is None else _integer(max_weight, "max_weight"))
    deadline = started + float(time_limit)
    common = dict(scope="protected_mixed_stabilizer_gauge_free", protected_logicals=k,
                  gauge_qubits=0, gauge_free_is_caller_precondition=True,
                  measurement_distance_certified=False,
                  measurement_distance_note="No measurement-record, circuit, or spacetime certificate is supplied.",
                  time_limit=float(time_limit), max_weight=cutoff,
                  max_candidates=max_candidates, seed=seed, method_requested=method)
    if k == 0:
        return dict(common, status="no_protected_logicals", distance=None,
                    lower_bound=None, proven_lower_bound=None, upper_bound=None,
                    witness=None, witness_validated=False, low_weight_x_witnesses=[],
                    low_weight_z_witnesses=[], low_weight_pauli_witnesses=[],
                    distance_convention="infinite_only_for_the_empty_protected_subsystem",
                    runtime_seconds=perf_counter() - started, enumerated_candidates=0)
    kernel = nullspace(h)
    logical = _complement(kernel, stabilizers)
    if len(logical) != 2 * k:
        raise RuntimeError("symplectic logical quotient has an inconsistent dimension")
    detector = np.hstack((logical[:, n:], logical[:, :n]))
    state = dict(lower_bound=1, upper_bound=None, distance=None, status="unknown",
                 witness=None, witness_validated=False, method="initial_logical_basis",
                 stop_reason="not_started", heuristic_trials=0, enumerated_candidates=0,
                 lower_bound_certificate="nonzero_logical_has_positive_physical_weight",
                 _candidates={})
    # These quotient representatives guarantee a valid initial upper bound;
    # its existence never relies on a solver or a heuristic finding a vector.
    for row in logical:
        _remember(state, row, h, detector, n)
        if state["upper_bound"] == 1 or perf_counter() >= deadline:
            break
    rng = np.random.default_rng(seed)
    now = perf_counter()
    if state["upper_bound"] != 1 and now < deadline:
        _heuristic(h, detector, state, rng, trials,
                   min(deadline, now + (deadline-now) * .10), n)
    _finish(state)
    budget = [max_candidates]
    if state["status"] == "exact":
        state["stop_reason"] = "positive_weight_bound_matches_witness"
    elif perf_counter() >= deadline:
        state["stop_reason"] = "time_limit"
    elif method == "heuristic":
        state.update(method="random_column_kernel_basis", stop_reason="heuristic_only")
    else:
        search_cutoff = min(cutoff, state["upper_bound"] - 1)
        if search_cutoff < state["lower_bound"]:
            state["stop_reason"] = "max_weight"
        elif method in ("auto", "gurobi"):
            result = _solve_gurobi(h, detector, search_cutoff, deadline,
                                   state["witness"], seed, physical_n=n)
            _use_gurobi_result(result, h, detector, state, search_cutoff, n)
            if (method == "auto" and result.get("status") in ("unavailable", "solver_error")
                    and perf_counter() < deadline):
                _enumerate_symplectic(h, detector, state, search_cutoff, deadline, budget, n)
        else:
            _enumerate_symplectic(h, detector, state, search_cutoff, deadline, budget, n)
    _finish(state)
    candidates = list(state.pop("_candidates").values())
    candidates = sorted((v for v in candidates if _physical_weight(v, n) <= cutoff),
                        key=lambda v: (_physical_weight(v, n), v))
    state["witness"] = _symplectic_witness(state["witness"], n)
    return dict(common, **state,
                low_weight_x_witnesses=[v[:n] for v in candidates],
                low_weight_z_witnesses=[v[n:] for v in candidates],
                low_weight_pauli_witnesses=[_symplectic_witness(v, n) for v in candidates],
                runtime_seconds=perf_counter()-started)


__all__ = ["build_css_protected_basis", "css_dressed_distance", "symplectic_gauge_free_distance"]
