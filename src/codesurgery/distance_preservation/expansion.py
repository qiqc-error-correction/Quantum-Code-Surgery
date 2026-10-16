"""Bounded-degree auxiliary-graph augmentation with checkable certificates.

The default implementation tries seeded random degree increments, testing
actual candidate-code distance before global expansion.  Random regular
graph expansion motivates the heuristic; it is not a guarantee for these
constrained overlays.  The previous Hamiltonian-cycle search is opt-in.
Original edges (including parallel edges) are preserved.  No sampled cut or
floating-point eigenvalue is used as a proof of expansion.

For small graphs we enumerate the relative expansion beta_d.  Otherwise an
exact integer test certifies the sufficient condition lambda_2(L) > 2, hence
h >= lambda_2/2 > 1.  This stronger global bound implies beta_d >= 1 for any
port subset.  Failure to obtain a certificate is reported as failure, never
as a distance guarantee.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..universal_adapters.graphs import graph_matrix, relative_expansion


class ExpansionConstructionError(ValueError):
    """A requested certified construction could not fit its stated limits."""

    def __init__(self, message, *, reason, details=None):
        super().__init__(message)
        self.reason = reason
        self.details = {} if details is None else details


def _integer(value, name, *, minimum=1, maximum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum or (maximum is not None and value > maximum)):
        interval = f"[{minimum},{maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {interval}")
    return int(value)


def _digest(b):
    digest = hashlib.sha256()
    digest.update(str(tuple(b.shape)).encode("ascii"))
    digest.update(b.tobytes(order="C"))
    return digest.hexdigest()


def _exact_spectral_test(b, *, certificate_limit, max_entries):
    """Certify lambda_2(L)>2 by an exact Sylvester/Bareiss test.

    A=nL-2nI+2J has zero row sum.  A principal (n-1)-minor being positive
    definite implies A is positive semidefinite with kernel span(1): write
    any vector as y+c*1 with y's last coordinate zero.  On 1-perp this is
    precisely the condition L-2I>0.  Fraction-free elimination gives its
    leading principal determinants using integer arithmetic only.
    """
    n = b.shape[1]
    order = n - 1
    work = order * max(0, order - 1) * (2 * order - 1) // 6
    if n > certificate_limit:
        return dict(proved=False, reason="graph exceeds certificate_limit",
                    resource_limited=True)
    if n * n > max_entries or work > max_entries:
        return dict(proved=False, reason="integer certificate exceeds max_entries work budget",
                    resource_limited=True)
    # Python integers prevent overflow even for parallel-edge multiplicities.
    laplacian = [[0] * n for _ in range(n)]
    for row in b:
        u, v = map(int, np.flatnonzero(row))
        laplacian[u][u] += 1
        laplacian[v][v] += 1
        laplacian[u][v] -= 1
        laplacian[v][u] -= 1
    matrix = [[n * laplacian[i][j] + 2 - (2 * n if i == j else 0)
               for j in range(order)] for i in range(order)]
    # Hadamard's bound also bounds all intermediate minors.  This conservative
    # bit budget keeps an accidentally enormous integer certificate bounded.
    max_abs = max((abs(x) for row in matrix for x in row), default=1)
    determinant_bits_bound = order * (max_abs.bit_length() + (order.bit_length() + 1) // 2)
    if determinant_bits_bound > 16384:
        return dict(proved=False, reason="integer certificate exceeds bigint budget",
                    resource_limited=True)
    previous = 1
    maximum_pivot_bits = 0
    for k in range(order):
        pivot = matrix[k][k]
        if pivot <= 0:
            return dict(proved=False, reason="strict spectral sufficient condition not established",
                        resource_limited=False, failed_pivot=k)
        maximum_pivot_bits = max(maximum_pivot_bits, pivot.bit_length())
        for i in range(k + 1, order):
            for j in range(i, order):
                numerator = pivot * matrix[i][j] - matrix[i][k] * matrix[k][j]
                value, remainder = divmod(numerator, previous)
                if remainder:
                    raise ArithmeticError("Nonexact division in integer expansion certificate")
                matrix[i][j] = matrix[j][i] = value
        previous = pivot
    return dict(proved=True, reason="exact integer Sylvester certificate",
                principal_minor_order=order, maximum_pivot_bits=maximum_pivot_bits,
                laplacian_gap_lower_bound=2, strict_gap=True,
                arithmetic="Python arbitrary-precision integers; fraction-free elimination",
                resource_limited=False)


def certify_expansion(b, *, ports=None, distance=None, exact_limit=20,
                      certificate_limit=128, max_entries=50_000_000):
    """Recompute an expansion certificate from the actual incidence matrix.

    The result uses the fields of ``relative_expansion``.  ``condition_met``
    is true only for exhaustive integer cut counts, a trivial connected-graph
    certificate, or the exact integer spectral proof.  A false result means
    the requested sufficient condition was not proved; it need not mean that
    the actual code distance decreases.
    """
    exact_limit = _integer(exact_limit, "exact_limit", maximum=24)
    certificate_limit = _integer(certificate_limit, "certificate_limit")
    max_entries = _integer(max_entries, "max_entries")
    b = graph_matrix(b, max_entries=max_entries)
    vertices = list(range(b.shape[1])) if ports is None else list(ports)
    report = relative_expansion(b, vertices, distance=distance,
                                exact_limit=exact_limit, max_entries=max_entries)
    report["graph_sha256"] = _digest(b)
    report["global_h_lower_bound"] = None
    report["global_h_certified"] = False
    if report["condition_met"]:
        # beta_t is global h only with all vertices and no truncated denominator.
        if len(vertices) == b.shape[1] and report["parameter_t"] >= b.shape[1] // 2:
            report["global_h_lower_bound"] = report["lower_bound"]
            report["global_h_certified"] = True
        return report
    # An exact counterexample cannot be repaired by a different certificate.
    if report["exact"]:
        return report
    proof = _exact_spectral_test(b, certificate_limit=certificate_limit,
                                 max_entries=max_entries)
    report["spectral_proof"] = proof
    if proof["proved"]:
        report.update(condition_met=True, lower_bound=1.0, beta=None,
                      certificate="exact integer Laplacian certificate: lambda_2 > 2 implies h >= 1",
                      global_h_lower_bound=1.0, global_h_certified=True)
    return report


def _enhance_hamiltonian(b, *, ports=None, distance=None, max_degree=12,
                      seed=0, attempts=12, exact_limit=20,
                      certificate_limit=128, max_entries=50_000_000,
                      allow_partial_overlay=False):
    """Preserve ``b`` and add bounded-degree cycle overlays until certified.

    A successful return is ``(augmented_incidence, report)``.  Every original
    row remains in the same position.  The returned maximum degree is at
    most ``max_degree``; failure within the degree, search, or certification
    budget raises ``ExpansionConstructionError``.  This is a finite search,
    not a claim that every feasible graph can be found with these budgets.

    With ``allow_partial_overlay=True``, a failed/unavailable full-cycle
    search falls back to adding individual edges across exact violating
    cuts on small graphs.  This uses spare degrees at individual vertices,
    rather than requiring two spare degrees at every vertex.  It is a
    separately identified implementation heuristic, not an algorithm
    specified by the cited paper.  Every successful candidate still has a
    complete expansion certificate; unsupported sizes fail closed.
    """
    max_degree = _integer(max_degree, "max_degree")
    seed = _integer(seed, "seed", minimum=0)
    attempts = _integer(attempts, "attempts")
    exact_limit = _integer(exact_limit, "exact_limit", maximum=24)
    certificate_limit = _integer(certificate_limit, "certificate_limit")
    max_entries = _integer(max_entries, "max_entries")
    if not isinstance(allow_partial_overlay, (bool, np.bool_)):
        raise ValueError("allow_partial_overlay must be a boolean")
    allow_partial_overlay = bool(allow_partial_overlay)
    base = graph_matrix(b, max_entries=max_entries)
    vertices = None if ports is None else list(ports)
    initial_degree = int(base.sum(axis=0, dtype=np.uint64).max())
    if initial_degree > max_degree:
        raise ExpansionConstructionError(
            f"Input graph degree {initial_degree} exceeds max_degree={max_degree}; "
            "increase the degree budget or supply a different base graph",
            reason="degree_budget", details=dict(initial_max_degree=initial_degree,
                                                  max_degree=max_degree))
    initial = certify_expansion(base, ports=vertices, distance=distance,
                                exact_limit=exact_limit, certificate_limit=certificate_limit,
                                max_entries=max_entries)

    def result(matrix, certificate, attempt, layers, *, strategy="hamiltonian_cycle_overlays",
               partial_attempts=0, partial_steps=None):
        report = dict(certificate)
        report.update(initial=initial, initial_edges=int(len(base)),
                      added_edges=[list(map(int, np.flatnonzero(row)))
                                   for row in matrix[len(base):]],
                      added_edge_count=int(len(matrix) - len(base)),
                      initial_max_degree=initial_degree,
                      final_max_degree=int(matrix.sum(axis=0, dtype=np.uint64).max()),
                      max_degree=max_degree, seed=seed, attempts_used=attempt,
                      overlay_layers=layers,
                      augmentation_strategy=strategy,
                      construction="bounded-degree random Hamiltonian-cycle overlays",
                      source="arXiv:2410.02213, Theorem 3, bounded-degree random-edge augmentation",
                      implementation_choice="deterministic seeded cycle overlays, accepted only after certification")
        if strategy == "exact_worst_cut_partial_overlay":
            report.update(
                construction="bounded-degree individual-edge augmentation across exact violating cuts",
                source="implementation choice; certified using the actual incidence matrix",
                implementation_choice="greedy low-degree, unused-pair preference with seeded ties; not a prescribed paper algorithm",
                partial_search_attempts=partial_attempts,
                partial_edge_steps=partial_steps,
            )
        return matrix, report

    def partial_search(rng, already_tried):
        n = base.shape[1]
        if n > exact_limit or 12 * (1 << n) > max_entries or not initial["exact"]:
            raise ExpansionConstructionError(
                "Partial-overlay augmentation requires an exact worst-cut certificate "
                "within exact_limit and max_entries",
                reason="certificate_budget", details=initial)
        degrees_at_start = np.asarray(base.sum(axis=0), dtype=np.int64)
        max_additions = int((max_degree - degrees_at_start).sum() // 2)
        last_certificate = initial
        candidates_tried = already_tried
        matrix_limited = False
        for attempt in range(1, attempts + 1):
            augmented = base.copy()
            degrees = degrees_at_start.copy()
            occupied = {tuple(map(int, np.flatnonzero(row))) for row in base}
            certificate = initial
            steps = []
            for _ in range(max_additions):
                if (len(augmented) + 1) * n > max_entries:
                    matrix_limited = True
                    break
                inside = set(certificate["worst_cut"])
                available_inside = [v for v in range(n) if v in inside and degrees[v] < max_degree]
                available_outside = [v for v in range(n) if v not in inside and degrees[v] < max_degree]
                preferred = []
                best_score = None
                for u in available_inside:
                    for v in available_outside:
                        edge = tuple(sorted((u, v)))
                        score = (max(int(degrees[u]), int(degrees[v])),
                                 int(degrees[u] + degrees[v]), edge in occupied)
                        if best_score is None or score < best_score:
                            best_score, preferred = score, [edge]
                        elif score == best_score:
                            preferred.append(edge)
                if not preferred:
                    break
                u, v = preferred[int(rng.integers(len(preferred)))]
                edge_row = np.zeros((1, n), dtype=np.uint8)
                edge_row[0, [u, v]] = 1
                augmented = np.vstack((augmented, edge_row))
                degrees[[u, v]] += 1
                occupied.add((u, v))
                steps.append(dict(worst_cut=list(certificate["worst_cut"]),
                                  previous_boundary=certificate["boundary_weight"],
                                  denominator=certificate["denominator"],
                                  added_edge=[u, v]))
                certificate = certify_expansion(
                    augmented, ports=vertices, distance=distance, exact_limit=exact_limit,
                    certificate_limit=certificate_limit, max_entries=max_entries)
                candidates_tried += 1
                last_certificate = certificate
                if certificate["condition_met"]:
                    return result(augmented, certificate, candidates_tried, 0,
                                  strategy="exact_worst_cut_partial_overlay",
                                  partial_attempts=attempt, partial_steps=steps)
        if matrix_limited:
            raise ExpansionConstructionError(
                "Partial-overlay incidence exceeds max_entries before certification",
                reason="matrix_budget", details=last_certificate)
        raise ExpansionConstructionError(
            "No certified partial overlay found within the degree and attempt budgets; "
            "increase max_degree/attempts, change seed, or provide a certified base graph",
            reason="degree_budget" if candidates_tried == already_tried else "search_budget",
            details=dict(last_certificate=last_certificate, candidates_tried=candidates_tried,
                         max_degree=max_degree, partial_attempts=attempts))

    if initial["condition_met"]:
        return result(base, initial, 0, 0)
    proof = initial.get("spectral_proof", {})
    if proof.get("resource_limited"):
        raise ExpansionConstructionError(
            "No expansion certificate fits the resource limits: " + proof["reason"],
            reason="certificate_budget", details=initial)
    # Each full cycle consumes two units of degree at every vertex.  For two
    # vertices its two edges are parallel (valid ordinary multigraph edges).
    n = base.shape[1]
    layers_available = (max_degree - initial_degree) // 2
    rng = np.random.default_rng(seed)
    if layers_available == 0:
        if allow_partial_overlay:
            return partial_search(rng, 0)
        raise ExpansionConstructionError(
            "Insufficient degree budget for a cycle overlay; increase max_degree",
            reason="degree_budget", details=initial)
    tried = 0
    last = initial
    for layers in range(1, layers_available + 1):
        if (len(base) + layers * n) * n > max_entries:
            if allow_partial_overlay:
                return partial_search(rng, tried)
            raise ExpansionConstructionError(
                "Expansion overlay incidence exceeds max_entries",
                reason="matrix_budget", details=last)
        for attempt in range(attempts):
            overlay = np.zeros((layers * n, n), dtype=np.uint8)
            for layer in range(layers):
                permutation = rng.permutation(n)
                for j in range(n):
                    u, v = int(permutation[j]), int(permutation[(j + 1) % n])
                    overlay[layer * n + j, [u, v]] = 1
            augmented = np.concatenate((base, overlay), axis=0)
            last = certify_expansion(augmented, ports=vertices, distance=distance,
                                      exact_limit=exact_limit, certificate_limit=certificate_limit,
                                      max_entries=max_entries)
            tried += 1
            if last["condition_met"]:
                return result(augmented, last, tried, layers)
    if allow_partial_overlay:
        return partial_search(rng, tried)
    raise ExpansionConstructionError(
        "No certified expansion found within the degree and attempt budgets; "
        "increase max_degree/attempts, change seed, or provide a certified base graph",
        reason="search_budget", details=dict(last_certificate=last, candidates_tried=tried,
                                              max_degree=max_degree, attempts_per_layer=attempts))


def _plain(value):
    """Keep audit records serializable even when an evaluator uses NumPy."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _degree_matching(base, cap, rng, max_entries):
    """Fill residual degree stubs, preferring new pairs and balanced deficits.

    This is a seeded, constrained matching heuristic, not a uniform sampler
    of regular graphs.  Parallel edges are permitted only when no unused
    pair is available for the selected endpoint; loops are never permitted.
    Odd total residual degree necessarily leaves an unmatched stub.  Other
    unmatchable residual stubs are reported by the caller as well.
    """
    n = base.shape[1]
    degrees = np.asarray(base.sum(axis=0), dtype=np.int64)
    residual = cap - degrees
    occupied = {tuple(map(int, np.flatnonzero(row))) for row in base}
    rows = []
    while True:
        available = np.flatnonzero(residual > 0)
        if len(available) < 2:
            break
        if (len(base) + len(rows) + 1) * n > max_entries:
            raise ExpansionConstructionError(
                "Degree-matching incidence exceeds max_entries",
                reason="matrix_budget")
        highest = available[residual[available] == residual[available].max()]
        u = int(rng.choice(highest))
        others = available[available != u]
        unused = [int(v) for v in others if tuple(sorted((u, int(v)))) not in occupied]
        choices = np.asarray(unused if unused else others, dtype=np.int64)
        choices = choices[residual[choices] == residual[choices].max()]
        v = int(rng.choice(choices))
        row = np.zeros(n, dtype=np.uint8)
        row[[u, v]] = 1
        rows.append(row)
        residual[[u, v]] -= 1
        occupied.add(tuple(sorted((u, v))))
    if not rows:
        return base.copy()
    return np.vstack((base, np.asarray(rows, dtype=np.uint8)))


def _strict_global_certificate(matrix, *, exact_limit, certificate_limit, max_entries):
    """Certify global h > 1, independent of ports and target distance."""
    report = certify_expansion(
        matrix, ports=list(range(matrix.shape[1])), distance=matrix.shape[1],
        exact_limit=exact_limit, certificate_limit=certificate_limit,
        max_entries=max_entries)
    # Use integer counts, never a floating tolerance at the strict boundary.
    if report.get("exact") and "boundary_weight" in report:
        strict = report["boundary_weight"] > report["denominator"]
        report["global_h_lower_bound"] = report["lower_bound"]
        report["global_h_certified"] = True
    else:
        strict = bool(report.get("spectral_proof", {}).get("proved"))
    report.update(condition_met=bool(strict), strict_global_h=bool(strict),
                  expansion_strict=bool(strict), expansion_evaluated=True,
                  expansion_requirement="global h > 1")
    if strict and report.get("spectral_proof", {}).get("proved"):
        report["certificate"] = "exact integer Laplacian certificate: lambda_2 > 2 implies h > 1"
    return report


def enhance_expansion(b, *, ports=None, distance=None, max_degree=12,
                      seed=0, attempts=12, exact_limit=20,
                      certificate_limit=128, max_entries=50_000_000,
                      allow_partial_overlay=False, strategy="degree_incremental",
                      candidate_evaluator=None):
    """Try small degree increments before spending extra vertex degree.

    The default adds random residual-degree matchings, retrying each degree
    cap before increasing it.  A regular degree-two base is first extended
    toward degree three.  Odd vertex/degree parity can only give a near-
    regular candidate and is reported explicitly.  Existing rows are never
    changed.  The configured maximum degree is a hard limit.

    ``candidate_evaluator(matrix, context)`` runs before expansion for each
    augmented candidate.  It must return ``distance_state`` (``adequate``,
    ``insufficient``, or ``unknown``) and Boolean ``eligible`` indicating
    that the complete candidate code meets its final sparsity constraints.
    Adequate, eligible candidates are accepted without calculating h.  For
    unknown distance, acceptance requires a rigorous global h > 1 proof;
    an insufficient-distance witness contradicting such a proof raises.
    An ineligible candidate is never accepted, regardless of expansion.

    With ``allow_partial_overlay=True`` exact worst cuts can guide local
    additions in residual degree slots.  After exhausting the smaller cap,
    the best exact-cut candidate from the preceding cap is first grown
    locally before full regularization at the next cap, retaining its seed.  This is our bounded search
    heuristic, not a prescribed algorithm from a paper.  Parallel edges
    are permitted and recorded, but unused pairs are preferred.

    ``strategy='hamiltonian'`` retains the legacy certificate-only search;
    it does not accept a candidate evaluator.  An already sufficient base
    relative-expansion certificate is returned unchanged in either mode.
    """
    if strategy not in ("degree_incremental", "hamiltonian"):
        raise ValueError("strategy must be 'degree_incremental' or 'hamiltonian'")
    if candidate_evaluator is not None and not callable(candidate_evaluator):
        raise ValueError("candidate_evaluator must be callable or None")
    if strategy == "hamiltonian":
        if candidate_evaluator is not None:
            raise ValueError("candidate_evaluator is supported only by degree_incremental")
        return _enhance_hamiltonian(
            b, ports=ports, distance=distance, max_degree=max_degree, seed=seed,
            attempts=attempts, exact_limit=exact_limit,
            certificate_limit=certificate_limit, max_entries=max_entries,
            allow_partial_overlay=allow_partial_overlay)
    max_degree = _integer(max_degree, "max_degree")
    seed = _integer(seed, "seed", minimum=0)
    attempts = _integer(attempts, "attempts")
    exact_limit = _integer(exact_limit, "exact_limit", maximum=24)
    certificate_limit = _integer(certificate_limit, "certificate_limit")
    max_entries = _integer(max_entries, "max_entries")
    if not isinstance(allow_partial_overlay, (bool, np.bool_)):
        raise ValueError("allow_partial_overlay must be a boolean")
    base = graph_matrix(b, max_entries=max_entries)
    n = base.shape[1]
    original_degrees = np.asarray(base.sum(axis=0), dtype=np.int64)
    initial_degree = int(original_degrees.max())
    if initial_degree > max_degree:
        raise ExpansionConstructionError(
            f"Input graph degree {initial_degree} exceeds max_degree={max_degree}",
            reason="degree_budget", details=dict(initial_max_degree=initial_degree,
                                                  max_degree=max_degree))
    initial = certify_expansion(
        base, ports=ports, distance=distance, exact_limit=exact_limit,
        certificate_limit=certificate_limit, max_entries=max_entries)
    history = []
    rng = np.random.default_rng(seed)
    last_certificate = initial
    matrix_limited = False

    def unevaluated():
        return dict(condition_met=False, exact=False, beta=None, lower_bound=0.0,
                    global_h_lower_bound=None, global_h_certified=False,
                    strict_global_h=False, expansion_strict=False,
                    expansion_evaluated=False, expansion_requirement="global h > 1",
                    certificate="global expansion not evaluated")

    def finish(matrix, certificate, context, evaluation, acceptance):
        degrees = np.asarray(matrix.sum(axis=0), dtype=np.int64)
        added = [list(map(int, np.flatnonzero(row))) for row in matrix[len(base):]]
        occupied = {tuple(map(int, np.flatnonzero(row))) for row in base}
        parallel = []
        for index, edge in enumerate(added, start=len(base)):
            if tuple(edge) in occupied:
                parallel.append(index)
            occupied.add(tuple(edge))
        report = dict(certificate)
        report.update(
            initial=initial, initial_edges=int(len(base)), added_edges=added,
            added_edge_count=len(added), initial_max_degree=initial_degree,
            final_max_degree=int(degrees.max()), final_min_degree=int(degrees.min()),
            final_degrees=degrees.tolist(), regular=bool(np.all(degrees == degrees[0])),
            degree_deficits=(context["stage_degree"] - degrees).tolist(),
            max_degree=max_degree, seed=seed, candidate_seed=context.get("candidate_seed"),
            attempts_used=len(history), attempts_per_degree=attempts,
            stage_degree=context["stage_degree"], overlay_layers=0,
            augmentation_strategy=("degree_incremental_worst_cut_repair"
                                   if context["stage"] == "worst_cut_repair"
                                   else "degree_incremental_random_matching"),
            strategy="degree_incremental", acceptance=acceptance,
            candidate_evaluation=_plain(evaluation), search_history=_plain(history),
            partial_edge_steps=_plain(context.get("targeted_steps", [])),
            parallel_edges_allowed=True, added_parallel_edge_indices=parallel,
            construction="seeded random residual-degree matching and optional exact-cut repair",
            source="implementation heuristic; random regular graph expansion is motivation, not a per-candidate proof",
            implementation_choice="distance first, then strict global h; increasing bounded degree and seeded retries")
        return matrix, _plain(report)

    def evaluate(matrix, context):
        nonlocal last_certificate
        evaluation = (dict(distance_state="unknown", eligible=True)
                      if candidate_evaluator is None else candidate_evaluator(matrix.copy(), dict(context)))
        if not isinstance(evaluation, dict):
            raise ValueError("candidate_evaluator must return a dictionary")
        if evaluation.get("distance_state") not in ("adequate", "insufficient", "unknown"):
            raise ValueError("candidate_evaluator distance_state must be adequate, insufficient, or unknown")
        if not isinstance(evaluation.get("eligible"), (bool, np.bool_)):
            raise ValueError("candidate_evaluator eligible must be a boolean")
        certificate = unevaluated()
        record = dict(context, evaluation=_plain(evaluation),
                      graph_sha256=_digest(matrix),
                      added_edges=[list(map(int, np.flatnonzero(row))) for row in matrix[len(base):]],
                      final_degrees=matrix.sum(axis=0).astype(int).tolist())
        if not evaluation["eligible"]:
            record.update(acceptance=None, certificate=certificate)
            history.append(record)
            return None, certificate
        if evaluation["distance_state"] == "adequate":
            record.update(acceptance="distance", certificate=certificate)
            history.append(record)
            return finish(matrix, certificate, context, evaluation, "distance"), certificate
        certificate = _strict_global_certificate(
            matrix, exact_limit=exact_limit, certificate_limit=certificate_limit,
            max_entries=max_entries)
        last_certificate = certificate
        record.update(certificate=certificate, acceptance=None)
        history.append(record)
        if certificate["condition_met"]:
            if evaluation["distance_state"] == "insufficient":
                raise ExpansionConstructionError(
                    "A candidate distance counterexample contradicts the strict global expansion certificate; "
                    "check the candidate-code construction and distance witness",
                    reason="distance_certificate_conflict",
                    details=dict(certificate=certificate, evaluation=_plain(evaluation),
                                 context=_plain(context)))
            record["acceptance"] = "expansion"
            return finish(matrix, certificate, context, evaluation, "expansion"), certificate
        return None, certificate

    base_context = dict(stage="base", stage_degree=initial_degree,
                        candidate_seed=None, attempt=0, targeted_steps=[])
    if initial["condition_met"]:
        preserved = dict(initial, expansion_evaluated=True, strict_global_h=False,
                         expansion_strict=False)
        return finish(base, preserved, base_context, None, "existing_certificate")

    # A candidate with an exact worst cut remains a repair starting point at
    # the next cap.  Its original seed and graph are retained, not regenerated.
    frontier = None

    def better_frontier(current, matrix, certificate, context):
        if not certificate.get("exact") or "worst_cut" not in certificate:
            return current
        candidate = (matrix.copy(), certificate, dict(context))
        if current is None:
            return candidate
        previous_matrix, previous_certificate, _ = current
        a, ad = certificate["boundary_weight"], certificate["denominator"]
        bnum, bd = previous_certificate["boundary_weight"], previous_certificate["denominator"]
        if a * bd > bnum * ad or (a * bd == bnum * ad and len(matrix) < len(previous_matrix)):
            return candidate
        return current

    def repair_start(start, cap, cap_frontier):
        nonlocal matrix_limited
        starting_matrix, starting_certificate, starting_context = start
        matrix = starting_matrix.copy()
        certificate = starting_certificate
        degrees = np.asarray(matrix.sum(axis=0), dtype=np.int64)
        occupied = {tuple(map(int, np.flatnonzero(row))) for row in matrix}
        local_rng = np.random.default_rng(starting_context["candidate_seed"])
        steps = list(starting_context.get("targeted_steps", []))
        while certificate.get("exact") and "worst_cut" in certificate:
            inside = set(certificate["worst_cut"])
            left = [v for v in inside if degrees[v] < cap]
            right = [v for v in range(n) if v not in inside and degrees[v] < cap]
            pairs = [(u, v) for u in left for v in right]
            if not pairs:
                break
            if (len(matrix) + 1) * n > max_entries:
                matrix_limited = True
                break
            def score(pair):
                u, v = pair
                return (tuple(sorted(pair)) in occupied,
                        max(int(degrees[u]), int(degrees[v])), int(degrees[u] + degrees[v]))
            best_score = min(map(score, pairs))
            preferred = [pair for pair in pairs if score(pair) == best_score]
            u, v = preferred[int(local_rng.integers(len(preferred)))]
            row = np.zeros((1, n), dtype=np.uint8)
            row[0, [u, v]] = 1
            steps = steps + [dict(worst_cut=list(certificate["worst_cut"]),
                                  previous_boundary=int(certificate["boundary_weight"]),
                                  denominator=int(certificate["denominator"]),
                                  added_edge=sorted((u, v)), stage_degree=cap)]
            matrix = np.vstack((matrix, row))
            degrees[[u, v]] += 1
            occupied.add(tuple(sorted((u, v))))
            context = dict(stage="worst_cut_repair", stage_degree=cap,
                           candidate_seed=starting_context["candidate_seed"],
                           attempt=starting_context["attempt"], targeted_steps=steps)
            accepted, certificate = evaluate(matrix, context)
            if accepted is not None:
                return accepted, cap_frontier
            cap_frontier = better_frontier(cap_frontier, matrix, certificate, context)
        return None, cap_frontier

    first_cap = initial_degree if np.any(original_degrees < initial_degree) else initial_degree + 1
    for cap in range(max(1, first_cap), max_degree + 1):
        previous_frontier = frontier
        cap_frontier = None
        # Preserve the best lower-degree seed and first spend only the slots
        # needed to repair its exact bad cuts.  Full regularization can then
        # be retried if those sparse additions do not suffice.
        if allow_partial_overlay and previous_frontier is not None:
            accepted, cap_frontier = repair_start(previous_frontier, cap, cap_frontier)
            if accepted is not None:
                return accepted
        for attempt in range(1, attempts + 1):
            candidate_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            try:
                matrix = _degree_matching(base, cap, np.random.default_rng(candidate_seed), max_entries)
            except ExpansionConstructionError as error:
                if error.reason != "matrix_budget":
                    raise
                matrix_limited = True
                break
            if len(matrix) == len(base):
                continue
            context = dict(stage="random_matching", stage_degree=cap,
                           candidate_seed=candidate_seed, attempt=attempt, targeted_steps=[])
            accepted, certificate = evaluate(matrix, context)
            if accepted is not None:
                return accepted
            cap_frontier = better_frontier(cap_frontier, matrix, certificate, context)
        if allow_partial_overlay and cap_frontier is not None:
            accepted, cap_frontier = repair_start(cap_frontier, cap, cap_frontier)
            if accepted is not None:
                return accepted
        frontier = cap_frontier if cap_frontier is not None else previous_frontier

    proof = last_certificate.get("spectral_proof", {})
    resource_limited = bool(proof.get("resource_limited"))
    reason = ("matrix_budget" if matrix_limited else "certificate_budget" if resource_limited
              else "degree_budget" if not history else "search_budget")
    raise ExpansionConstructionError(
        "No acceptable distance or strict global expansion certificate found within "
        "the fixed degree, sparsity, attempt, and resource budgets",
        reason=reason,
        details=_plain(dict(last_certificate=last_certificate, candidates_tried=len(history),
                           max_degree=max_degree, seed=seed, attempts_per_degree=attempts,
                           search_history=history)))
