"""Resource-bounded, exact irreducible-cluster distance certification.

Grow a partial Pauli at its first unsatisfied stabilizer check, adding an
anticommuting one-site Pauli on unused support.  This search is complete for
minimum-weight logical representatives, including for degenerate codes:
such a representative has no proper nonempty zero-syndrome restriction.
A zero-syndrome restriction would either be a shorter logical or a
stabilizer that can be removed to obtain a shorter logical.  Every proper
partial restriction therefore has an unsatisfied check, and some remaining
site of the representative anticommutes with that check.

The budget counts visited, distinct partial Pauli assignments *per target
weight*, including seeds.  These are search nodes, not just completed
zero-syndrome candidates.  Repeated nodes at a later target weight count
again.  No lower-bound increment is allowed for an interrupted weight.
"""
from __future__ import annotations

from copy import deepcopy


def cluster_distance(check_x, check_z, initial, *, max_candidates=2_000_000):
    """Certify distance from validated checks and a valid initial report.

    ``initial`` normally comes from the zero-enumeration baseline and may
    carry an improved upper-bound witness from a random information-set
    search.  It is not mutated.  CSS matrices use separate pure X/Z searches;
    mixed checks retain all three one-site Paulis and symplectic weight.
    The stabilizer row space is always the full final-code row space.
    """
    # Import lazily: distance.py owns validation and the public dispatcher.
    from .distance import _basis, _bits, _contains, _integer, _witness

    max_candidates = _integer(max_candidates, "max_candidates")
    report = deepcopy(initial)
    n = report["n"]
    xs = [_bits(row) for row in check_x]
    zs = [_bits(row) for row in check_z]
    css = all(not a or not b for a, b in zip(xs, zs))
    report.update(
        method="css_irreducible_cluster" if css else "symplectic_irreducible_cluster",
        max_candidates=max_candidates,
        checked_candidates=0,
        candidate_count_kind="unique_partial_pauli_assignments_per_weight_limit",
    )
    if report["status"] in ("exact", "no_logicals"):
        return report
    report.update(distance=None, stopped_at_weight=None, stop_reason=None)
    stabilizers = _basis(a | (b << n) for a, b in zip(xs, zs))
    sx = [_bits(check_z[:, q]) for q in range(n)]
    sz = [_bits(check_x[:, q]) for q in range(n)]

    # A local option is (site, binary symplectic Pauli, syndrome).  Selecting
    # only options anticommuting with the first unsatisfied check guarantees
    # that each growth step connects to the existing syndrome cluster.
    if css:
        sectors = [
            [(q, 1 << q, sx[q]) for q in range(n)],
            [(q, 1 << (n + q), sz[q]) for q in range(n)],
        ]
    else:
        sectors = [[option for q in range(n) for option in (
            (q, 1 << q, sx[q]),
            (q, 1 << (n + q), sz[q]),
            (q, (1 << q) | (1 << (n + q)), sx[q] ^ sz[q]),
        )]]
    options_by_check = [
        [[option for option in options if (option[2] >> row) & 1]
         for row in range(len(xs))]
        for options in sectors
    ]

    def seeds():
        for sector, options in enumerate(sectors):
            for q, value, syndrome in options:
                yield value, syndrome, 1 << q, 1, q, sector

    def children(value, syndrome, support, depth, root, sector):
        row = (syndrome & -syndrome).bit_length() - 1
        for q, local_value, local_syndrome in options_by_check[sector][row]:
            # Every support has a unique smallest site, which is its seed.
            # Growth order of its other sites is unrestricted.
            if q > root and not (support >> q) & 1:
                yield (value | local_value, syndrome ^ local_syndrome,
                       support | (1 << q), depth + 1, root, sector)

    limit = min(report["max_weight"], report["upper_bound"] - 1)
    first_weight = max(1, report["proven_lower_bound"])
    for weight in range(first_weight, limit + 1):
        seen = set()
        stack = [iter(seeds())]
        while stack:
            node = next(stack[-1], None)
            if node is None:
                stack.pop()
                continue
            value, syndrome, support, depth, root, sector = node
            if value in seen:
                continue
            if report["checked_candidates"] == max_candidates:
                report.update(
                    status="bounded" if report["proven_lower_bound"] > 1 else "unknown",
                    stopped_at_weight=weight, stop_reason="candidate_budget",
                )
                return report
            seen.add(value)
            report["checked_candidates"] += 1
            if syndrome == 0:
                if not _contains(value, stabilizers):
                    report.update(
                        status="exact", distance=depth, proven_lower_bound=depth,
                        upper_bound=depth, witness=_witness(value, n),
                        stopped_at_weight=weight, stop_reason="logical_witness_found",
                    )
                    return report
                # A minimum logical cannot contain this stabilizer as a
                # proper local restriction.  Its supersets need not grow.
                continue
            if depth < weight:
                stack.append(iter(children(*node)))
        report["completed_weight"] = weight
        report["proven_lower_bound"] = weight + 1

    if report["proven_lower_bound"] == report["upper_bound"]:
        report.update(status="exact", distance=report["upper_bound"], stop_reason="bounds_meet")
    else:
        report.update(status="bounded", stop_reason="weight_limit")
    return report


__all__ = ["cluster_distance"]
