"""Random-information-set upper bounds, following the QDistRnd algorithm.

Reference: https://qec-pages.github.io/QDistRnd/doc/chap3.html .  Gaussian
elimination chooses a random pivot-column order, keeping bit coordinates in
their original order throughout. This is equivalent to permute/RREF/unpermute.
For mixed codes the permutation acts on all 2n binary columns, and physical
weight is always evaluated after restoring the original paired coordinates.
"""
from copy import deepcopy

import numpy as np


def _random_rref(rows, order):
    rows = list(rows)
    pivot = 0
    for column in order:
        bit = 1 << int(column)
        found = next((i for i in range(pivot, len(rows)) if rows[i] & bit), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        selected = rows[pivot]
        for i in range(len(rows)):
            if i != pivot and rows[i] & bit:
                rows[i] ^= selected
        pivot += 1
        if pivot == len(rows):
            break
    return rows[:pivot]


def ris_distance(check_x, check_z, initial, *, trials=64, seed=0):
    """Improve only the upper bound; no statistical estimate is a certificate."""
    from .distance import _bits, _basis, _contains, _kernel_rows, _witness, _integer
    trials, seed = _integer(trials, "ris_trials"), _integer(seed, "seed")
    report = deepcopy(initial)
    n = report["n"]
    xs, zs = list(map(_bits, check_x)), list(map(_bits, check_z))
    stabilizers = _basis(a | (b << n) for a, b in zip(xs, zs))
    css = all(not a or not b for a, b in zip(xs, zs))
    if css:
        spaces = [(list(_kernel_rows(zs, n)), n, 0),
                  (list(_kernel_rows(xs, n)), n, n)]
    else:
        spaces = [(list(_kernel_rows((b | (a << n) for a, b in zip(xs, zs)), 2*n)), 2*n, 0)]
    rng = np.random.default_rng(seed)
    stats = dict(trials_requested=trials, trials_completed=0, candidate_rows=0,
                 logical_candidates=0, improvements=0, seed=seed,
                 guarantee="upper bound only; lower bound unchanged",
                 initial_upper_bound=report["upper_bound"])
    report["ris"] = stats
    for _ in range(trials):
        for rows, width, shift in spaces:
            for value in _random_rref(rows, rng.permutation(width)):
                value <<= shift
                stats["candidate_rows"] += 1
                if _contains(value, stabilizers):
                    continue
                stats["logical_candidates"] += 1
                witness = _witness(value, n)
                if witness["weight"] < report["upper_bound"]:
                    report.update(upper_bound=witness["weight"], witness=witness)
                    stats["improvements"] += 1
        stats["trials_completed"] += 1
        if report["upper_bound"] == report["proven_lower_bound"]:
            report.update(status="exact", distance=report["upper_bound"], stop_reason="bounds_meet")
            break
    return report
