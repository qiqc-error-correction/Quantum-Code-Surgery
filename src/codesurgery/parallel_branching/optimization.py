"""Reduce branching supports within fixed signed logical classes.

The incumbent compatible representatives are always available as a fallback.
Reducing each target independently is not enough for mixed Pauli targets:
compatibility and the actual tree's ancilla count are checked before acceptance.
"""
from __future__ import annotations

import numpy as np

from ..logical_optimization import optimize_prepared_logical, verify_logical_optimization
from .targets import qubitwise_commuting


def branching_ancilla_count(check_x, check_z, x, z, *, force_copy=False):
    """Exact sticker size for the current balanced tree, without building it.

    Each descendant sees its parent's cap checks restricted to its union;
    these retain the corresponding original check-row identities. For QWC
    representatives, sibling junctions commute locally with their shared
    axes, so they add no incident rows. Count every original row, including
    redundancy, and sum support size plus incident-row count at each node.
    """
    if not qubitwise_commuting(x, z):
        raise ValueError("Tree cost requires qubit-wise commuting representatives")

    def sticker(group):
        px = np.bitwise_or.reduce(x[group], axis=0)
        pz = np.bitwise_or.reduce(z[group], axis=0)
        support = np.flatnonzero(px | pz)
        incident = np.zeros(len(check_x), dtype=bool)
        for q in support:
            incident |= ((check_x[:, q] * pz[q]) ^ (check_z[:, q] * px[q])).astype(bool)
        return len(support) + int(np.count_nonzero(incident))

    def split(group):
        mid = len(group) // 2
        total = 0
        for child in (group[:mid], group[mid:]):
            total += sticker(child)
            if len(child) > 1:
                total += split(child)
        return total

    if len(x) == 1:
        return sticker([0]) if force_copy else 0
    return split(list(range(len(x))))


def optimize_branching_targets(code, prepared, *, enabled=True, options=None, force_copy=False,
                               cost_function=None):
    """Return selected x/z/signs and an auditable candidate-selection report."""
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("optimize_logical must be a boolean")
    if options is not None and not isinstance(options, dict):
        raise ValueError("logical_optimization_options must be a dict or None")
    if not enabled and options is not None:
        raise ValueError("logical_optimization_options requires optimize_logical=True")
    allowed = {"method", "threshold", "max_trials", "seed", "time_limit"}
    if options and set(options) - allowed:
        raise ValueError("Unknown logical_optimization_options: " + ", ".join(sorted(set(options) - allowed)))
    settings = dict(method="heuristic", threshold="2d", max_trials=128, seed=0, time_limit=10.0)
    if options is not None:
        settings.update(options)
    bx, bz = prepared["target_x"].copy(), prepared["target_z"].copy()
    bs = prepared["target_signs"].copy()
    cx, cz = prepared["source_check_x"], prepared["source_check_z"]
    count = (cost_function if cost_function is not None else
             lambda x, z: branching_ancilla_count(cx, cz, x, z, force_copy=force_copy))
    cost_before = count(bx, bz)
    weights_before = np.count_nonzero(bx | bz, axis=1).tolist()
    report = dict(enabled=bool(enabled), options=settings if enabled else None,
        before_weights=weights_before, candidate_weights=weights_before.copy(),
        after_weights=weights_before.copy(), before_ancillas=cost_before,
        candidate_ancillas=cost_before, after_ancillas=cost_before,
        selection="disabled", accepted=[False] * len(bx), targets=[],
        guarantee="Signed logical classes and QWC preserved; finite incumbent routing cost never increases. No joint optimality claim.")
    if not enabled:
        return bx, bz, bs, report

    basis = dict(jx=prepared["source_jx_x"], jz=prepared["source_jz_z"], origin=prepared["basis_source"])
    candidates_x, candidates_z, candidates_s = bx.copy(), bz.copy(), bs.copy()
    for i in range(len(bx)):
        candidate, candidate_report = optimize_prepared_logical(
            code, (bx[i], bz[i], int(bs[i])), basis, **settings)
        # Recheck the actual coefficient and phase witness at this boundary;
        # optimizer status/optimality flags are not an acceptance certificate.
        verify_logical_optimization(code, candidate, basis, candidate_report)
        candidates_x[i], candidates_z[i], candidates_s[i] = candidate
        report["targets"].append(dict(index=i, candidate_report=candidate_report, decision="unchanged"))
    changed = np.any(candidates_x != bx, axis=1) | np.any(candidates_z != bz, axis=1) | (candidates_s != bs)
    report["candidate_weights"] = np.count_nonzero(candidates_x | candidates_z, axis=1).tolist()
    compatible_batch = qubitwise_commuting(candidates_x, candidates_z)
    candidate_cost = count(candidates_x, candidates_z) if compatible_batch else None
    report["candidate_ancillas"] = candidate_cost
    if (compatible_batch and candidate_cost is not None
            and (cost_before is None or candidate_cost <= cost_before)):
        x, z, signs = candidates_x, candidates_z, candidates_s
        accepted = changed.copy()
        report["selection"] = "batch"
    else:
        x, z, signs = bx.copy(), bz.copy(), bs.copy()
        accepted = np.zeros(len(bx), dtype=bool)
        current_cost = cost_before
        # One accepted substitution can make a previously rejected candidate
        # compatible. Each target is accepted at most once, so this terminates.
        progress = True
        while progress:
            progress = False
            for i in np.flatnonzero(changed & ~accepted):
                trial_x, trial_z = x.copy(), z.copy()
                trial_x[i], trial_z[i] = candidates_x[i], candidates_z[i]
                if not qubitwise_commuting(trial_x, trial_z):
                    report["targets"][i]["decision"] = "rejected_qwc"
                    continue
                trial_cost = count(trial_x, trial_z)
                if trial_cost is None:
                    report["targets"][i]["decision"] = "rejected_attachment_budget"
                    continue
                if current_cost is not None and trial_cost > current_cost:
                    report["targets"][i]["decision"] = "rejected_ancilla_increase"
                    continue
                x, z, signs[i] = trial_x, trial_z, candidates_s[i]
                current_cost = trial_cost
                accepted[i] = True
                progress = True
        report["selection"] = "sequential"
    for i, accepted_i in enumerate(accepted):
        if accepted_i:
            report["targets"][i]["decision"] = "accepted"
    report["accepted"] = accepted.tolist()
    report["after_weights"] = np.count_nonzero(x | z, axis=1).tolist()
    report["after_ancillas"] = count(x, z)
    return x, z, signs, report


__all__ = ["branching_ancilla_count", "optimize_branching_targets"]
