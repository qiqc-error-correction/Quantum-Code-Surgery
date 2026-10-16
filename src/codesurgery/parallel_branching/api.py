"""Balanced branching trees, separate from all measurement constructors."""
from __future__ import annotations

import numpy as np

from .construction import append_branch, lift_paulis
from .models import BranchedCode, equivalence_witnesses
from .optimization import optimize_branching_targets
from .policy import plan_branching, BranchingBudgetError
from .direct_attachment import preview_direct_attachment
from .targets import prepare_targets


def branching_deform(code, targets=None, *, logical_paulis=None, pauli="Z",
                     logical_indices=None, logical_basis=None, representatives="auto",
                     leaf_pauli="auto", force_copy=False, max_entries=50_000_000,
                     optimize_logical=True, logical_optimization_options=None,
                     branching_mode="adaptive", max_check_increase=2, max_qubit_increase=2):
    """Route compatible logical targets directly or through branching stickers.

    The input is a CSSCode. Omit targets/basis to select all standard-form Z
    logicals. ``targets`` are signed physical IXYZ strings of length n;
    ``logical_paulis`` instead specifies logical strings of length k.
    Representatives must be qubit-wise commuting, possibly after a signed
    stabilizer-equivalent rewrite in the selected logical basis (auto mode).

    Support reduction is enabled by default, using a seeded heuristic when
    weight > 2*code.distance. The earlier threshold rule skips unknown
    distance; set logical_optimization_options={"threshold": None} to always
    try. Candidate representatives must retain QWC and cannot increase the
    selected routing's ancilla count. Set optimize_logical=False to disable this stage.
    ``representatives`` controls the initial basis rewrite independently;
    exact physical preservation needs both preserve mode and optimization off.

    Adaptive mode leaves targets on memory whenever direct paired-boundary
    gauging would respect each original check's row-weight increase and each
    original physical qubit's degree increase (both default to 2). Otherwise
    it groups targets into branches and rechecks cumulative source growth.
    The remaining direct representatives may overlap while remaining QWC.
    ``always`` reproduces the previous full binary tree; force_copy forces a
    single target onto a sticker. New auxiliary weights are reported separately.

    The returned core preserves every encoded qubit and performs no measurement.
    A separate measurement_preview contains actual paired-graph matrices for
    budgeting only; it has no expansion or distance certificate. All original
    checks are retained, including redundancy. No distance solver runs implicitly;
    the branching core retains its weight-nonincreasing retraction certificate.
    """
    if not isinstance(force_copy, (bool, np.bool_)):
        raise ValueError("force_copy must be a boolean")
    if (isinstance(max_entries, (bool, np.bool_))
            or not isinstance(max_entries, (int, np.integer)) or max_entries < 1):
        raise ValueError("max_entries must be a positive integer")
    if not isinstance(leaf_pauli, str) or leaf_pauli.upper() not in ("AUTO", "X", "Z"):
        raise ValueError("leaf_pauli must be auto, X, or Z")
    leaf_pauli = leaf_pauli.upper()
    p = prepare_targets(code, targets, logical_paulis=logical_paulis, pauli=pauli,
                        logical_indices=logical_indices, logical_basis=logical_basis,
                        representatives=representatives)
    plan_cache = {}

    def planned(x, z):
        key = (x.tobytes(), z.tobytes())
        if key not in plan_cache:
            plan_cache[key] = plan_branching(p["source_check_x"], p["source_check_z"], x, z,
                branching_mode=branching_mode, max_check_increase=max_check_increase,
                max_qubit_increase=max_qubit_increase, force_copy=force_copy)
        return plan_cache[key]

    # Validate options before optimization, but an infeasible representative
    # is not an impossibility proof for its stabilizer equivalence class.
    # For example, a zero row-growth budget can become feasible after cleaning
    # the target away from every locally anticommuting check.
    try:
        planned(p["target_x"], p["target_z"])
    except BranchingBudgetError:
        pass

    def route_cost(x, z):
        try:
            return planned(x, z)["ancillas"]
        except BranchingBudgetError:
            return None

    selected_x, selected_z, selected_s, optimization = optimize_branching_targets(
        code, p, enabled=optimize_logical, options=logical_optimization_options,
        force_copy=force_copy, cost_function=route_cost)
    try:
        routing = planned(selected_x, selected_z)
    except BranchingBudgetError as error:
        stage = "after the configured support-optimization attempt" if optimize_logical else "with support optimization disabled"
        raise BranchingBudgetError(
            f"The current prepared representatives have no feasible branching route {stage}. "
            "This is not an impossibility proof for their logical classes; try different "
            "stabilizer-equivalent representatives or relax the growth limits. " + str(error)
        ) from error
    optimization["cost_model"] = "selected_" + branching_mode + "_routing"
    cx, cz, cs = p["source_check_x"], p["source_check_z"], p["source_check_signs"]
    bx = np.vstack((p["source_jx_x"], p["source_jz_x"]))
    bz = np.vstack((p["source_jx_z"], p["source_jz_z"]))
    bs = np.concatenate((p["source_jx_signs"], p["source_jz_signs"]))
    tx, tz, ts = selected_x.copy(), selected_z.copy(), selected_s.copy()
    ox, oz, os = p["original_target_x"].copy(), p["original_target_z"].copy(), p["original_target_signs"].copy()
    source = dict(check_x=cx.copy(), check_z=cz.copy(), check_signs=cs.copy(),
        basis_x=bx.copy(), basis_z=bz.copy(), basis_signs=bs.copy(),
        target_x=ox.copy(), target_z=oz.copy(), target_signs=os.copy(),
        prepared_x=tx.copy(), prepared_z=tz.copy(), prepared_signs=ts.copy(),
        pre_optimization_x=p["target_x"].copy(), pre_optimization_z=p["target_z"].copy(),
        pre_optimization_signs=p["target_signs"].copy(),
        initial_canonicalization_witnesses=p["canonicalization_witnesses"].copy(),
        support_reduction_witnesses=equivalence_witnesses(cx, cz, cs,
            p["target_x"], p["target_z"], p["target_signs"], tx, tz, ts),
        canonicalization_witnesses=equivalence_witnesses(cx, cz, cs, ox, oz, os, tx, tz, ts))
    branches = []

    def add(group, depth, parent):
        nonlocal cx, cz, cs, bx, bz, bs, tx, tz, ts, ox, oz, os
        lx = np.bitwise_or.reduce(tx[group], axis=0)
        lz = np.bitwise_or.reduce(tz[group], axis=0)
        # The target preparation establishes QWC. Recheck at every node
        # rather than treating global commutation as local compatibility.
        if np.any((tx[group] * lz) ^ (tz[group] * lx)):
            raise ValueError("Targets are not locally compatible at a branching node")
        axis = ("X" if not np.any(lz) else "Z") if leaf_pauli == "AUTO" else leaf_pauli
        branch = append_branch(cx, cz, cs, lx, lz, leaf_pauli=axis, max_entries=max_entries)
        mask = (tx[group][:, branch["support"]] | tz[group][:, branch["support"]]).copy()
        if np.any((branch["F"] @ mask.T) % 2):
            raise ValueError("A requested leaf would not commute with the terminal caps")
        bx, bz, bs = lift_paulis(bx, bz, bs, branch)
        ox, oz, os = lift_paulis(ox, oz, os, branch)
        tx, tz, ts = lift_paulis(tx, tz, ts, branch)
        tx[group], tz[group] = 0, 0
        out = tz if axis == "Z" else tx
        out[np.ix_(group, branch["leaf_columns"])] = mask
        cx, cz, cs = branch["check_x"], branch["check_z"], branch["check_signs"]
        desc = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in branch.items() if key not in ("check_x", "check_z", "check_signs")}
        desc.update(id=len(branches), parent=parent, depth=depth, targets=list(group))
        branches.append(desc)
        return desc["id"]

    def split(group, depth=1, parent=None):
        mid = len(group) // 2
        for child in (group[:mid], group[mid:]):
            branch_id = add(child, depth, parent)
            if len(child) > 1:
                split(child, depth + 1, branch_id)

    for group in routing["root_groups"]:
        branch_id = add(group, 1, None)
        if len(group) > 1:
            split(group, 2, branch_id)
    if cx.shape[1] - code.n != optimization["after_ancillas"]:
        raise RuntimeError("Actual branching size disagrees with the selected routing cost")
    witness = equivalence_witnesses(cx, cz, cs, ox, oz, os, tx, tz, ts)
    metadata = dict(scheme="Parallel Logical Measurements: branching", paper="https://arxiv.org/abs/2503.05003v3",
        stage="branching_only", logical_measurement_performed=False,
        basis_source=p["basis_source"], input_mode=p["input_mode"],
        representation_action=p["representation_action"], logical_signatures=p["logical_signatures"].tolist(),
        support_optimization=optimization,
        target_logical_rank=p["target_logical_rank"], source_name=code.name,
        new_check_sign_convention="+1 reference stabilizer sector; measured syndrome outcomes require Pauli-frame tracking",
        static_distance_certificate={"relation": "d_output >= d_input",
            "method": "weight-nonincreasing logical retraction at every two-layer sticker",
            "input_distance_supplied": code.distance,
            "numeric_input_distance_verified_here": False},
        circuit_distance_certified=False,
        protocol_note="Static matrices only. Fault-tolerant syndrome schedule and subsequent measurements are separate.")
    preview = preview_direct_attachment(cx, cz, tx, tz, check_signs=cs, target_signs=ts,
                                        max_entries=max_entries)
    result = BranchedCode(cx, cz, cs, source, bx, bz, bs, ox, oz, os, tx, tz, ts,
                         witness, branches, metadata, int(max_entries), routing, preview)
    result.metadata["validation"] = result.validate()
    return result


__all__ = ["branching_deform"]
