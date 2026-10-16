"""Public devised-sticking interface, independent of the CKBB construction."""

import numpy as np

from ..gf2 import binary_matrix, binary_vector, matmul, nullspace, rank, solve
from ..gadget_common import logical_basis_for, signature
from ..models import CSSCode
from ..symplectic import parse_pauli
from .construction import assemble_deformed, protected_logical_generators, validate_readout
from .glue import build_devised_glue
from .models import DevisedDeformedCode


def _targets(code, logicals, pauli, logical_index):
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
        raise ValueError("A single devised-sticking measurement sticker supports same-type X or Z targets; general Y/ZX requires the separate Appendix I protocol")
    pauli = pauli.upper()
    if logicals is None:
        return code.logical(pauli, logical_index)[None, :], np.zeros(1, dtype=np.uint8), pauli
    if isinstance(logicals, str):
        items = [logicals]
    else:
        try:
            items = list(logicals)
        except TypeError as error:
            raise ValueError("logicals must be a physical Pauli string, binary vector, or collection of them") from error
        if not items:
            raise ValueError("At least one logical observable is required")
        if not any(isinstance(item, str) for item in items):
            array = np.asarray(items)
            if array.ndim == 1:
                items = [array]
    vectors, signs = [], []
    for item in items:
        if isinstance(item, str):
            x, z, sign = parse_pauli(item, code.n)
            if np.any(x) and np.any(z):
                raise ValueError("A single devised-sticking sticker does not accept mixed Y/ZX; use the separate Appendix I protocol")
            if (pauli == "Z" and np.any(x)) or (pauli == "X" and np.any(z)):
                raise ValueError(f"All targets must be pure {pauli}; specify pauli='{ 'X' if pauli == 'Z' else 'Z' }' for the opposite type")
            vector = z if pauli == "Z" else x
        else:
            vector = binary_vector(item, length=code.n, name="logical")
            sign = 0
        vectors.append(vector)
        signs.append(sign)
    targets = binary_matrix(vectors, n_cols=code.n, name="logicals")
    opposite = code.hx if pauli == "Z" else code.hz
    dual = code.standard_logicals().jx if pauli == "Z" else code.standard_logicals().jz
    if np.any(matmul(targets, opposite.T)):
        raise ValueError("Every requested observable must commute with the memory checks")
    signatures = matmul(targets, dual.T)
    if np.any(~np.any(signatures, axis=1)):
        raise ValueError("Identity and stabilizer observables are not nontrivial logical targets")
    if rank(signatures) != len(targets):
        raise ValueError("Requested logical observables must be independent modulo memory stabilizers")
    return targets, np.asarray(signs, dtype=np.uint8), pauli


def _select_representatives(code, targets, target_signs, pauli, *, enabled, options):
    """Retain one signed, checkable stabilizer-coset record per input row."""
    from ..logical_optimization import optimize_prepared_logical

    basis = logical_basis_for(code)
    original_x = targets.copy() if pauli == "X" else np.zeros_like(targets)
    original_z = targets.copy() if pauli == "Z" else np.zeros_like(targets)
    selected, selected_signs, reports = [], [], []
    for x, z, sign in zip(original_x, original_z, target_signs):
        original = (x, z, int(sign))
        if enabled:
            chosen, report = optimize_prepared_logical(code, original, basis, **(options or {}))
        else:
            chosen = (x.copy(), z.copy(), int(sign))
            weight = int(np.count_nonzero(x | z))
            logical_signature = signature(x, z, basis).tolist()
            report = dict(
                status="disabled", method=None,
                objective="Pauli support weight: count(x OR z)",
                coset="original Pauli times original positive CSS stabilizers",
                search_kind=pauli, original_x=x.tolist(), original_z=z.tolist(),
                original_sign=int(sign), x=x.tolist(), z=z.tolist(), sign=int(sign),
                before_weight=weight, after_weight=weight,
                hx_coefficients=[0] * len(code.hx), hz_coefficients=[0] * len(code.hz),
                original_signature=logical_signature, optimized_signature=logical_signature.copy(),
                optimal=False, equivalence_verified=True, signature_verified=True,
                sign_convention="(-1)^sign times literal tensor I/X/Y/Z",
            )
        report["enabled"] = bool(enabled)
        selected.append(chosen[0] if pauli == "X" else chosen[1])
        selected_signs.append(chosen[2])
        reports.append(report)
    return (binary_matrix(selected, n_cols=code.n, name="selected_logicals"),
            np.asarray(selected_signs, dtype=np.uint8), original_x, original_z, reports)


def devised_deform(code, logicals=None, *, pauli="Z", logical_index=0, d_R=None, ldpc=True,
                   max_entries=50_000_000, optimize_logical=False,
                   logical_optimization_options=None):
    """Construct one finely devised sticker measuring a same-type logical set.

    ``logicals`` is a physical I/X/Z string, a binary vector, a row matrix,
    or a list of signed strings. Omitted targets use one paired standard-form
    generator. Simultaneous targets may overlap, but must be independent
    modulo the memory stabilizers. ``d_R`` is the repetition length, not the
    total number of alternating Tanner layers. No CKBB builder is called.

    By default, every supplied physical representative is used unchanged.
    ``optimize_logical=True`` first searches each target's own stabilizer coset
    for a lighter representative, preserving its signed logical class and
    input order. The default threshold is strictly ``weight > 2 * distance``;
    an unknown distance skips this search. Set
    ``logical_optimization_options={"threshold": None}`` to always attempt it.
    The original and selected signed operators, weights, and stabilizer
    coefficient witnesses are retained in ``metadata["logical_optimization"]``
    (one report per target), including when optimization is disabled.
    Individual weight reduction does not guarantee a smaller total sticker.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    if not isinstance(ldpc, (bool, np.bool_)):
        raise ValueError("ldpc must be a boolean")
    if not isinstance(optimize_logical, (bool, np.bool_)):
        raise ValueError("optimize_logical must be a Boolean")
    if logical_optimization_options is not None and not isinstance(logical_optimization_options, dict):
        raise ValueError("logical_optimization_options must be a dictionary")
    if logical_optimization_options is not None and not optimize_logical:
        raise ValueError("logical_optimization_options requires optimize_logical=True")
    targets, target_signs, pauli = _targets(code, logicals, pauli, logical_index)
    if targets.size > max_entries:
        raise ValueError("Target matrix exceeds max_entries")
    if d_R is None:
        d_R = code.distance if code.distance is not None else code.n
    if isinstance(d_R, (bool, np.bool_)) or not isinstance(d_R, (int, np.integer)) or d_R < 1:
        raise ValueError("d_R must be a positive integer")
    d_R = int(d_R)
    input_signs = target_signs.copy()
    targets, target_signs, input_x, input_z, optimization = _select_representatives(
        code, targets, target_signs, pauli, enabled=bool(optimize_logical),
        options=logical_optimization_options)
    glue = build_devised_glue(code, targets, pauli=pauli, ldpc=bool(ldpc), max_entries=max_entries)
    hx, hz, layout = assemble_deformed(code, glue, pauli=pauli, d_R=d_R, max_entries=max_entries)
    readout = validate_readout(code, hx, hz, targets, pauli=pauli, max_entries=max_entries)
    n = hx.shape[1]
    if 2 * (len(hx) + len(hz)) * n > max_entries:
        raise ValueError("Deformed symplectic checks exceed max_entries")
    cx = np.vstack((hx, np.zeros_like(hz)))
    cz = np.vstack((np.zeros_like(hx), hz))
    same_checks = hz if pauli == "Z" else hx
    target_checks = np.pad(targets, ((0, 0), (0, n - code.n)))
    same_witnesses = solve(same_checks.T, target_checks.T).T
    witnesses = np.zeros((len(targets), len(cx)), dtype=np.uint8)
    if pauli == "Z":
        witnesses[:, len(hx):] = same_witnesses
    else:
        witnesses[:, :len(hx)] = same_witnesses
    # Preserve positive signs of the original memory checks and keep every
    # stabilizer dependency positive, while matching signed input targets.
    if len(cx) ** 2 > max_entries:
        raise ValueError("Stabilizer dependency coefficients exceed max_entries")
    dependencies = nullspace(np.hstack((cx, cz)).T)
    memory_rows = list(range(len(code.hx))) + list(range(len(hx), len(hx) + len(code.hz)))
    memory_constraints = np.zeros((len(memory_rows), len(cx)), dtype=np.uint8)
    memory_constraints[np.arange(len(memory_rows)), memory_rows] = 1
    constraints = np.vstack((witnesses, dependencies, memory_constraints))
    if constraints.size > max_entries:
        raise ValueError("Signed witness constraints exceed max_entries")
    rhs = np.concatenate((target_signs, np.zeros(len(dependencies) + len(memory_constraints), dtype=np.uint8)))
    check_signs = solve(constraints, rhs)
    measurement_x = targets.copy() if pauli == "X" else np.zeros_like(targets)
    measurement_z = targets.copy() if pauli == "Z" else np.zeros_like(targets)
    protected_x, protected_z, protected_information = protected_logical_generators(
        code, glue, targets, pauli=pauli, d_R=d_R, max_entries=max_entries)
    lower_bound = min(code.distance, d_R) if code.distance is not None else None
    metadata = {
        "construction": "Zhang--Li finely devised glue + measurement sticker (Eqs. 45/46)",
        "pauli": pauli, "d_R": d_R, "ldpc": bool(ldpc), "max_entries": int(max_entries),
        "optimize_logical": bool(optimize_logical),
        "input_measurement_x": input_x, "input_measurement_z": input_z,
        "input_measurement_signs": input_signs,
        "logical_optimization": optimization,
        "distance_lower_bound": lower_bound,
        "distance_preservation_guaranteed": code.distance is not None and d_R >= code.distance,
        "distance_note": "Protected subsystem distance min(d/|S|,d_R), with |S|=1; supplied memory distance is assumed, not computed",
        "layout": layout, "protected_logicals": protected_information, **readout,
    }
    result = DevisedDeformedCode(hx, hz, cx, cz, check_signs, measurement_x, measurement_z,
                                target_signs, witnesses, code.n, code.k - len(targets),
                                protected_x, protected_z, glue, code, metadata)
    result.validate()
    return result
