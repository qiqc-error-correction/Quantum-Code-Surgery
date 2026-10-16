"""CKBB alternating layers and explicit disjoint-branch parity bridges.

Reference: Cohen, Kim, Bartlett, Brown, Sci. Adv. 8 eabn1717 (2022),
arXiv:2110.10794, Figs. 2--5. ``layers`` counts dual layers r;
there are r-1 primal layers and 2r-1 auxiliary Tanner layers in total.
"""
from types import SimpleNamespace
import numpy as np
from .gf2 import binary_vector, binary_matrix, nullspace, rank, in_rowspace, solve
from .models import CSSCode, DeformedCode
from .symplectic import parse_pauli
from .measurement import UnintendedLogicalMeasurementError, validate_single_logical_measurement


def _logical_information(kernel, s, opposite_checks, conjugate_logicals):
    """Embed Q=KS and identify logical classes by opposite-type pairing.

    For Z candidates, Q H_X^T is the syndrome and Q J_X^T the logical
    signature; for X candidates use H_Z and J_Z. The logical basis must
    be complete and paired, as supplied by CSSCode.standard_logicals().
    """
    q = (kernel @ s) % 2
    syndrome = (q @ opposite_checks.T) % 2
    if np.any(syndrome):
        raise RuntimeError("Support kernel contains an operator outside the normalizer")
    signature = (q @ conjugate_logicals.T) % 2
    return SimpleNamespace(q=q, syndrome=syndrome, signature=signature,
                           rank=rank(signature))


def _resolve_layers(code, layers):
    if layers is None:
        return code.distance if code.distance is not None else code.n
    if isinstance(layers, (bool, np.bool_)) or not isinstance(layers, (int, np.integer)) or layers < 1:
        raise ValueError("layers must be a positive integer (number of dual layers)")
    return int(layers)


def _build_css_primitive(code, logical_vector, pauli="Z", layers=None,
                         allow_stabilizer=False, max_entries=50_000_000):
    """Build the unmodified support sticker, without automatic completion.

    This is an intermediate matrix construction. Its support can contain
    additional logical modes: the public API checks the *final* readout after
    any bridge or non-CSS fusion, and rejects unwanted logical measurements.
    """
    if not isinstance(pauli, str):
        raise ValueError("pauli must be a string")
    pauli = pauli.upper()
    if pauli not in ("X", "Z"):
        raise ValueError("CSS primitive pauli must be X or Z")
    logical = binary_vector(logical_vector, length=code.n, name="logical")
    same, opposite = (code.hx, code.hz) if pauli == "X" else (code.hz, code.hx)
    if not np.any(logical):
        raise ValueError("Cannot measure the identity operator")
    if np.any((opposite @ logical) % 2):
        raise ValueError("Requested operator does not commute with the input checks")
    is_stabilizer = in_rowspace(logical, same)
    if is_stabilizer and not allow_stabilizer:
        raise ValueError("Requested operator is a stabilizer, not a nontrivial logical operator")
    r = _resolve_layers(code, layers)
    support = np.flatnonzero(logical)
    s = np.zeros((len(support), code.n), dtype=np.uint8)
    s[np.arange(len(support)), support] = 1
    incident = np.flatnonzero(np.any(opposite[:, support], axis=1))
    restriction = opposite[np.ix_(incident, support)].copy()
    embedding = np.zeros((len(opposite), len(restriction)), dtype=np.uint8)
    embedding[incident, np.arange(len(incident))] = 1
    return _assemble_css_layers(code, logical, pauli, r, restriction, embedding, s,
                                max_entries=max_entries,
                                metadata_extra={"opposite_check_indices": incident,
                                                "construction": "CKBB Fig. 2 support sticker"})


def _assemble_css_layers(code, logical_vector, pauli, layers, restriction, embedding, s,
                         *, restriction_before_bridges=None, metadata_extra=None,
                         max_entries=50_000_000):
    """Layer a given (F,B,S), including the separate-branch Fig. 3 data.

    Bridge assembly supplies block-diagonal branch restrictions, with an
    auxiliary copy per branch even when an original check touches both.
    No arbitrary kernel-completion rows are ever inserted here.
    """
    logical = binary_vector(logical_vector, length=code.n, name="logical")
    r = _resolve_layers(code, layers)
    same, opposite = (code.hx, code.hz) if pauli == "X" else (code.hz, code.hx)
    s = binary_matrix(s, n_cols=code.n, name="support selector S")
    w = len(s)
    if np.any(s.sum(axis=1) != 1) or np.any(s.sum(axis=0) > 1):
        raise ValueError("CKBB selector rows must select distinct physical qubits")
    if not np.array_equal((np.ones(w, dtype=np.uint8) @ s) % 2, logical):
        raise ValueError("Support selector must cover exactly the requested target")
    restriction = binary_matrix(restriction, n_cols=w, name="restriction F")
    c = len(restriction)
    embedding = binary_matrix(embedding, n_cols=c, name="pasting B")
    if len(embedding) != len(opposite):
        raise ValueError("Pasting matrix must have one row per original opposite check")
    if np.any((restriction @ np.ones(w, dtype=np.uint8)) % 2):
        raise ValueError("Restriction rows must commute with the target")
    if not np.array_equal((embedding @ restriction) % 2, (opposite @ s.T) % 2):
        raise RuntimeError("Restriction/pasting compatibility failed")
    before = restriction if restriction_before_bridges is None else binary_matrix(
        restriction_before_bridges, n_cols=w, name="restriction before bridges")
    kernel = nullspace(before)
    final_kernel = nullspace(restriction)
    basis = code.standard_logicals()
    conjugate_logicals = basis.jx if pauli == "Z" else basis.jz
    target_signature = (logical @ conjugate_logicals.T) % 2
    initial_information = _logical_information(kernel, s, opposite, conjugate_logicals)
    final_information = _logical_information(final_kernel, s, opposite, conjugate_logicals)
    n_out = code.n + r * c + (r - 1) * w
    same_rows, opposite_rows = len(same) + r * w, len(opposite) + (r - 1) * c
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    if 2 * (same_rows + opposite_rows) * n_out > max_entries:
        raise ValueError("Dense result exceeds max_entries; choose fewer layers or increase the limit explicitly")
    merged_same = np.zeros((same_rows, n_out), dtype=np.uint8)
    merged_opposite = np.zeros((opposite_rows, n_out), dtype=np.uint8)
    merged_same[:len(same), :code.n] = same
    merged_opposite[:len(opposite), :code.n] = opposite
    dual_cols = np.arange(code.n, code.n + r * c).reshape(r, c)
    primal_cols = np.arange(code.n + r * c, n_out).reshape(r - 1, w)
    dual_rows = np.arange(len(same), same_rows).reshape(r, w)
    if c:
        merged_opposite[np.ix_(np.arange(len(opposite)), dual_cols[0])] = embedding
    for j in range(r):
        rows = dual_rows[j]
        merged_same[np.ix_(rows, dual_cols[j])] = restriction.T
        if j == 0:
            merged_same[np.ix_(rows, np.arange(code.n))] = s
        else:
            merged_same[rows, primal_cols[j - 1]] = 1
        if j < r - 1:
            merged_same[rows, primal_cols[j]] = 1
    for j in range(1, r):
        rows = np.arange(len(opposite) + (j - 1) * c, len(opposite) + j * c)
        merged_opposite[np.ix_(rows, primal_cols[j - 1])] = restriction
        merged_opposite[rows, dual_cols[j - 1]] = 1
        merged_opposite[rows, dual_cols[j]] = 1
    if np.any((merged_same @ merged_opposite.T) % 2):
        raise RuntimeError("CKBB check commutation invariant failed")
    metadata = {
        "pauli": pauli, "layers": r, "total_auxiliary_tanner_layers": 2 * r - 1,
        "support": np.argmax(s, axis=1),
        "restriction_matrix": restriction, "pasting_matrix": embedding,
        "dual_check_rows": dual_rows, "dual_qubit_columns": dual_cols,
        "primal_qubit_columns": primal_cols, "bridge_pairs": [],
        "support_completion": "none",
        "original_same_check_rows": np.arange(len(same)),
        "original_opposite_check_rows": np.arange(len(opposite)),
        "logical_image_rank_before_bridges": initial_information.rank,
        "logical_image_rank_after_bridges": final_information.rank,
        "support_kernel_dimension_before_completion": len(kernel),
        "irreducible_support_before_completion": len(kernel) == 1,
        "logical_information_test": "paired_logical_commutation",
        "support_selector": s,
        "kernel_basis_before_completion": kernel,
        "kernel_basis": final_kernel,
        "support_normalizers_before_completion": initial_information.q,
        "support_normalizers": final_information.q,
        "normalizer_syndrome": final_information.syndrome,
        "logical_signature_before_completion": initial_information.signature,
        "logical_signature": final_information.signature,
        "target_logical_signature": target_signature,
        "conjugate_logical_generators": conjugate_logicals,
        "conjugate_logical_type": "X" if pauli == "Z" else "Z",
        "component_is_stabilizer": in_rowspace(logical, same),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    hx, hz = (merged_same, merged_opposite) if pauli == "X" else (merged_opposite, merged_same)
    return SimpleNamespace(hx=hx, hz=hz, metadata=metadata, n=n_out)


def _cycle_gauges(code, primitive, pauli):
    """Independent cycle gauges and paired conjugates; not measured checks."""
    md = primitive.metadata
    a, b = md["restriction_matrix"], md["pasting_matrix"]
    c, r = len(a), md["layers"]
    opposite = code.hz if pauli == "X" else code.hx
    merged_opposite = primitive.hz if pauli == "X" else primitive.hx
    local_cycles = nullspace(a.T)
    selected = []
    span = merged_opposite.copy()
    for cycle in local_cycles:
        embedded = np.zeros(primitive.n, dtype=np.uint8)
        embedded[md["dual_qubit_columns"][0]] = cycle
        if not in_rowspace(embedded, span):
            selected.append(cycle)
            span = np.vstack((span, embedded))
    cycles = np.array(selected, dtype=np.uint8).reshape(-1, c) if c else np.zeros((0, 0), dtype=np.uint8)
    g = len(cycles)
    gauge_opposite = np.zeros((g, primitive.n), dtype=np.uint8)
    gauge_same = np.zeros((g, primitive.n), dtype=np.uint8)
    if g:
        gauge_opposite[:, md["dual_qubit_columns"][0]] = cycles
        system = np.block([[opposite, b], [np.zeros((g, code.n), dtype=np.uint8), cycles]])
        rhs = np.vstack((np.zeros((len(opposite), g), dtype=np.uint8), np.eye(g, dtype=np.uint8)))
        solved = solve(system, rhs)
        gauge_same[:, :code.n] = solved[:code.n].T
        for j in range(r):
            gauge_same[:, md["dual_qubit_columns"][j]] = solved[code.n:].T
    gx, gz = (gauge_same, gauge_opposite) if pauli == "X" else (gauge_opposite, gauge_same)
    if np.any((gx @ primitive.hz.T) % 2) or np.any((gz @ primitive.hx.T) % 2) or not np.array_equal((gx @ gz.T) % 2, np.eye(g, dtype=np.uint8)):
        raise RuntimeError("Gauge pairing invariant failed")
    return gx, gz


def _parse_target(code, logical, pauli, logical_index):
    if not isinstance(pauli, str):
        raise ValueError("pauli must be a string")
    pauli = pauli.upper()
    if pauli not in ("X", "Y", "Z"):
        raise ValueError("pauli must be X, Y or Z")
    if logical is None:
        if pauli == "Y":
            x = code.logical("X", logical_index)
            z = code.logical("Z", logical_index)
            # Logical Y=i Xbar Zbar; convert to canonical physical P(x,z).
            sign = ((1 - int(np.sum(x * z))) // 2) % 2
            return x, z, sign
        logical = code.logical(pauli, logical_index)
    if isinstance(logical, str):
        return parse_pauli(logical, code.n)
    if isinstance(logical, dict):
        x = binary_vector(logical.get("x", np.zeros(code.n)), length=code.n, name="logical.x")
        z = binary_vector(logical.get("z", np.zeros(code.n)), length=code.n, name="logical.z")
        sign = logical.get("sign", 0)
        if sign not in (0, 1):
            raise ValueError("logical.sign is a sign bit (0 or 1)")
        return x, z, int(sign)
    vector = binary_vector(logical, length=code.n, name="logical")
    zero = np.zeros(code.n, dtype=np.uint8)
    if pauli == "Y":
        return vector, vector.copy(), 0
    return (vector, zero, 0) if pauli == "X" else (zero, vector, 0)


def _parse_factors(code, factors, pauli):
    """Parse an explicit pure same-type Fig. 3 factorization."""
    if isinstance(factors, (str, bytes, dict)):
        raise ValueError("logical_factors must be a sequence of at least two pure X or Z operators")
    try:
        factors = list(factors)
    except TypeError as error:
        raise ValueError("logical_factors must be a sequence of operators") from error
    if len(factors) < 2:
        raise ValueError("logical_factors requires at least two operators; use logical for a single target")
    vectors, signs, factor_type = [], [], None
    x, z = np.zeros(code.n, dtype=np.uint8), np.zeros(code.n, dtype=np.uint8)
    occupied = np.zeros(code.n, dtype=bool)
    sign = 0
    for factor in factors:
        if factor is None:
            raise ValueError("A logical factor cannot be None")
        fx, fz, fs = _parse_target(code, factor, pauli, 0)
        if bool(np.any(fx)) == bool(np.any(fz)):
            raise ValueError("Fig. 3 logical factors must be nonidentity pure X or pure Z operators")
        kind = "X" if np.any(fx) else "Z"
        if factor_type is not None and kind != factor_type:
            raise ValueError("Fig. 3 logical factors must have the same Pauli type; use a mixed target for Y or ZX")
        factor_type = kind
        vector = fx if kind == "X" else fz
        overlap = np.flatnonzero(occupied & (vector != 0))
        if len(overlap):
            raise ValueError(f"CKBB Fig. 3 requires physically disjoint same-Pauli factors; overlapping qubits: {overlap.tolist()}")
        occupied |= vector != 0
        vectors.append(vector)
        signs.append(fs)
        x ^= fx
        z ^= fz
        sign ^= fs
    return vectors, signs, factor_type, x, z, sign


def ckbb_deform(code, logical=None, *, pauli="Z", logical_index=0, layers=None,
                logical_factors=None, bridge=True, max_entries=50_000_000):
    """Return a CKBB deformed code for one requested logical Pauli.

    logical: physical I/X/Y/Z string, binary vector with pauli=X/Z, or
    {x: binary vector, z: binary vector, sign: 0/1}. Omitted logical uses
    the first Z generator of the paired CSS standard form by default.
    CSSCode.distance is an asserted input distance, not computed here.
    Omitted distance and layers uses r=n, a conservative upper bound on d.
    logical_factors explicitly requests Fig. 3 parity bridging for disjoint,
    same-type operators. A raw logical never receives automatic support
    completion. Every final construction is checked for unintended logical
    readout; failure raises UnintendedLogicalMeasurementError before return.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    if not isinstance(bridge, (bool, np.bool_)):
        raise ValueError("bridge must be a boolean")
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    factors, factor_signs, factor_type = None, None, None
    if logical_factors is not None:
        if not bridge:
            raise ValueError("Explicit logical_factors require the Fig. 3 bridge; bridge=False disables it")
        factors, factor_signs, factor_type, x, z, sign = _parse_factors(code, logical_factors, pauli)
        if logical is not None:
            supplied_x, supplied_z, supplied_sign = _parse_target(code, logical, pauli, logical_index)
            if not np.array_equal(x, supplied_x) or not np.array_equal(z, supplied_z) or sign != supplied_sign:
                raise ValueError("Explicit logical must equal the signed product of logical_factors")
    else:
        x, z, sign = _parse_target(code, logical, pauli, logical_index)
    if not np.any(x | z):
        raise ValueError("Cannot measure the identity operator")
    if np.any((code.hx @ z) % 2) or np.any((code.hz @ x) % 2):
        raise ValueError("Requested Pauli does not commute with the input stabilizers")
    if in_rowspace(x, code.hx) and in_rowspace(z, code.hz):
        raise ValueError("Requested Pauli is a stabilizer, not a nontrivial logical operator")
    r = _resolve_layers(code, layers)
    if np.any(x) and np.any(z):
        from .mixed import construct_mixed
        built = construct_mixed(code, x, z, r, _build_css_primitive, sign=sign,
                                bridge=bridge, max_entries=max_entries)
        result = DeformedCode(**built, original_n=code.n, protected_k=code.k - 1)
    else:
        pure = "X" if np.any(x) else "Z"
        if factors is not None:
            from .bridges import construct_css_bridge
            primitive = construct_css_bridge(code, factors, factor_type, r,
                                             _assemble_css_layers, max_entries=max_entries)
            primitive.metadata["factor_signs"] = factor_signs
        else:
            primitive = _build_css_primitive(code, x if pure == "X" else z, pure, r,
                                            max_entries=max_entries)
        hx, hz = primitive.hx, primitive.hz
        check_x = np.vstack((hx, np.zeros_like(hz)))
        check_z = np.vstack((np.zeros_like(hx), hz))
        witness = np.zeros(len(check_x), dtype=np.uint8)
        dual_rows = primitive.metadata["dual_check_rows"].ravel()
        witness[dual_rows + (0 if pure == "X" else len(hx))] = 1
        signs = np.zeros(len(check_x), dtype=np.uint8)
        if sign:
            # Redefine a consistent generator presentation for the -target
            # eigenspace, respecting every dependency and original check.
            from .mixed import _consistent_signs
            fixed = list(range(len(code.hx))) + list(range(len(hx), len(hx) + len(code.hz)))
            signs = _consistent_signs(check_x, check_z, witness, sign, fixed)
        result = DeformedCode(check_x, check_z, signs, x, z, sign, witness,
                              code.n, code.k - 1, primitive.metadata, hx, hz)
        # Check the actual check group before assigning protected/gauge modes.
        # Counting its k alone cannot identify which memory observables it reads.
        result.metadata.update(validate_single_logical_measurement(code, check_x, check_z, x, z))
        gx, gz = _cycle_gauges(code, primitive, pure)
        result.gauge_x, result.gauge_z = gx, gz
        if len(gx) != result.gauge_qubits:
            raise RuntimeError("Cycle-gauge count does not match the protected subsystem dimension")
    branches = [result.metadata]
    if np.any(x) and np.any(z):
        branches = [result.metadata["x_sticker"], result.metadata["z_sticker"]]
    if factors is not None:
        irreducible = all(len(nullspace(f)) == 1 for f in result.metadata["factor_restrictions"])
        result.metadata["factor_irreducible_support"] = [len(nullspace(f)) == 1 for f in result.metadata["factor_restrictions"]]
    else:
        irreducible = all(branch["irreducible_support_before_completion"] for branch in branches)
    certificate_applicable = irreducible
    if not irreducible:
        distance_note = "No extra independent logical class, but support contains proper-subset logical representatives; the original irreducible-support distance certificate is not applied"
    else:
        distance_note = "CKBB protected subsystem distance, not raw stabilizer distance; input distance is assumed" if code.distance is not None else "Input distance unspecified; r=n preserves input distance conservatively" if r >= code.n else "Input distance unspecified; no distance-preservation certificate for this layer count"
    result.metadata.update({
        "support_completion": "none",
        "distance_support_condition_met": irreducible,
        "distance_lower_bound": min(code.distance, r) if code.distance is not None and certificate_applicable else None,
        "distance_preservation_guaranteed": certificate_applicable and r >= (code.distance or code.n),
        "distance_note": distance_note,
        "ancilla_initialization_and_readout": "Z basis" if np.any(x) and not np.any(z) else "X basis" if np.any(z) and not np.any(x) else "component-dependent; see mixed metadata",
    })
    result.validate()
    return result


def ckbb_batch(code, logicals=None, *, pauli="Z", **kwargs):
    """Independent deformations, one per request; never a simultaneous merge.

    With logicals=None return one independent result for each generator of
    the requested standard-form logical basis. Anticommuting requests may
    be studied separately because these are independent output codes.
    """
    if not isinstance(pauli, str):
        raise ValueError("pauli must be a string")
    if logicals is None:
        basis = code.standard_logicals()
        if pauli.upper() not in ("X", "Y", "Z"):
            raise ValueError("Default logical basis type must be X, Y or Z")
        if pauli.upper() == "Y":
            logicals = []
            for index in range(code.k):
                x, z, sign = _parse_target(code, None, "Y", index)
                logicals.append({"x": x, "z": z, "sign": sign})
        else:
            logicals = basis.jx if pauli.upper() == "X" else basis.jz
    return [ckbb_deform(code, operator, pauli=pauli, **kwargs) for operator in logicals]
