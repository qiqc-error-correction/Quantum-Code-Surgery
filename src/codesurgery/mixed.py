"""CKBB's non-CSS fusion, with explicit binary Pauli phases.

Figures 4 and 5 of Cohen--Kim--Bartlett--Brown (2022) fuse the
measurement generators of an X sticker and a Z sticker.  A mixed
deformation is represented by two matrices ``check_x`` and ``check_z``;
they are the halves of one symplectic matrix, not CSS check matrices.

The disjoint-support case first grows the two dangling connector arms
shown in Figure 5.  Merely constructing two independent stickers would
measure X and Z separately, and is therefore deliberately not used.
"""

from __future__ import annotations

import numpy as np

from .gf2 import binary_vector, in_rowspace, matmul, nullspace, solve
from .symplectic import commutation_matrix, pauli_product


def _metadata(result):
    return result.metadata if hasattr(result, "metadata") else result.details


def _consistent_signs(cx, cz, witness, measurement_sign, fixed_rows):
    """Choose real signs without a hidden -I or a changed target phase.

    The canonical Hermitian representative is
    P(x,z)=i**sum(x*z) X(x)Z(z).  Binary dependence alone cannot detect
    -I.  For every dependence, require its signed product to be +I,
    and separately require the witness to have the requested sign.
    Original memory check signs are held at +1.
    """
    full = np.hstack((cx, cz))
    dependencies = nullspace(full.T)
    equations = [row for row in dependencies]
    rhs = []
    for row in dependencies:
        _, _, phase = pauli_product(cx, cz, selected=np.flatnonzero(row))
        if phase % 2:
            raise RuntimeError("A commuting check dependence has an imaginary phase")
        rhs.append(phase // 2)
    _, _, phase = pauli_product(cx, cz, selected=np.flatnonzero(witness))
    if phase % 2:
        raise RuntimeError("A measurement witness has an imaginary phase")
    equations.append(witness)
    rhs.append(measurement_sign ^ (phase // 2))
    for index in fixed_rows:
        equation = np.zeros(len(cx), dtype=np.uint8)
        equation[index] = 1
        equations.append(equation)
        rhs.append(0)
    signs = solve(np.array(equations, dtype=np.uint8), np.array(rhs, dtype=np.uint8))
    return signs


def construct_mixed(code, x, z, layers, pure_builder, *, sign=0, bridge=True, max_entries=50_000_000):
    """Return CKBB's single mixed-Pauli deformation as a dictionary.

    ``x`` and ``z`` are binary physical vectors in the input qubit order.
    Both must be nonzero. ``sign`` is 0 for +P(x,z), 1 for
    -P(x,z). ``pure_builder`` takes ``(code, vector, pauli=..., layers=...,
    allow_stabilizer=True, max_entries=...)`` and returns ``hx``, ``hz``
    and metadata. The pure branches retain their original support
    restrictions; no additional path checks are inferred from their
    kernels. ``bridge`` controls the explicit Figure 5 connector arms.

    Its metadata contains ``support``, ``dual_check_rows`` (layers by
    support), ``primal_qubit_columns`` (layers-1 by support), and
    ``layers``. The input data qubits are always the first code.n columns.
    Each branch can have more than one kernel mode. This function checks
    the final fused construction and rejects it unless its logical
    readout space is exactly the requested one-dimensional space.
    """
    x = binary_vector(x, length=code.n, name="mixed X component")
    z = binary_vector(z, length=code.n, name="mixed Z component")
    if not np.any(x) or not np.any(z):
        raise ValueError("construct_mixed requires nonzero X and Z components")
    if sign not in (0, 1):
        raise ValueError("sign must be 0 (+) or 1 (-)")
    if np.any(matmul(code.hz, x)) or np.any(matmul(code.hx, z)):
        raise ValueError("Requested mixed Pauli must commute with all memory checks")
    original = np.block([
        [code.hx, np.zeros_like(code.hx)],
        [np.zeros_like(code.hz), code.hz],
    ])
    if in_rowspace(np.concatenate((x, z)), original):
        raise ValueError("Requested mixed Pauli is a memory stabilizer, not a logical measurement")
    if not np.any(x & z) and not bridge:
        raise ValueError("Disjoint X/Z supports require the CKBB Fig. 5 connector arms; enable bridge=True")

    px = pure_builder(code, x, pauli="X", layers=layers, allow_stabilizer=True,
                      max_entries=max_entries)
    pz = pure_builder(code, z, pauli="Z", layers=layers, allow_stabilizer=True,
                      max_entries=max_entries)
    mx, mz = _metadata(px), _metadata(pz)
    if mx["layers"] != mz["layers"]:
        raise ValueError("The two stickers must have the same number of layers")
    r = int(mx["layers"])
    n = code.n
    nx, nz = px.hx.shape[1], pz.hx.shape[1]
    total_n = nx + nz - n
    map_x = np.arange(nx, dtype=int)
    map_z = np.concatenate((np.arange(n), np.arange(nx, total_n)))
    rx, rz = code.hx.shape[0], code.hz.shape[0]
    support_x, support_z = list(mx["support"]), list(mz["support"])
    overlap = sorted(set(support_x) & set(support_z))
    max_rows = len(px.hx) + len(px.hz) + len(pz.hx) + len(pz.hz) - rx - rz
    final_n = total_n
    if not overlap:
        final_n += 4 * r - 2
        max_rows += 4 * r - 3
    if 2 * max_rows * final_n > max_entries:
        raise ValueError("Mixed deformation exceeds max_entries; reduce layers or increase the explicit limit")
    records = []
    row_map = {}

    def add_record(local, mapping, pauli, *, witness=False, label=""):
        sx = np.zeros(total_n, dtype=np.uint8)
        sz = np.zeros(total_n, dtype=np.uint8)
        (sx if pauli == "X" else sz)[mapping] = local
        records.append({"x": sx, "z": sz, "witness": witness, "label": label})
        return len(records) - 1

    # Keep each original memory check once, with the extension supplied by
    # the opposite-type sticker. Dropping its untouched duplicate matters:
    # that duplicate would anticommute with the new measurement checks.
    for row in pz.hx[:rx]:
        add_record(row, map_z, "X", label="memory X")
    for row in px.hz[:rz]:
        add_record(row, map_x, "Z", label="memory Z")
    fixed_rows = list(range(len(records)))
    for branch, result, mapping, metadata in (
        ("X", px, map_x, mx), ("Z", pz, map_z, mz)
    ):
        dual_set = set(np.asarray(metadata["dual_check_rows"]).reshape(-1).tolist())
        for pauli, checks, original_count in (("X", result.hx, rx), ("Z", result.hz, rz)):
            for index in range(original_count, len(checks)):
                row_map[branch, pauli, index] = add_record(
                    checks[index], mapping, pauli,
                    witness=pauli == branch and index in dual_set,
                    label=f"{branch} sticker {pauli} check {index}",
                )

    def extend_columns(count):
        nonlocal total_n
        columns = np.arange(total_n, total_n + count)
        for record in records:
            record["x"] = np.pad(record["x"], (0, count))
            record["z"] = np.pad(record["z"], (0, count))
        total_n += count
        return columns

    def append_check(sx, sz, *, witness, label):
        records.append({"x": sx, "z": sz, "witness": witness, "label": label})
        return len(records) - 1

    def connector_arm(branch, metadata, mapping):
        """Grow one virtual support endpoint and its connecting edge."""
        anchor = 0
        columns = extend_columns(2 * r - 1)
        edge_qubits, primal_endpoints = columns[:r], columns[r:]
        dual_rows = np.asarray(metadata["dual_check_rows"], dtype=int)
        primal = np.asarray(metadata["primal_qubit_columns"], dtype=int).reshape(r - 1, len(metadata["support"]))
        boundary = []
        same_key = "x" if branch == "X" else "z"
        for layer in range(r):
            index = row_map[branch, branch, int(dual_rows[layer, anchor])]
            records[index][same_key][edge_qubits[layer]] = 1
            sx, sz = np.zeros(total_n, dtype=np.uint8), np.zeros(total_n, dtype=np.uint8)
            same = sx if branch == "X" else sz
            same[edge_qubits[layer]] = 1
            if layer > 0:
                same[primal_endpoints[layer - 1]] = 1
            if layer < r - 1:
                same[primal_endpoints[layer]] = 1
            boundary.append(append_check(sx, sz, witness=True, label=f"{branch} connector boundary {layer + 1}"))
        for layer in range(1, r):
            sx, sz = np.zeros(total_n, dtype=np.uint8), np.zeros(total_n, dtype=np.uint8)
            opposite = sz if branch == "X" else sx
            opposite[[mapping[primal[layer - 1, anchor]], primal_endpoints[layer - 1],
                      edge_qubits[layer - 1], edge_qubits[layer]]] = 1
            append_check(sx, sz, witness=False, label=f"{branch} connector primal {layer + 1}")
        return boundary, primal_endpoints, edge_qubits

    pairs, relative_pairs = [], []
    connector_details = []
    if overlap:
        for physical in overlap:
            ix, iz = support_x.index(physical), support_z.index(physical)
            for layer in range(r):
                row_x = int(np.asarray(mx["dual_check_rows"])[layer, ix])
                row_z = int(np.asarray(mz["dual_check_rows"])[layer, iz])
                pairs.append((row_map["X", "X", row_x], row_map["Z", "Z", row_z]))
            for layer in range(r - 1):
                qx = map_x[int(np.asarray(mx["primal_qubit_columns"])[layer, ix])]
                qz = map_z[int(np.asarray(mz["primal_qubit_columns"])[layer, iz])]
                relative_pairs.append((int(qx), int(qz)))
    else:
        bx, qx, ex = connector_arm("X", mx, map_x)
        bz, qz, ez = connector_arm("Z", mz, map_z)
        pairs.extend(zip(bx, bz))
        relative_pairs.extend(zip(qx, qz))
        connector_details = [
            {"branch": "X", "edge_qubits": ex.tolist(), "primal_endpoints": qx.tolist()},
            {"branch": "Z", "edge_qubits": ez.tolist(), "primal_endpoints": qz.tolist()},
        ]

    removed = set()
    for left, right in pairs:
        a, b = records[left], records[right]
        # Use canonical Hermitian P, rather than multiplying every layer
        # by i. Only first-layer overlapping pairs anticommute; literal
        # i*Xa*Za on the disjoint later layers would be anti-Hermitian.
        append_check(a["x"] ^ b["x"], a["z"] ^ b["z"], witness=True, label="fused measurement check")
        removed.update((left, right))
    for qx, qz in relative_pairs:
        sx, sz = np.zeros(total_n, dtype=np.uint8), np.zeros(total_n, dtype=np.uint8)
        sz[qx], sx[qz] = 1, 1
        append_check(sx, sz, witness=False, label="relative XZ check")
    records = [record for index, record in enumerate(records) if index not in removed]
    cx = np.array([record["x"] for record in records], dtype=np.uint8)
    cz = np.array([record["z"] for record in records], dtype=np.uint8)
    witness = np.array([record["witness"] for record in records], dtype=np.uint8)
    if np.any(commutation_matrix(cx, cz)):
        raise RuntimeError("Mixed CKBB fusion produced noncommuting checks")
    signs = _consistent_signs(cx, cz, witness, int(sign), fixed_rows)
    sx, sz, phase = pauli_product(cx, cz, signs, np.flatnonzero(witness))
    if not np.array_equal(sx, np.pad(x, (0, total_n - n))) or not np.array_equal(sz, np.pad(z, (0, total_n - n))) or phase != 2 * sign:
        raise RuntimeError("Mixed CKBB witness failed to reproduce the requested signed operator")

    # Checking only the number of measured observables would also accept a
    # different rank-one observable. The shared validator eliminates the
    # ancillary support and compares the complete logical signature space
    # with the requested target, without adding checks to repair the input.
    from .measurement import validate_single_logical_measurement
    validation = validate_single_logical_measurement(code, cx, cz, x, z)
    return {
        "check_x": cx, "check_z": cz, "check_signs": signs,
        "measurement_x": x, "measurement_z": z, "measurement_sign": int(sign),
        "measurement_witness": witness,
        "metadata": {
            "layers": r, "construction": "CKBB Fig. 4 overlap fusion" if overlap else "CKBB Fig. 5 connector-arm fusion",
            "overlap": overlap, "connector_arms": connector_details,
            "x_sticker": mx, "z_sticker": mz,
            "check_labels": [record["label"] for record in records],
            **validation,
            "phase_convention": "sign=0: +P, sign=1: -P; P(x,z)=i**sum(x*z) X(x)Z(z)",
            "sign_adjusted_rows": np.flatnonzero(signs).tolist(),
        },
    }
