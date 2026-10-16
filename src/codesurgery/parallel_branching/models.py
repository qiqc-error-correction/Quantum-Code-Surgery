"""Auditable output of branching: checks, complete logical map and leaf copies."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ..gf2 import binary_matrix, binary_vector, matmul, nullspace, rank, solve
from ..symplectic import pauli_product, pauli_string
from .construction import append_branch, lift_paulis, _signed_check_rank
from .direct_attachment import preview_direct_attachment
from .policy import evaluate_routing


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _same_record(actual, expected):
    """Compare an exported construction record without trusting status flags."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_record(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, np.ndarray):
        return np.array_equal(actual, expected)
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_same_record(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def equivalence_witnesses(cx, cz, cs, ax, az, ass, bx, bz, bs):
    """Find and check signed identities B = A times a stabilizer product."""
    witness = solve(np.hstack((cx, cz)).T, np.hstack((ax ^ bx, az ^ bz)).T).T
    check_witnesses(cx, cz, cs, ax, az, ass, bx, bz, bs, witness)
    return witness


def check_witnesses(cx, cz, cs, ax, az, ass, bx, bz, bs, witness):
    w = binary_matrix(witness, n_cols=len(cx), name="equivalence witnesses")
    if len(w) != len(ax):
        raise ValueError("Wrong number of equivalence witnesses")
    for i, row in enumerate(w):
        sx, sz, phase = pauli_product(cx, cz, cs, np.flatnonzero(row))
        if phase % 2:
            raise ValueError("Non-Hermitian stabilizer product")
        px, pz, phase = pauli_product(np.stack((ax[i], sx)), np.stack((az[i], sz)),
                                     [ass[i], phase // 2])
        if (phase != 2 * int(bs[i]) or not np.array_equal(px, bx[i])
                or not np.array_equal(pz, bz[i])):
            raise ValueError("Signed logical-equivalence witness failed")


def _paired_basis(cx, cz, x, z, k):
    if x.shape != (2 * k, cx.shape[1]) or z.shape != x.shape:
        raise ValueError("Complete paired logical basis has the wrong shape")
    if np.any(matmul(x, cz.T) ^ matmul(z, cx.T)):
        raise ValueError("Logical basis does not normalize the stabilizers")
    omega = np.zeros((2 * k, 2 * k), np.uint8)
    omega[:k, k:] = np.eye(k, dtype=np.uint8)
    omega[k:, :k] = np.eye(k, dtype=np.uint8)
    if not np.array_equal(matmul(x, z.T) ^ matmul(z, x.T), omega):
        raise ValueError("Logical basis is not a complete paired symplectic basis")


@dataclass
class BranchedCode:
    check_x: np.ndarray
    check_z: np.ndarray
    check_signs: np.ndarray
    source: dict
    logical_basis_x: np.ndarray
    logical_basis_z: np.ndarray
    logical_basis_signs: np.ndarray
    lifted_target_x: np.ndarray
    lifted_target_z: np.ndarray
    lifted_target_signs: np.ndarray
    leaf_x: np.ndarray
    leaf_z: np.ndarray
    leaf_signs: np.ndarray
    equivalence_witnesses: np.ndarray
    branches: list[dict]
    metadata: dict = field(default_factory=dict)
    max_entries: int = 50_000_000
    routing: dict | None = None
    measurement_preview: dict | None = None

    @property
    def n(self):
        return self.check_x.shape[1]

    @property
    def k(self):
        return self.n - rank(np.hstack((self.check_x, self.check_z)))

    @property
    def protected_k(self):
        return self.source["basis_x"].shape[0] // 2

    @property
    def gauge_qubits(self):
        return 0

    @property
    def original_n(self):
        return self.source["check_x"].shape[1]

    @property
    def is_css(self):
        return not np.any(np.any(self.check_x, axis=1) & np.any(self.check_z, axis=1))

    @property
    def hx(self):
        return self.check_x[np.any(self.check_x, axis=1)] if self.is_css else None

    @property
    def hz(self):
        return self.check_z[np.any(self.check_z, axis=1)] if self.is_css else None

    def validate(self):
        """Rebuild every sticker and check signs, logical transport and no readout.

        Metadata is descriptive and is never used as a proof of correctness.
        This verifies a static stabilizer construction, not a syndrome circuit.
        """
        x = binary_matrix(self.check_x, name="check_x")
        z = binary_matrix(self.check_z, n_cols=x.shape[1], name="check_z")
        signs = binary_vector(self.check_signs, length=len(x), name="check_signs")
        if x.shape != z.shape:
            raise ValueError("Output check shapes differ")
        if (2 * x.size + len(x) ** 2 > self.max_entries):
            raise ValueError("Validation exceeds max_entries")
        if self.routing is None and self.measurement_preview is not None:
            raise ValueError("Measurement preview requires a routing record")
        final_rank = _signed_check_rank(x, z, signs)
        source = self.source
        sx = binary_matrix(source["check_x"], name="source check_x")
        sz = binary_matrix(source["check_z"], n_cols=sx.shape[1], name="source check_z")
        ss = binary_vector(source["check_signs"], length=len(sx), name="source signs")
        if sx.shape != sz.shape:
            raise ValueError("Source check shapes differ")
        source_rank = _signed_check_rank(sx, sz, ss)
        k = sx.shape[1] - source_rank
        if x.shape[1] - final_rank != k:
            raise ValueError("Branching changed the number of logical qubits")
        bx = binary_matrix(source["basis_x"], n_cols=sx.shape[1], name="source basis_x")
        bz = binary_matrix(source["basis_z"], n_cols=sx.shape[1], name="source basis_z")
        bsign = binary_vector(source["basis_signs"], length=2*k, name="source basis signs")
        _paired_basis(sx, sz, bx, bz, k)
        tx = binary_matrix(source["target_x"], n_cols=sx.shape[1], name="source targets")
        tz = binary_matrix(source["target_z"], n_cols=sx.shape[1], name="source targets")
        ts = binary_vector(source["target_signs"], length=len(tx), name="source target signs")
        if len(tx) == 0 or tx.shape != tz.shape:
            raise ValueError("A nonempty list of logical targets is required")
        if np.any(matmul(tx, sz.T) ^ matmul(tz, sx.T)):
            raise ValueError("Source targets do not normalize source checks")
        sig = matmul(tx, bz.T) ^ matmul(tz, bx.T)
        if np.any(~np.any(sig, axis=1)):
            raise ValueError("Source target is a trivial logical")
        prepared_x = binary_matrix(source["prepared_x"], n_cols=sx.shape[1])
        prepared_z = binary_matrix(source["prepared_z"], n_cols=sx.shape[1])
        prepared_s = binary_vector(source["prepared_signs"], length=len(tx))
        if prepared_x.shape != tx.shape or prepared_z.shape != tx.shape:
            raise ValueError("Prepared targets have the wrong shape")
        check_witnesses(sx, sz, ss, tx, tz, ts, prepared_x, prepared_z, prepared_s,
                        source["canonicalization_witnesses"])
        reduction_keys = ("pre_optimization_x", "pre_optimization_z", "pre_optimization_signs",
                          "initial_canonicalization_witnesses", "support_reduction_witnesses")
        if any(key in source for key in reduction_keys):
            if not all(key in source for key in reduction_keys):
                raise ValueError("Incomplete support-reduction audit data")
            pre_x = binary_matrix(source["pre_optimization_x"], n_cols=sx.shape[1])
            pre_z = binary_matrix(source["pre_optimization_z"], n_cols=sx.shape[1])
            pre_s = binary_vector(source["pre_optimization_signs"], length=len(tx))
            if pre_x.shape != tx.shape or pre_z.shape != tx.shape:
                raise ValueError("Pre-optimization targets have the wrong shape")
            check_witnesses(sx, sz, ss, tx, tz, ts, pre_x, pre_z, pre_s,
                            source["initial_canonicalization_witnesses"])
            check_witnesses(sx, sz, ss, pre_x, pre_z, pre_s, prepared_x, prepared_z, prepared_s,
                            source["support_reduction_witnesses"])
            if np.any(np.count_nonzero(prepared_x | prepared_z, axis=1)
                      > np.count_nonzero(pre_x | pre_z, axis=1)):
                raise ValueError("Support reduction increased a target weight")
        # Replaying the append-only construction validates its structural
        # certificate; no stored status flag or distance number is trusted.
        rx, rz, rs = sx, sz, ss
        current_x, current_z, current_s = prepared_x.copy(), prepared_z.copy(), prepared_s.copy()
        locations = [None] * len(tx)
        for desc in self.branches:
            if self.routing is not None:
                group = desc.get("targets")
                if (not isinstance(group, list) or not group or len(set(group)) != len(group)
                        or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(tx)
                               for i in group)):
                    raise ValueError("Invalid branch target group")
                if (not np.array_equal(desc["local_x"], np.bitwise_or.reduce(current_x[group], axis=0))
                        or not np.array_equal(desc["local_z"], np.bitwise_or.reduce(current_z[group], axis=0))):
                    raise ValueError("Branch axes disagree with the routed target representatives")
            branch = append_branch(rx, rz, rs, desc["local_x"], desc["local_z"],
                                   leaf_pauli=desc["leaf_pauli"], max_entries=self.max_entries)
            for key in ("old_n", "old_checks", "support", "F", "T", "incident_rows",
                        "dual_columns", "leaf_columns", "junction_rows", "cap_rows"):
                if not np.array_equal(desc[key], branch[key]):
                    raise ValueError(f"Branch descriptor mismatch: {key}")
            bx, bz, bsign = lift_paulis(bx, bz, bsign, branch)
            tx, tz, ts = lift_paulis(tx, tz, ts, branch)
            if self.routing is not None:
                mask = (current_x[group][:, branch["support"]]
                        | current_z[group][:, branch["support"]]).copy()
                if np.any(matmul(branch["F"], mask.T)):
                    raise ValueError("Routed target cannot be cleaned onto its branch")
                current_x, current_z, current_s = lift_paulis(current_x, current_z, current_s, branch)
                current_x[group], current_z[group] = 0, 0
                (current_z if desc["leaf_pauli"] == "Z" else current_x)[
                    np.ix_(group, branch["leaf_columns"])] = mask
                for i in group:
                    locations[i] = desc["id"]
            rx, rz, rs = branch["check_x"], branch["check_z"], branch["check_signs"]
        for actual, expected in ((x, rx), (z, rz), (signs, rs),
                (self.logical_basis_x, bx), (self.logical_basis_z, bz),
                (self.logical_basis_signs, bsign), (self.lifted_target_x, tx),
                (self.lifted_target_z, tz), (self.lifted_target_signs, ts)):
            if not np.array_equal(actual, expected):
                raise ValueError("Checks or full logical transport disagree with the branch construction")
        _paired_basis(x, z, bx, bz, k)
        lx = binary_matrix(self.leaf_x, n_cols=x.shape[1], name="leaf_x")
        lz = binary_matrix(self.leaf_z, n_cols=x.shape[1], name="leaf_z")
        ls = binary_vector(self.leaf_signs, length=len(tx), name="leaf signs")
        if lx.shape != tx.shape or lz.shape != tx.shape:
            raise ValueError("Leaf targets have wrong shape")
        support = lx | lz
        disjoint = not bool(np.any(np.sum(support, axis=0) > 1))
        if self.routing is None and not disjoint:
            raise ValueError("Leaf target supports overlap")
        if np.any(matmul(lx, z.T) ^ matmul(lz, x.T)):
            raise ValueError("Leaf targets do not normalize output checks")
        check_witnesses(x, z, signs, tx, tz, ts, lx, lz, ls, self.equivalence_witnesses)
        limits_satisfied = None
        if self.routing is not None:
            if not all(np.array_equal(a, b) for a, b in ((lx, current_x), (lz, current_z), (ls, current_s))):
                raise ValueError("Final measurement representatives do not match actual branch routing")
            route = self.routing
            if route.get("mode") not in ("adaptive", "always"):
                raise ValueError("Unknown branching routing mode")
            for key in ("max_check_increase", "max_qubit_increase"):
                if (isinstance(route.get(key), (bool, np.bool_))
                        or not isinstance(route.get(key), (int, np.integer)) or route[key] < 0):
                    raise ValueError("Invalid per-node branching growth limit")
            for index, desc in enumerate(self.branches):
                if desc.get("id") != index:
                    raise ValueError("Branch ids disagree with append order")
                parent = desc.get("parent")
                if parent is None:
                    if desc.get("depth") != 1:
                        raise ValueError("Root branch has incorrect depth")
                elif (isinstance(parent, bool) or not isinstance(parent, int) or not 0 <= parent < index
                      or desc.get("depth") != self.branches[parent]["depth"] + 1
                      or not set(desc["targets"]).issubset(self.branches[parent]["targets"])):
                    raise ValueError("Invalid branch ancestry")
            roots = [desc["targets"] for desc in self.branches if desc["parent"] is None]
            if roots != route.get("root_groups"):
                raise ValueError("Stored routing does not match actual root branches")
            evaluated = evaluate_routing(sx, sz, prepared_x, prepared_z, roots)
            for key in ("ancillas", "direct_targets", "branched_targets",
                        "projected_original_row_growth", "projected_original_column_growth"):
                if not np.array_equal(route.get(key), evaluated[key]):
                    raise ValueError(f"Routing record mismatch: {key}")
            if evaluated["ancillas"] != self.n - sx.shape[1]:
                raise ValueError("Routing does not describe the constructed subtree sizes")
            direct = [i for i, loc in enumerate(locations) if loc is None]
            if direct != evaluated["direct_targets"]:
                raise ValueError("Direct target locations disagree with the routing")
            moved = evaluated["branched_targets"]
            if moved and (np.any(support[moved, :sx.shape[1]])
                          or np.any(np.sum(support[moved], axis=0) > 1)):
                raise ValueError("Branched representatives are not separated on auxiliary leaves")
            if direct and np.any(support[direct, sx.shape[1]:]):
                raise ValueError("A direct representative has left the original memory")
            preview = preview_direct_attachment(x, z, lx, lz, check_signs=signs,
                                                target_signs=ls, max_entries=self.max_entries)
            if self.measurement_preview is None or not _same_record(self.measurement_preview, preview):
                raise ValueError("Measurement preview differs from the actual paired-graph construction")
            projected = preview["check_x"] | preview["check_z"]
            row_growth = projected[:len(sx)].sum(axis=1, dtype=np.int64) - (sx | sz).sum(axis=1)
            col_growth = projected[:, :sx.shape[1]].sum(axis=0, dtype=np.int64) - (sx | sz).sum(axis=0)
            if (not np.array_equal(row_growth, evaluated["projected_original_row_growth"])
                    or not np.array_equal(col_growth, evaluated["projected_original_column_growth"])):
                raise ValueError("Actual preview matrices violate the predicted source growth")
            limits_satisfied = bool(np.all(row_growth <= route["max_check_increase"])
                                    and np.all(col_growth <= route["max_qubit_increase"]))
            if route["mode"] == "adaptive" and not limits_satisfied:
                raise ValueError("Adaptive routing exceeds an original check or qubit growth limit")
            if preview["k"] != k - rank(sig):
                raise ValueError("Preview reads a different number of logical observables")
        # Eliminate auxiliary coordinates from all check products. Any
        # remaining memory-only operator must be an original stabilizer.
        n0 = sx.shape[1]
        coeff = nullspace(np.hstack((x[:, n0:], z[:, n0:])).T)
        memory_x, memory_z = matmul(coeff, x[:, :n0]), matmul(coeff, z[:, :n0])
        memory_s = []
        for row in coeff:
            _, _, phase = pauli_product(x, z, signs, np.flatnonzero(row))
            memory_s.append(phase // 2)
        zero = np.zeros_like(memory_x)
        equivalence_witnesses(sx, sz, ss, zero, zero, np.zeros(len(zero), np.uint8),
                              memory_x, memory_z, np.asarray(memory_s, np.uint8))
        return {"valid": True, "checks_commute": True, "signed_relations_valid": True,
                "logical_qubits_preserved": k, "disjoint_leaf_supports": disjoint,
                "measurement_attachment_limits_satisfied": limits_satisfied,
                "unintended_logical_readout_rank": 0, "static_distance_nondecreasing": True}

    def summary(self):
        weights = self.check_x | self.check_z
        result = {"n": self.n, "k": self.k, "original_n": self.original_n,
                "ancillas": self.n - self.original_n, "is_css": self.is_css,
                "target_count": len(self.leaf_x), "stickers": len(self.branches),
                "tree_depth": max((b["depth"] for b in self.branches), default=0),
                "max_check_weight": int(weights.sum(axis=1).max(initial=0)),
                "max_qubit_degree": int(weights.sum(axis=0).max(initial=0)),
                "leaf_weights": np.sum(self.leaf_x | self.leaf_z, axis=1).tolist(),
                "leaf_paulis": [pauli_string(x,z,s) for x,z,s in zip(self.leaf_x,self.leaf_z,self.leaf_signs)],
                "logical_measurement_performed": False,
                "fully_disjoint": not bool(np.any(np.sum(self.leaf_x | self.leaf_z, axis=0) > 1))}
        if self.routing is not None:
            result.update(branching_mode=self.routing["mode"],
                direct_target_count=len(self.routing["direct_targets"]),
                branched_target_count=len(self.routing["branched_targets"]))
        if self.measurement_preview is not None:
            p = self.measurement_preview
            weights = p["check_x"] | p["check_z"]
            n0, m0 = self.original_n, len(self.source["check_x"])
            source_weights = self.source["check_x"] | self.source["check_z"]
            row_growth = weights[:m0].sum(axis=1, dtype=np.int64) - source_weights.sum(axis=1)
            col_growth = weights[:, :n0].sum(axis=0, dtype=np.int64) - source_weights.sum(axis=0)
            result.update(projected_measurement_n=p["n"], projected_measurement_k=p["k"],
                max_original_check_increase=int(row_growth.max(initial=0)),
                max_original_qubit_increase=int(col_growth.max(initial=0)),
                projected_max_new_check_weight=int(weights[m0:].sum(axis=1).max(initial=0)),
                projected_max_new_qubit_degree=int(weights[:, n0:].sum(axis=0).max(initial=0)))
        return result

    def save(self, path):
        """Validate before writing JSON or NPZ; include the complete audit data."""
        self.validate()
        path = Path(path)
        if path.suffix.lower() not in (".json", ".npz"):
            raise ValueError("Output must end in .json or .npz")
        data = {key: getattr(self, key) for key in self.__dataclass_fields__}
        data["summary"] = self.summary()
        data["format"] = "codesurgery.parallel_branching.v2"
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            arrays = {k: v for k,v in data.items() if isinstance(v, np.ndarray)}
            arrays["metadata"] = np.asarray(json.dumps(jsonable(self.metadata), ensure_ascii=False))
            arrays["record_json"] = np.asarray(json.dumps(jsonable(data), ensure_ascii=False))
            with path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
        return path


__all__ = ["BranchedCode"]
