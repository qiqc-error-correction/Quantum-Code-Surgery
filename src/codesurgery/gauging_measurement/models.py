"""Sealed, signed stabilizer output of auxiliary-graph surgery."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from ..models import DeformedCode
from ..gf2 import nullspace, rank
from ..symplectic import pauli_product
from ..gadget_common import memory_checks, validate_readout


@dataclass
class GraphMeasurementCode(DeformedCode):
    _source_code: object = field(default=None, repr=False)
    _digest: bytes | None = field(default=None, repr=False)

    def _fingerprint(self):
        digest = hashlib.sha256()
        def add(value):
            if isinstance(value, np.ndarray):
                digest.update(b"array")
                add(value.dtype.str)
                add(value.shape)
                digest.update(value.tobytes())
            elif isinstance(value, dict):
                digest.update(b"dict")
                add(len(value))
                for key in sorted(value):
                    add(key)
                    add(value[key])
            elif isinstance(value, (tuple, list)):
                digest.update(b"sequence")
                add(len(value))
                for item in value:
                    add(item)
            elif isinstance(value, np.generic):
                add(value.item())
            elif value is None or isinstance(value, (str, bool, int, float)):
                encoded = repr(value).encode()
                digest.update(str(len(encoded)).encode() + b":" + encoded)
            else:
                raise ValueError("Unsupported graph-gadget metadata value")
        for key in ("check_x", "check_z", "check_signs", "measurement_x", "measurement_z",
                    "measurement_sign", "measurement_witness", "original_n", "protected_k",
                    "hx", "hz", "gauge_x", "gauge_z", "metadata"):
            add(getattr(self, key))
        if self._source_code is not None:
            add((self._source_code.hx, self._source_code.hz, self._source_code.distance))
        return digest.digest()

    def _seal(self):
        if self._digest is not None:
            raise ValueError("Construction is already sealed")
        self._digest = self._fingerprint()

    def _unchanged(self):
        if self._digest is None or self._digest != self._fingerprint():
            raise ValueError("Gadget matrices or certificate changed; rebuild the gadget")

    def validate(self):
        self._unchanged()
        super().validate()
        source = self._source_code
        if source is None or source.n != self.original_n:
            raise ValueError("Source code does not match the gadget")
        if "logical_optimization" in self.metadata:
            from ..logical_optimization import verify_logical_optimization
            verify_logical_optimization(source,
                (self.measurement_x, self.measurement_z, self.measurement_sign),
                self.metadata["logical_basis"], self.metadata["logical_optimization"])
        b, s, m, cycles = (self.metadata[key] for key in
                           ("graph_incidence", "port_selector", "matching_matrix", "cycle_basis"))
        e, v = b.shape
        kind = self.metadata.get("incidence_kind", "graph")
        if kind == "graph":
            if np.any(b.sum(axis=1) != 2) or rank(b) != v - 1:
                raise ValueError("Gadget graph must be an ordinary connected graph")
        elif kind == "even_tanner":
            if v < 1 or np.any(b.sum(axis=1) % 2):
                raise ValueError("Tanner restriction must have even rows")
        else:
            raise ValueError("Unknown gadget incidence kind")
        if np.any((cycles @ b) % 2) or rank(cycles) != e - rank(b):
            raise ValueError("Auxiliary kernel check space is incomplete")
        support = self.measurement_x | self.measurement_z
        if np.any(s.sum(axis=1) > 1) or not np.array_equal(s.sum(axis=0), support):
            raise ValueError("Ports must attach each target qubit exactly once")
        oldx, oldz = memory_checks(source)
        r = len(oldx)
        local = (oldx * self.measurement_z + oldz * self.measurement_x) % 2
        if not np.array_equal((m @ b) % 2, (local @ s.T) % 2):
            raise ValueError("Matching boundary is inconsistent with memory checks")
        ex = np.zeros((r + v + len(cycles), source.n + e), dtype=np.uint8)
        ez = np.zeros_like(ex)
        ex[:r, :source.n], ez[:r, :source.n] = oldx, oldz
        ex[r:r + v, :source.n] = s * self.measurement_x
        ez[r:r + v, :source.n] = s * self.measurement_z
        same, opposite = (ez, ex) if self.metadata["ancilla_vertex_pauli"] == "Z" else (ex, ez)
        opposite[:r, source.n:] = m
        same[r:r + v, source.n:] = b.T
        opposite[r + v:, source.n:] = cycles
        if not np.array_equal(ex, self.check_x) or not np.array_equal(ez, self.check_z):
            raise ValueError("Exported graph data disagree with actual checks")
        if np.any(self.check_signs[:r]) or np.any(self.check_signs[r + v:]):
            raise ValueError("Memory and flux check signs must be positive")
        for dep in nullspace(np.hstack([ex, ez]).T):
            _, _, phase = pauli_product(ex, ez, self.check_signs, np.flatnonzero(dep))
            if phase:
                raise ValueError("Signed checks contain a -I dependency")
        validate_readout(source, ex, ez, (self.measurement_x, self.measurement_z, self.measurement_sign),
                         self.metadata["logical_basis"])
        if self.k_stabilizer != source.k - 1 or self.protected_k != self.k_stabilizer:
            raise ValueError("Deformed code must preserve exactly k-1 logical qubits")
        if self.is_css:
            if not np.array_equal(self.hx, ex[np.any(ex, axis=1)]) or not np.array_equal(self.hz, ez[np.any(ez, axis=1)]):
                raise ValueError("CSS arrays disagree with symplectic checks")
        elif self.hx is not None or self.hz is not None:
            raise ValueError("Mixed-Pauli checks require symplectic output")
        return True

    def summary(self):
        self._unchanged()
        out = super().summary()
        out.pop("layers", None)
        out.pop("support_completion", None)
        out.update(scheme=self.metadata["scheme"],
                   graph_vertices=self.metadata["graph_incidence"].shape[1],
                   cycle_checks=len(self.metadata["cycle_basis"]),
                   max_check_weight=self.metadata["max_check_weight"],
                   max_qubit_degree=self.metadata["max_qubit_degree"],
                   relative_expansion=self.metadata["relative_expansion"],
                   adapter_qubits=self.metadata.get("adapter", {}).get("adapter_size", 0))
        origin = self.metadata.get("initial_construction", {})
        design = self.metadata.get("graph_design") or origin.get("graph_design") or {}
        factors = self.metadata.get("factors") or origin.get("factors") or []
        designs = ([design] if "component_count" in design else
                   [factor.get("graph_design", {}) for factor in factors])
        component_designs = [item for item in designs if "component_count" in item]
        outer_adapter = self.metadata.get("adapter") or origin.get("adapter") or {}
        out["adapter_qubits"] = outer_adapter.get("adapter_size", 0)
        if component_designs:
            added = sum(item["added_adapter_edges"] for item in component_designs)
            out["component_adapter_qubits"] = added
            out["adapter_qubits"] += added
            if len(component_designs) == len(designs):
                out["initial_graph_components"] = sum(item["component_count"] for item in designs)
                out["initial_component_logical_rank"] = rank(np.vstack(
                    [item["component_signature_matrix"] for item in designs]))
        if self.metadata.get("distance_postprocessing") is not None:
            report = self.metadata["distance_postprocessing"]
            out["distance_required"] = True
            out["postprocessing_changed"] = report["changed"]
            out["sparsity_certificate"] = report["sparsity_certificate"]
        if "distance_calculation" in self.metadata:
            out["distance_calculation"] = self.metadata["distance_calculation"]
            out["source_distance_calculation"] = self.metadata["source_distance_calculation"]
            out["distance_verification_method"] = self.metadata["distance_verification_method"]
            out["sparsity_threshold"] = self.metadata["distance_postprocessing"]["sparsity_threshold"]
        if self.metadata.get("boundary_expansion") is not None:
            out["boundary_expansion"] = self.metadata["boundary_expansion"]
        for key in ("cheeger_constant", "cheeger_lower_bound", "expansion_certified", "added_edge_qubits"):
            if key in self.metadata:
                out[key] = self.metadata[key]
        return out

    def save(self, path):
        if Path(path).suffix not in (".json", ".npz"):
            raise ValueError("Output extension must be .json or .npz")
        self.validate()
        return super().save(path)
