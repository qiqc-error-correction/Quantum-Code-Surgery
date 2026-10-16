"""Validated CSS input codes and phase-aware deformed-code outputs."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from .gf2 import binary_matrix, rank
from .logicals import standard_form_logicals
from .symplectic import commutation_matrix, pauli_product, pauli_string


@dataclass
class CSSCode:
    hx: np.ndarray
    hz: np.ndarray
    distance: int | None = None
    name: str = "CSS code"

    def __post_init__(self):
        self.hx = binary_matrix(self.hx, name="hx")
        self.hz = binary_matrix(self.hz, name="hz")
        if self.hx.shape == (0, 0) and self.hz.shape[1]:
            self.hx = np.zeros((0, self.hz.shape[1]), dtype=np.uint8)
        if self.hz.shape == (0, 0) and self.hx.shape[1]:
            self.hz = np.zeros((0, self.hx.shape[1]), dtype=np.uint8)
        if self.hx.shape[1] != self.hz.shape[1] or self.hx.shape[1] == 0:
            raise ValueError("hx and hz must have the same positive column count")
        if np.any((self.hx @ self.hz.T) % 2):
            raise ValueError("Invalid CSS code: hx @ hz.T != 0 over GF(2)")
        if self.distance is not None:
            if isinstance(self.distance, bool) or not isinstance(self.distance, (int, np.integer)) or not 1 <= self.distance <= self.n:
                raise ValueError("distance must be an integer between 1 and n")
            self.distance = int(self.distance)

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k(self):
        return self.n - rank(self.hx) - rank(self.hz)

    def standard_logicals(self):
        return standard_form_logicals(self.hx, self.hz)

    def logical(self, pauli="Z", index=0):
        basis = self.standard_logicals()
        if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
            raise ValueError("Logical basis type must be X or Z")
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < self.k:
            raise ValueError(f"logical index must be in [0, {self.k})")
        return (basis.jx if pauli.upper() == "X" else basis.jz)[index].copy()

    def logical_product(self, coefficients, pauli="Z"):
        c = binary_matrix(np.asarray(coefficients).reshape(1, -1), name="coefficients")
        if c.shape[1] != self.k:
            raise ValueError("There must be one coefficient per logical qubit")
        basis = self.standard_logicals()
        if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
            raise ValueError("Logical basis type must be X or Z")
        return ((c @ (basis.jx if pauli.upper() == "X" else basis.jz)) % 2)[0]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _jsonable(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class DeformedCode:
    check_x: np.ndarray
    check_z: np.ndarray
    check_signs: np.ndarray
    measurement_x: np.ndarray
    measurement_z: np.ndarray
    measurement_sign: int
    measurement_witness: np.ndarray
    original_n: int
    protected_k: int
    metadata: dict = field(default_factory=dict)
    hx: np.ndarray | None = None
    hz: np.ndarray | None = None
    gauge_x: np.ndarray | None = None
    gauge_z: np.ndarray | None = None

    @property
    def n(self):
        return self.check_x.shape[1]

    @property
    def k_stabilizer(self):
        return self.n - rank(np.hstack((self.check_x, self.check_z)))

    @property
    def gauge_qubits(self):
        return self.k_stabilizer - self.protected_k

    @property
    def is_css(self):
        return not np.any(np.any(self.check_x, axis=1) & np.any(self.check_z, axis=1))

    def validate(self):
        if self.check_x.shape != self.check_z.shape:
            raise ValueError("check_x and check_z shapes differ")
        if np.any(commutation_matrix(self.check_x, self.check_z)):
            raise ValueError("Deformed stabilizer checks do not commute")
        sx, sz, phase = pauli_product(self.check_x, self.check_z, self.check_signs,
                                    np.flatnonzero(self.measurement_witness))
        target_x = np.pad(self.measurement_x, (0, self.n - self.original_n))
        target_z = np.pad(self.measurement_z, (0, self.n - self.original_n))
        if not np.array_equal(sx, target_x) or not np.array_equal(sz, target_z) or phase != 2 * self.measurement_sign:
            raise ValueError("Measurement witness does not equal the requested signed Pauli")
        if self.gauge_qubits < 0:
            raise ValueError("Deformation measured extra protected information")
        return True

    def summary(self):
        return {
            "scheme": "ckbb", "original_n": self.original_n, "deformed_n": self.n,
            "construction": self.metadata.get("construction"),
            "logical_measurement_rank": self.metadata.get("logical_measurement_rank"),
            "ancilla_qubits": self.n - self.original_n, "checks": len(self.check_x),
            "protected_k": self.protected_k, "stabilizer_k": self.k_stabilizer,
            "gauge_qubits": self.gauge_qubits, "is_css": self.is_css,
            "measurement": pauli_string(self.measurement_x, self.measurement_z, self.measurement_sign),
            "layers": self.metadata.get("layers"),
            "support_completion": self.metadata.get("support_completion"),
            "distance_preservation_guaranteed": self.metadata.get("distance_preservation_guaranteed"),
            "distance_lower_bound": self.metadata.get("distance_lower_bound"),
            "distance_note": self.metadata.get("distance_note"),
        }

    def save(self, path):
        """Save .npz arrays + JSON metadata, or a human-readable .json file."""
        path = Path(path)
        arrays = {key: getattr(self, key) for key in ("check_x", "check_z", "check_signs", "measurement_x", "measurement_z", "measurement_witness", "hx", "hz", "gauge_x", "gauge_z") if getattr(self, key) is not None}
        information = {"summary": self.summary(), "metadata": self.metadata,
                       "measurement_sign": self.measurement_sign}
        if path.suffix == ".npz":
            np.savez_compressed(path, **arrays, information=json.dumps(_jsonable(information), ensure_ascii=False))
        elif path.suffix == ".json":
            path.write_text(json.dumps(_jsonable({**arrays, **information}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            raise ValueError("Output extension must be .json or .npz")
