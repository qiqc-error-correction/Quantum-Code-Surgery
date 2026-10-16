"""Independent, multiple-observable output for Zhang--Li devised sticking."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ..gf2 import binary_matrix, binary_vector, matmul, nullspace, rank
from ..gadget_common import logical_basis_for
from ..models import _jsonable
from ..symplectic import pauli_product, pauli_string
from .construction import assemble_deformed, validate_readout


@dataclass
class DevisedDeformedCode:
    hx: np.ndarray
    hz: np.ndarray
    check_x: np.ndarray
    check_z: np.ndarray
    check_signs: np.ndarray
    measurement_x: np.ndarray
    measurement_z: np.ndarray
    measurement_signs: np.ndarray
    measurement_witnesses: np.ndarray
    original_n: int
    protected_k: int
    protected_x: np.ndarray
    protected_z: np.ndarray
    glue: object
    _source_code: object = field(repr=False)
    metadata: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k_stabilizer(self):
        return self.n - rank(self.hx) - rank(self.hz)

    @property
    def gauge_qubits(self):
        return self.k_stabilizer - self.protected_k

    @property
    def is_css(self):
        return True

    def validate(self):
        """Recheck commutation, exact readout space, signs and protected pairs."""
        hx = binary_matrix(self.hx, n_cols=self.n, name="hx")
        hz = binary_matrix(self.hz, n_cols=self.n, name="hz")
        if np.any(matmul(hx, hz.T)):
            raise ValueError("Devised-sticking CSS checks do not commute")
        cx = binary_matrix(self.check_x, n_cols=self.n, name="check_x")
        cz = binary_matrix(self.check_z, n_cols=self.n, name="check_z")
        expected_x = np.vstack((hx, np.zeros_like(hz)))
        expected_z = np.vstack((np.zeros_like(hx), hz))
        if not np.array_equal(cx, expected_x) or not np.array_equal(cz, expected_z):
            raise ValueError("Symplectic checks do not match hx/hz")
        rebuilt_x, rebuilt_z, _ = assemble_deformed(
            self._source_code, self.glue, pauli=self.metadata["pauli"],
            d_R=self.metadata["d_R"], max_entries=self.metadata["max_entries"])
        if not np.array_equal(hx, rebuilt_x) or not np.array_equal(hz, rebuilt_z):
            raise ValueError("Exported glue does not match the actual deformed checks")
        signs = binary_vector(self.check_signs, length=len(cx), name="check_signs")
        mx = binary_matrix(self.measurement_x, n_cols=self.original_n, name="measurement_x")
        mz = binary_matrix(self.measurement_z, n_cols=self.original_n, name="measurement_z")
        if mx.shape != mz.shape:
            raise ValueError("Measurement X/Z halves have different shapes")
        target_signs = binary_vector(self.measurement_signs, length=len(mx), name="measurement_signs")
        pauli = self.metadata["pauli"]
        if pauli not in ("X", "Z") or np.any(mz if pauli == "X" else mx):
            raise ValueError("Devised-sticking targets must have the selected pure X/Z type")
        self._validate_optimization(mx, mz, target_signs)
        witnesses = binary_matrix(self.measurement_witnesses, n_cols=len(cx), name="measurement_witnesses")
        if len(witnesses) != len(mx):
            raise ValueError("Each requested observable must have one witness")
        for x, z, sign, witness in zip(mx, mz, target_signs, witnesses):
            wx, wz, phase = pauli_product(cx, cz, signs, np.flatnonzero(witness))
            if not np.array_equal(wx, np.pad(x, (0, self.n - self.original_n))) or not np.array_equal(wz, np.pad(z, (0, self.n - self.original_n))) or phase != 2 * int(sign):
                raise ValueError("Measurement witness does not equal the requested signed observable")
        dependencies = nullspace(np.hstack((cx, cz)).T)
        if np.any(matmul(dependencies, signs)):
            raise ValueError("Signed checks contain a dependency producing -I")
        targets = mz if pauli == "Z" else mx
        validate_readout(self._source_code, hx, hz, targets, pauli=pauli,
                         max_entries=self.metadata["max_entries"])
        px = binary_matrix(self.protected_x, n_cols=self.n, name="protected_x")
        pz = binary_matrix(self.protected_z, n_cols=self.n, name="protected_z")
        if px.shape != (self.protected_k, self.n) or pz.shape != px.shape:
            raise ValueError("Protected logical bases have inconsistent dimensions")
        if np.any(matmul(px, hz.T)) or np.any(matmul(pz, hx.T)):
            raise ValueError("Protected logical operators do not commute with the checks")
        if not np.array_equal(matmul(px, pz.T), np.eye(self.protected_k, dtype=np.uint8)):
            raise ValueError("Protected logical bases are not paired")
        if self.gauge_qubits < 0:
            raise ValueError("Deformation removed additional protected logical information")
        return True

    def _validate_optimization(self, mx, mz, target_signs):
        """Recompute each signed equivalence from original check coefficients."""
        from ..logical_optimization import verify_logical_optimization

        try:
            enabled = self.metadata["optimize_logical"]
            if not isinstance(enabled, (bool, np.bool_)):
                raise ValueError("optimize_logical metadata must be a Boolean")
            original_x = binary_matrix(self.metadata["input_measurement_x"],
                                       n_cols=self.original_n, name="input_measurement_x")
            original_z = binary_matrix(self.metadata["input_measurement_z"],
                                       n_cols=self.original_n, name="input_measurement_z")
            original_signs = binary_vector(self.metadata["input_measurement_signs"],
                                          length=len(mx), name="input_measurement_signs")
            reports = self.metadata["logical_optimization"]
            if original_x.shape != mx.shape or original_z.shape != mz.shape:
                raise ValueError("Original and selected targets have inconsistent dimensions")
            if np.any(original_z if self.metadata["pauli"] == "X" else original_x):
                raise ValueError("Original devised-sticking targets must have the selected pure X/Z type")
            if not isinstance(reports, list) or len(reports) != len(mx):
                raise ValueError("Each devised-sticking target must retain one optimization report")
            basis = logical_basis_for(self._source_code)
            for index, report in enumerate(reports):
                if not isinstance(report, dict):
                    raise ValueError("Logical optimization report must be a dictionary")
                if (not isinstance(report.get("enabled"), (bool, np.bool_))
                        or bool(report["enabled"]) != bool(enabled)):
                    raise ValueError("Logical optimization report disagrees with the selected mode")
                if (not np.array_equal(report["original_x"], original_x[index])
                        or not np.array_equal(report["original_z"], original_z[index])
                        or report["original_sign"] != original_signs[index]):
                    raise ValueError("Logical optimization report does not match its original input target")
                chosen = (mx[index], mz[index], int(target_signs[index]))
                verify_logical_optimization(self._source_code, chosen, basis, report)
                if not enabled and (
                        report.get("status") != "disabled"
                        or np.any(report["hx_coefficients"]) or np.any(report["hz_coefficients"])
                        or not np.array_equal(mx[index], original_x[index])
                        or not np.array_equal(mz[index], original_z[index])
                        or target_signs[index] != original_signs[index]):
                    raise ValueError("Disabled optimization must preserve the supplied physical target exactly")
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Malformed devised-sticking logical optimization metadata") from error

    def summary(self):
        return {
            "scheme": "zhang_li_devised_sticking",
            "original_n": self.original_n,
            "deformed_n": self.n,
            "ancilla_qubits": self.n - self.original_n,
            "checks": len(self.check_x),
            "measurements": [pauli_string(x, z, int(s)) for x, z, s in zip(self.measurement_x, self.measurement_z, self.measurement_signs)],
            "optimize_logical": self.metadata["optimize_logical"],
            "input_measurement_weights": [entry["before_weight"] for entry in self.metadata["logical_optimization"]],
            "selected_measurement_weights": [entry["after_weight"] for entry in self.metadata["logical_optimization"]],
            "logical_optimization_statuses": [entry["status"] for entry in self.metadata["logical_optimization"]],
            "logical_measurement_rank": self.metadata["logical_measurement_rank"],
            "protected_k": self.protected_k,
            "stabilizer_k": self.k_stabilizer,
            "gauge_qubits": self.gauge_qubits,
            "is_css": True,
            "d_R": self.metadata["d_R"],
            "glue_bits": self.glue.h_g.shape[1],
            "glue_checks": self.glue.h_g.shape[0],
            "ldpc": self.metadata["ldpc"],
            "distance_lower_bound": self.metadata["distance_lower_bound"],
            "distance_preservation_guaranteed": self.metadata["distance_preservation_guaranteed"],
        }

    def save(self, path):
        """Validate before exporting; invalid outputs never overwrite a file."""
        path = Path(path)
        if path.suffix not in (".json", ".npz"):
            raise ValueError("Output extension must be .json or .npz")
        self.validate()
        arrays = {name: getattr(self, name) for name in (
            "hx", "hz", "check_x", "check_z", "check_signs", "measurement_x", "measurement_z",
            "measurement_signs", "measurement_witnesses", "protected_x", "protected_z")}
        arrays.update(h_g=self.glue.h_g, s=self.glue.s, t=self.glue.t)
        information = {"summary": self.summary(), "metadata": self.metadata, "glue_metadata": self.glue.metadata}
        if path.suffix == ".json":
            path.write_text(json.dumps(_jsonable({**arrays, **information}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            np.savez_compressed(path, **arrays, information=json.dumps(_jsonable(information), ensure_ascii=False))
