"""Exportable shared-ancilla deformations, with independently checked readout."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from ..gf2 import binary_matrix, binary_vector, matmul, rank, solve
from ..symplectic import pauli_product, pauli_string
from .validation import (merged_checks, check_chain_maps, inspect_readout,
                         validate_signed_group, readout_witnesses, weight_report)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, slice):
        return dict(start=value.start, stop=value.stop, step=value.step)
    return value


def physical_checks(prepared, state, ancilla_hadamard=False):
    x, z = merged_checks(prepared['source_x'], prepared['source_z'], state)
    if ancilla_hadamard:
        n = prepared['source_x'].shape[1]
        x[:, n:], z[:, n:] = z[:, n:].copy(), x[:, n:].copy()
    return x, z


@dataclass
class HighRateCode:
    code: object
    state: object
    prepared: dict
    check_x: np.ndarray
    check_z: np.ndarray
    check_signs: np.ndarray
    measurement_witnesses: np.ndarray
    readout_sign_corrections: np.ndarray
    max_check_weight: int
    max_qubit_degree: int
    ancilla_hadamard: bool = False
    metadata: dict = field(default_factory=dict)
    subsystem: dict | None = None

    @property
    def n(self): return self.check_x.shape[1]

    @property
    def original_n(self): return self.code.n

    @property
    def protected_k(self): return self.code.k - self.prepared['target_rank']

    @property
    def k_stabilizer(self): return self.n - rank(np.hstack((self.check_x, self.check_z)))

    @property
    def gauge_qubits(self): return self.k_stabilizer - self.protected_k

    @property
    def is_css(self):
        return not np.any(np.any(self.check_x, axis=1) & np.any(self.check_z, axis=1))

    @property
    def hx(self):
        return self.check_x[np.any(self.check_x, axis=1)] if self.is_css else None

    @property
    def hz(self):
        return self.check_z[np.any(self.check_z, axis=1)] if self.is_css else None

    @property
    def target_x(self): return self.prepared['target_x']

    @property
    def target_z(self): return self.prepared['target_z']

    @property
    def target_signs(self): return self.prepared['target_signs']

    def validate(self):
        p = self.prepared
        check_chain_maps(p['source_x'], p['source_z'], self.state)
        x = binary_matrix(self.check_x, name='check_x')
        z = binary_matrix(self.check_z, n_cols=x.shape[1], name='check_z')
        if x.shape != z.shape:
            raise ValueError('Mismatched symplectic checks')
        signs = binary_vector(self.check_signs, length=len(x), name='check_signs')
        expected = physical_checks(p, self.state, self.ancilla_hadamard)
        if not np.array_equal(x, expected[0]) or not np.array_equal(z, expected[1]):
            raise ValueError('Deformed checks no longer match the exported chain maps')
        readout = inspect_readout(self.code, p['basis'], x, z, p['target_signatures'])
        validate_signed_group(self.code, p['source_x'], p['source_z'], x, z, signs, readout)
        witnesses, corrections = readout_witnesses(x, z, signs, self.target_x,
                                                  self.target_z, self.target_signs)
        # Accept any valid witness, not just solve()'s particular representative.
        w = binary_matrix(self.measurement_witnesses, n_cols=len(x), name='measurement_witnesses')
        if w.shape[0] != len(witnesses):
            raise ValueError('One measurement witness is required per target')
        corr = binary_vector(self.readout_sign_corrections, length=len(w), name='readout_sign_corrections')
        for i, row in enumerate(w):
            px, pz, phase = pauli_product(x, z, signs, np.flatnonzero(row))
            if (not np.array_equal(px[:self.code.n], self.target_x[i])
                    or not np.array_equal(pz[:self.code.n], self.target_z[i])
                    or np.any(px[self.code.n:] | pz[self.code.n:])
                    or phase != 2*(int(corr[i]) ^ int(self.target_signs[i]))):
                raise ValueError('Signed target readout witness failed')
        # Recheck original-vs-reduced representatives against the ORIGINAL code.
        difference = np.hstack((p['original_target_x'] ^ self.target_x,
                                p['original_target_z'] ^ self.target_z))
        source_coeff = solve(np.hstack((p['source_x'], p['source_z'])).T, difference.T).T
        for i, row in enumerate(source_coeff):
            xx = np.vstack((p['original_target_x'][i:i+1], p['source_x']))
            zz = np.vstack((p['original_target_z'][i:i+1], p['source_z']))
            ss = np.zeros(len(xx), np.uint8)
            ss[0] = p['original_target_signs'][i]
            _, _, phase = pauli_product(xx, zz, ss, [0, *(np.flatnonzero(row)+1).tolist()])
            if phase != 2*int(self.target_signs[i]):
                raise ValueError('Support reduction changed the original target sign')
        weights = weight_report(x, z)
        if weights['max_check_weight'] > self.max_check_weight or weights['max_qubit_degree'] > self.max_qubit_degree:
            raise ValueError('Complete deformed check matrix exceeds the configured row/column weight bound')
        if self.gauge_qubits < 0:
            raise ValueError('Extra protected logical information was removed')
        if self.subsystem is not None:
            self._validate_subsystem()
        return True

    def _validate_subsystem(self):
        if not self.is_css:
            raise ValueError('Only a CSS protected-subsystem certificate is supported')
        s = self.subsystem
        gx = binary_matrix(s['gauge_x'], n_cols=self.n)
        gz = binary_matrix(s['gauge_z'], n_cols=self.n)
        lx = binary_matrix(s['bare_x'], n_cols=self.n)
        lz = binary_matrix(s['bare_z'], n_cols=self.n)
        if len(gx) != self.gauge_qubits or len(gz) != len(gx) or len(lx) != self.protected_k or len(lz) != len(lx):
            raise ValueError('Subsystem logical/gauge counts do not match the final code')
        if any(np.any(matmul(a, b.T)) for a, b in
               [(gx, self.hz), (gz, self.hx), (lx, self.hz), (lz, self.hx), (gx, lz), (gz, lx)]):
            raise ValueError('Subsystem operators have invalid commutation relations')
        if not np.array_equal(matmul(gx, gz.T), np.eye(len(gx), dtype=np.uint8)):
            raise ValueError('Gauge pairs are not canonical')
        if not np.array_equal(matmul(lx, lz.T), np.eye(len(lx), dtype=np.uint8)):
            raise ValueError('Protected logical pairs are not canonical')

    def summary(self):
        weights = weight_report(self.check_x, self.check_z)
        distance = self.metadata.get('distance', {})
        return dict(scheme='high_rate_surgery', paper='arXiv:2510.08523v1',
                    original_n=self.code.n, deformed_n=self.n, ancilla_qubits=self.n-self.code.n,
                    logical_measurement_rank=self.prepared['target_rank'],
                    target_count=len(self.target_x), protected_k=self.protected_k,
                    stabilizer_k=self.k_stabilizer, gauge_qubits=self.gauge_qubits,
                    layers=len(self.state.layers), is_css=self.is_css,
                    max_check_weight=weights['max_check_weight'], max_qubit_degree=weights['max_qubit_degree'],
                    check_weight_limit=self.max_check_weight, qubit_degree_limit=self.max_qubit_degree,
                    targets=[pauli_string(x,z,s) for x,z,s in zip(self.target_x,self.target_z,self.target_signs)],
                    status=self.metadata.get('status'), distance_status=distance.get('status'),
                    distance_lower_bound=distance.get('proven_lower_bound'),
                    distance_upper_bound=distance.get('upper_bound'),
                    protected_distance_certified=self.metadata.get('protected_distance_certified', False),
                    measurement_fault_distance_certified=False,
                    information_extraction_rate=self.prepared['target_rank']/max(1, sum(self.state.boundary.shape)+len(self.state.cycles)))

    def save(self, path):
        self.validate()
        path = Path(path)
        arrays = dict(check_x=self.check_x, check_z=self.check_z, check_signs=self.check_signs,
                      boundary=self.state.boundary, cycles=self.state.cycles, gamma0=self.state.gamma0,
                      attachment_x=self.state.attachment_x, attachment_z=self.state.attachment_z,
                      target_x=self.target_x, target_z=self.target_z, target_signs=self.target_signs,
                      measurement_witnesses=self.measurement_witnesses,
                      readout_sign_corrections=self.readout_sign_corrections,
                      original_hx=self.code.hx, original_hz=self.code.hz,
                      original_target_x=self.prepared['original_target_x'],
                      original_target_z=self.prepared['original_target_z'],
                      original_target_signs=self.prepared['original_target_signs'],
                      logical_jx=self.prepared['basis']['jx'], logical_jz=self.prepared['basis']['jz'])
        if self.is_css:
            arrays.update(hx=self.hx, hz=self.hz)
        if self.subsystem is not None:
            arrays.update({key:self.subsystem[key] for key in ['gauge_x','gauge_z','bare_x','bare_z']})
        information = dict(summary=self.summary(), metadata=self.metadata,
            construction_metadata=getattr(self.state, 'metadata', {}),
            ancilla_hadamard=self.ancilla_hadamard,
            readout_rule='target outcome = (-1)^readout_sign_corrections[i] times product of selected signed-check outcomes',
            source_distance=self.code.distance, source_name=self.code.name,
            basis_source=self.prepared['basis']['origin'],
            layers=[dict(internal_qubits=list(layer.internal_qubits), check_slice=layer.check_slice)
                    for layer in self.state.layers], bridges=self.state.bridges)
        if path.suffix == '.json':
            path.write_text(json.dumps(jsonable(dict(**arrays, **information)), ensure_ascii=False, indent=2)+'\n')
        elif path.suffix == '.npz':
            np.savez_compressed(path, **arrays, information=json.dumps(jsonable(information), ensure_ascii=False))
        else:
            raise ValueError('Output must be .json or .npz')
