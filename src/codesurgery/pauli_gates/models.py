"""Serializable sequential gate plans and an explicit raw-outcome Pauli frame."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ..models import _jsonable
from ..symplectic import parse_pauli, pauli_string
from .protocol import BitExpr, MeasurementStep, FrameUpdate, SOURCES, integer


@dataclass
class GatePlan:
    k: int
    ancilla: int
    gates: tuple
    events: tuple
    backend: str | None = None
    gadgets: dict = field(default_factory=dict, repr=False)
    source_code: object = field(default=None, repr=False)
    logical_basis: dict | None = field(default=None, repr=False)
    backend_options: dict = field(default_factory=dict)

    @property
    def measurements(self):
        return tuple(e for e in self.events if isinstance(e, MeasurementStep))

    def validate(self):
        from .protocol import decompose
        from ..models import CSSCode
        from ..gadget_common import logical_basis_for, prepare
        from .backends import _verify_output
        k, ancilla, specs, events = decompose(self.k, self.gates, ancilla=self.ancilla)
        if specs != self.gates or events != self.events:
            raise ValueError('Plan events/corrections do not implement its declared gate sequence')
        basis = None
        if self.source_code is not None:
            if not isinstance(self.source_code, CSSCode) or self.source_code.k != k:
                raise ValueError('Plan source code does not match its logical register')
            # Revalidate matrices too: a CSSCode object can have been mutated.
            source = CSSCode(self.source_code.hx, self.source_code.hz,
                             self.source_code.distance, self.source_code.name)
            basis = logical_basis_for(source, self.logical_basis)
        elif self.gadgets:
            raise ValueError('Compiled measurements require the original source code')
        seen = set()
        for event in self.events:
            x, z, sign = parse_pauli(event.logical_pauli, self.k)
            if sign or not np.any(x | z):
                raise ValueError('Plan events require unsigned nonidentity logical Paulis')
            if isinstance(event, MeasurementStep):
                if event.id in seen:
                    raise ValueError('Duplicate measurement ID')
                seen.add(event.id)
            elif isinstance(event, FrameUpdate):
                if not set(event.exponent.terms) <= seen:
                    raise ValueError('Correction depends on an outcome not yet measured')
            else:
                raise ValueError('Unknown plan event')
        if self.gadgets and set(self.gadgets) != seen:
            raise ValueError('A compiled plan needs a gadget for every measurement')
        checked = set()
        for measurement in self.measurements:
            if measurement.id not in self.gadgets:
                continue
            result = self.gadgets[measurement.id]
            key = (measurement.logical_pauli, id(result))
            if key in checked:
                continue
            a, b, sign = parse_pauli(measurement.logical_pauli, k)
            _, target, _ = prepare(source, dict(logical_x=a, logical_z=b, sign=sign),
                                   logical_basis=basis)
            _verify_output(result, source, target, basis)
            checked.add(key)
        return True

    def summary(self):
        return dict(k=self.k, ancilla=self.ancilla,
                    data_logicals=[q for q in range(self.k) if q != self.ancilla],
                    gate_count=len(self.gates), measurement_count=len(self.measurements),
                    compiled_measurements=len(self.gadgets),
                    unique_deformations=len({id(g) for g in self.gadgets.values()}),
                    backend=self.backend,
                    peak_physical_qubits=max((g.n for g in self.gadgets.values()), default=None),
                    logical_output_slots='unchanged',
                    ancilla_final_state='+Z in the corrected frame',
                    measurement_circuit_distance_certified=False)

    def frame_expressions(self):
        """Final correction as affine expressions in raw bits, identity initial frame."""
        fx, fz = [BitExpr() for _ in range(self.k)], [BitExpr() for _ in range(self.k)]
        effective = {}
        for event in self.events:
            x, z, _ = parse_pauli(event.logical_pauli, self.k)
            if isinstance(event, MeasurementStep):
                value = BitExpr((event.id,))
                for q in range(self.k):
                    if z[q]:
                        value ^= fx[q]
                    if x[q]:
                        value ^= fz[q]
                effective[event.id] = value
            else:
                value = BitExpr(constant=event.exponent.constant)
                for name in event.exponent.terms:
                    value ^= effective[name]
                for q in range(self.k):
                    if x[q]:
                        fx[q] ^= value
                    if z[q]:
                        fz[q] ^= value
        return dict(x=[v.to_dict() for v in fx], z=[v.to_dict() for v in fz],
                    effective_outcomes={name:v.to_dict() for name,v in effective.items()},
                    variables='raw_outcome_bits', initial_frame='I' * self.k)

    def resolve(self, outcomes, *, outcome_format='bits', initial_frame=None):
        """Track rather than physically apply all conditional Pauli corrections.

        Raw inputs are decoded outcomes of the requested *signed logical*
        measurements after the backend's witness/sign and split corrections.
        They are not unprocessed individual check outcomes.  For each P,
        effective_bit = raw_bit XOR symplectic(frame, P). Global phases of
        the final Pauli are irrelevant. No sampling or hardware execution occurs.
        """
        self.validate()
        if outcome_format not in ('bits', 'eigenvalues'):
            raise ValueError('outcome_format must be bits or eigenvalues')
        ids = [m.id for m in self.measurements]
        if isinstance(outcomes, Mapping):
            if set(outcomes) != set(ids):
                raise ValueError('Provide exactly one outcome for every measurement ID')
            raw = dict(outcomes)
        else:
            if isinstance(outcomes, (str, bytes)):
                raise ValueError('outcomes must be a mapping or a sequence of numbers')
            try:
                values = list(outcomes)
            except TypeError as error:
                raise ValueError('outcomes must be a mapping or sequence') from error
            if len(values) != len(ids):
                raise ValueError('Outcome count does not match measurement count')
            raw = dict(zip(ids, values))
        for name, value in raw.items():
            if outcome_format == 'bits':
                value = integer(value, f'outcome {name}')
                if value > 1:
                    raise ValueError('Outcome bits must be 0 or 1')
                raw[name] = value
            else:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (-1, 1):
                    raise ValueError('Eigenvalues must be +1 or -1')
                raw[name] = int(value == -1)
        fx, fz, _ = parse_pauli('I' * self.k if initial_frame is None else initial_frame, self.k)
        effective, flips, trace = {}, {}, []
        for event in self.events:
            x, z, _ = parse_pauli(event.logical_pauli, self.k)
            if isinstance(event, MeasurementStep):
                flip = int((int(fx @ z) + int(fz @ x)) % 2)
                flips[event.id] = flip
                effective[event.id] = raw[event.id] ^ flip
            else:
                bit = event.exponent.evaluate(effective)
                if bit:
                    fx ^= x
                    fz ^= z
                trace.append(dict(gate_index=event.gate_index, reason=event.reason,
                                  logical_pauli=event.logical_pauli, applied_to_frame=bool(bit)))
        data_x, data_z = fx.copy(), fz.copy()
        data_x[self.ancilla] = data_z[self.ancilla] = 0
        return dict(raw_outcomes=raw, effective_outcomes=effective,
                    measurement_sign_flips=flips, frame_x=fx.tolist(), frame_z=fz.tolist(),
                    correction_pauli=pauli_string(fx, fz),
                    data_correction_pauli=pauli_string(data_x, data_z),
                    ancilla_correction_pauli=pauli_string(fx ^ data_x, fz ^ data_z), trace=trace,
                    convention='physical_state = final_frame * ideal_corrected_state (up to global phase)',
                    correction_action='apply correction_pauli or carry the frame to later measurements')

    def to_dict(self):
        self.validate()
        source = None if self.source_code is None else dict(
            name=self.source_code.name, n=self.source_code.n, k=self.source_code.k,
            distance=self.source_code.distance, hx=self.source_code.hx, hz=self.source_code.hz)
        out = dict(format_version=1, summary=self.summary(),
                   gates=[g.to_dict() for g in self.gates], operations=[e.to_dict() for e in self.events],
                   frame_expressions=self.frame_expressions(), source_code=source,
                   logical_basis=self.logical_basis, backend_options=self.backend_options,
                   deformations={name:result.summary() for name,result in self.gadgets.items()},
                   sources=SOURCES,
                   assumptions=[
                       'The reserved logical ancilla carries no user data and may be reinitialized.',
                       'Each Pauli measurement is a complete nondestructive merge/measure/split operation on the original memory code.',
                       'Backend readout decoding and split-induced Pauli corrections precede the logical outcome/frame interface.',
                       'Static deformed matrices do not certify a measurement circuit or its noisy spacetime distance.',
                   ])
        return _jsonable(out)

    def save(self, directory, *, matrix_format='npz', overwrite=False):
        self.validate()
        if matrix_format not in ('npz', 'json'):
            raise ValueError('matrix_format must be npz or json')
        if not isinstance(overwrite, bool):
            raise ValueError('overwrite must be a boolean')
        directory = Path(directory)
        names = ['plan.json'] + [f'{m.id}.{matrix_format}' for m in self.measurements if m.id in self.gadgets]
        if not overwrite and any((directory / name).exists() for name in names):
            raise FileExistsError('Plan output files exist; use another directory or overwrite=True')
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for measurement in self.measurements:
            if measurement.id in self.gadgets:
                name = f'{measurement.id}.{matrix_format}'
                self.gadgets[measurement.id].save(directory / name)
                files[measurement.id] = name
        document = self.to_dict()
        document['deformed_code_files'] = files
        path = directory / 'plan.json'
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return path
