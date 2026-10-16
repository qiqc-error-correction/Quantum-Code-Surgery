"""Compile H/S/CNOT into original-code Pauli measurement surgery gadgets."""
from copy import deepcopy

from ..models import CSSCode
from ..gadget_common import logical_basis_for
from .protocol import decompose
from .models import GatePlan


def decompose_gates(k, gates, *, ancilla=0):
    """Logical schedule only; no code matrices or distance calculations."""
    k, ancilla, gates, events = decompose(k, gates, ancilla=ancilla)
    result = GatePlan(k, ancilla, gates, events)
    result.validate()
    return result


def compile_gates(code, gates, *, ancilla=0, logical_basis=None,
                  backend='gauging', backend_options=None, build=True):
    """Compile a sequential Clifford circuit with one reserved logical ancilla.

    Indices refer to the input paired logical basis (zero-based). The first
    logical qubit, index 0, is reserved by default; it is prepared and reset
    through measurement and tracked Pauli corrections. Data slots stay fixed.
    Backend defaults, including automatic distance processing, are preserved.
    Every primitive acts on the original memory after the previous split.
    """
    if not isinstance(code, CSSCode):
        raise TypeError('code must be a CSSCode')
    if not isinstance(build, bool):
        raise ValueError('build must be a boolean')
    if backend_options is not None and not isinstance(backend_options, dict):
        raise ValueError('backend_options must be a dictionary')
    plan = decompose_gates(code.k, gates, ancilla=ancilla)
    plan.source_code = CSSCode(code.hx.copy(), code.hz.copy(), code.distance, code.name)
    plan.logical_basis = deepcopy(logical_basis_for(plan.source_code, logical_basis))
    plan.backend_options = deepcopy(backend_options or {})
    from .backends import build_measurement, describe_backend
    plan.backend = describe_backend(backend)['name']
    if build:
        cache = {}
        for measurement in plan.measurements:
            key = measurement.logical_pauli
            if key not in cache:
                try:
                    cache[key] = build_measurement(plan.source_code, key,
                        logical_basis=plan.logical_basis, backend=plan.backend,
                        backend_options=plan.backend_options)
                except ValueError as error:
                    # Keep backend errors and their candidate result intact.
                    error.measurement_id = measurement.id
                    error.gate_index = measurement.gate_index
                    raise
            plan.gadgets[measurement.id] = cache[key]
    plan.validate()
    return plan


def compile_gate(code, gate, *, target=None, control=None, ancilla=0, **kwargs):
    """Convenience entry; omitted operands use the first available data slots."""
    if not isinstance(code, CSSCode):
        raise TypeError('code must be a CSSCode')
    if not isinstance(gate, str):
        raise ValueError('gate must be H, S, or CNOT')
    name = gate.upper()
    available = [q for q in range(code.k) if q != ancilla]
    if name in ('CNOT', 'CX'):
        if control is None:
            if not available:
                raise ValueError('CNOT needs two data logicals and an ancilla')
            control = available[0]
        if target is None:
            choices = [q for q in available if q != control]
            if not choices:
                raise ValueError('CNOT needs two data logicals and an ancilla')
            target = choices[0]
        spec = (name, control, target)
    else:
        if control is not None:
            raise ValueError('control is only valid for CNOT')
        if target is None:
            if not available:
                raise ValueError('H/S need a data logical and an ancilla')
            target = available[0]
        spec = (name, target)
    return compile_gates(code, [spec], ancilla=ancilla, **kwargs)
