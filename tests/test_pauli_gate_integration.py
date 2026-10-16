import json
import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, direct_sum, steane_code
from codesurgery.pauli_gates import compile_gate, compile_gates
from codesurgery.symplectic import commutation_matrix


@pytest.mark.parametrize('gate', ['H', 'S'])
def test_gauging_default_small_code_with_distance_enabled(gate):
    plan = compile_gate(four_qubit_code(), gate)
    assert plan.ancilla == 0
    assert plan.gates[0].qubits == (1,)
    assert plan.validate()
    for result in plan.gadgets.values():
        assert not np.any(commutation_matrix(result.check_x, result.check_z))
        assert result.original_n == 4
        assert result.protected_k == 1
        assert 'distance_calculation' in result.metadata


def test_cnot_keeps_original_memory_for_every_measurement(tmp_path):
    code = direct_sum(four_qubit_code(), four_qubit_code())
    plan = compile_gate(code, 'CNOT', backend_options={'require_distance':False})
    assert plan.gates[0].qubits == (1, 2)
    assert len(plan.gadgets) == 4
    assert all(g.original_n == 8 and g.protected_k == 3 for g in plan.gadgets.values())
    path = plan.save(tmp_path / 'gate', matrix_format='json')
    saved = json.loads(path.read_text())
    assert saved['summary']['measurement_count'] == 4
    assert len(saved['deformed_code_files']) == 4
    for filename in saved['deformed_code_files'].values():
        checks = json.loads((path.parent / filename).read_text())
        assert 'check_x' in checks and 'check_z' in checks and 'check_signs' in checks
    with pytest.raises(FileExistsError):
        plan.save(path.parent, matrix_format='json')


def test_cache_reuses_only_equal_observables_and_fixed_basis():
    plan = compile_gates(four_qubit_code(), [('H', 1), ('S', 1)],
                         backend_options={'require_distance':False})
    assert plan.gadgets['g0_prep'] is plan.gadgets['g0_m3']
    assert plan.gadgets['g0_m3'] is plan.gadgets['g1_m2']
    assert plan.summary()['unique_deformations'] < plan.summary()['measurement_count']
    assert plan.summary()['measurement_circuit_distance_certified'] is False


def test_decomposition_only_uses_basis_but_builds_no_gadgets():
    plan = compile_gate(four_qubit_code(), 'H', build=False)
    assert plan.gadgets == {}
    assert plan.logical_basis['origin'] == 'Zhang--Li standard form'
    assert len(plan.measurements) == 4
    with pytest.raises(ValueError, match='reserved'):
        compile_gate(four_qubit_code(), 'H', target=0, build=False)
    with pytest.raises(ValueError):
        compile_gate(steane_code(), 'H', build=False)


def test_custom_paired_basis_and_alternative_backend():
    code = four_qubit_code()
    logicals = code.standard_logicals()
    basis = dict(jx=logicals.jx[::-1], jz=logicals.jz[::-1])
    plan = compile_gate(code, 'S', logical_basis=basis, backend='high_rate',
                        backend_options={'ensure_distance':False})
    assert plan.validate()
    assert np.array_equal(plan.logical_basis['jx'], basis['jx'])
    assert plan.logical_basis['origin'] == 'user supplied paired CSS basis'


def test_backend_error_keeps_measurement_location_and_no_silent_fallback():
    with pytest.raises(ValueError) as caught:
        compile_gate(four_qubit_code(), 'S', backend='devised_sticking')
    assert caught.value.backend == 'devised_sticking'
    assert caught.value.measurement_id == 'g0_m1'
    assert caught.value.gate_index == 0


def test_valid_gadget_for_wrong_step_is_rejected():
    plan = compile_gate(four_qubit_code(), 'H', backend_options={'require_distance':False})
    plan.gadgets['g0_m1'] = plan.gadgets['g0_prep']
    with pytest.raises(ValueError):
        plan.validate()


def test_modified_correction_or_step_id_is_rejected(tmp_path):
    from dataclasses import replace
    from codesurgery.pauli_gates import BitExpr
    plan = compile_gate(four_qubit_code(), 'H', build=False)
    events = list(plan.events)
    events[-2] = replace(events[-2], exponent=BitExpr())
    plan.events = tuple(events)
    with pytest.raises(ValueError, match='declared gate sequence'):
        plan.save(tmp_path / 'bad')
