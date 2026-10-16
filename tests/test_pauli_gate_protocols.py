"""Independent matrix/instrument checks, including all measurement branches."""
from itertools import product
import numpy as np
import pytest

from codesurgery.pauli_gates import decompose_gates, MeasurementStep, BitExpr


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j])
PAULI = dict(I=I, X=X, Y=Y, Z=Z)


def matrix(text):
    out = np.array([[1]], complex)
    for axis in text:
        out = np.kron(out, PAULI[axis])
    return out


def embed_input(k, ancilla=0):
    """Isometry |data> -> |0>_ancilla |data>, arbitrary original slot."""
    out = np.zeros((2 ** k, 2 ** (k - 1)), complex)
    for j in range(2 ** (k - 1)):
        bits = list(f'{j:0{k-1}b}')
        bits.insert(ancilla, '0')
        out[int(''.join(bits), 2), j] = 1
    return out


def gate_matrix(k, name, qubits):
    if name in ('H', 'S'):
        result = np.array([[1]], complex)
        for q in range(k):
            result = np.kron(result, (H if name == 'H' else S) if q == qubits[0] else I)
        return result
    control, target = qubits
    result = np.zeros((2 ** k, 2 ** k), complex)
    for j in range(2 ** k):
        bit = (j >> (k - 1 - control)) & 1
        i = j ^ (bit << (k - 1 - target))
        result[i, j] = 1
    return result


def equivalent(actual, expected):
    alpha = np.vdot(expected, actual) / np.vdot(expected, expected)
    assert np.allclose(actual, alpha * expected, atol=1e-11)
    return abs(alpha) ** 2


@pytest.mark.parametrize('name,k,qubits', [('H', 2, (1,)), ('S', 2, (1,)), ('CNOT', 3, (1, 2))])
def test_all_frames_and_all_branch_kraus_operators(name, k, qubits):
    # Full operator equality proves correctness even with an entangled spectator.
    plan = decompose_gates(k, [(name, *qubits)])
    input_map = embed_input(k)
    desired = gate_matrix(k, name, qubits) @ input_map
    projectors = [(np.eye(2 ** k) + matrix(m.logical_pauli)) / 2 for m in plan.measurements]
    nonzero = 0
    for labels in product('IXYZ', repeat=k):
        initial_frame = ''.join(labels)
        actual_input = matrix(initial_frame) @ input_map
        for outcomes in product((0, 1), repeat=len(plan.measurements)):
            branch = actual_input.copy()
            for p, bit in zip(projectors, outcomes):
                branch = (p if not bit else np.eye(2 ** k) - p) @ branch
            if np.linalg.norm(branch) < 1e-12:
                continue
            response = plan.resolve(outcomes, initial_frame=initial_frame)
            equivalent(branch, matrix(response['correction_pauli']) @ desired)
            nonzero += 1
    assert nonzero > 0


def test_composed_noncommuting_gates_all_outcome_branches():
    gates = [('H', 1), ('S', 2), ('CNOT', 1, 2)]
    plan = decompose_gates(3, gates)
    initial = embed_input(3)
    desired = initial.copy()
    for name, *qubits in gates:
        desired = gate_matrix(3, name, qubits) @ desired
    positive = [(np.eye(8) + matrix(m.logical_pauli)) / 2 for m in plan.measurements]
    total_probability = 0
    for outcomes in product((0, 1), repeat=len(plan.measurements)):
        branch = initial.copy()
        for projector, bit in zip(positive, outcomes):
            branch = (projector if not bit else np.eye(8) - projector) @ branch
        if np.linalg.norm(branch) < 1e-12:
            continue
        response = plan.resolve(outcomes)
        total_probability += equivalent(branch, matrix(response['correction_pauli']) @ desired)
    assert total_probability == pytest.approx(1)


@pytest.mark.parametrize('ancilla', [0, 1, 2])
def test_ancilla_can_be_reassigned_without_relabeling_data(ancilla):
    operands = [q for q in range(3) if q != ancilla]
    plan = decompose_gates(3, [('CNOT', *operands)], ancilla=ancilla)
    initial = embed_input(3, ancilla)
    desired = gate_matrix(3, 'CNOT', operands) @ initial
    for outcomes in product((0, 1), repeat=4):
        branch = initial.copy()
        for measurement, bit in zip(plan.measurements, outcomes):
            branch = (np.eye(8) + (-1) ** bit * matrix(measurement.logical_pauli)) @ branch / 2
        response = plan.resolve(outcomes)
        equivalent(branch, matrix(response['correction_pauli']) @ desired)


def test_raw_affine_frame_formulas_agree_with_effective_interpreter():
    plan = decompose_gates(3, [('H', 1), ('S', 1), ('CNOT', 1, 2), ('H', 2)])
    formulas = plan.frame_expressions()
    rng = np.random.default_rng(9)
    for _ in range(40):
        outcomes = {m.id: int(rng.integers(2)) for m in plan.measurements}
        response = plan.resolve(outcomes)
        for key, answer in [('x', 'frame_x'), ('z', 'frame_z')]:
            assert [BitExpr(tuple(f['xor']), f['constant']).evaluate(outcomes) for f in formulas[key]] == response[answer]
        for key, formula in formulas['effective_outcomes'].items():
            assert BitExpr(tuple(formula['xor']), formula['constant']).evaluate(outcomes) == response['effective_outcomes'][key]


def test_outcome_formats_and_exact_names():
    plan = decompose_gates(2, [('S', 1)])
    assert plan.resolve([1, 0, 1]) == plan.resolve([-1, 1, -1], outcome_format='eigenvalues')
    with pytest.raises(ValueError):
        plan.resolve([0, 1])
    with pytest.raises(ValueError):
        plan.resolve({m.id: 0 for m in plan.measurements} | {'extra': 0})
    for invalid in ([0, 2, 0], [True, 0, 0], [0.0, 0, 0], '010'):
        with pytest.raises(ValueError):
            plan.resolve(invalid)
    with pytest.raises(ValueError):
        plan.resolve([0, 1, 1], outcome_format='eigenvalues')


@pytest.mark.parametrize('k,gates,ancilla', [
    (1, [('H', 0)], 0), (2, [('H', 0)], 0), (2, [('CNOT', 1, 1)], 0),
    (2, [('H', 2)], 0), (3, [('CNOT', 0, 1)], 0), (3, [('T', 1)], 0),
    (3, [('H', True)], 0), (3, [('H', 1)], True), (3, [('H', 1)], 3),
    (3, [], 0), (3, [{'gate':'H', 'target':1, 'control':2}], 0),
    (3, [{'gate':'CNOT', 'control':1}], 0),
])
def test_rejects_bad_or_reserved_operands(k, gates, ancilla):
    with pytest.raises(ValueError):
        decompose_gates(k, gates, ancilla=ancilla)


def test_qwc_is_only_required_within_a_measurement_not_across_steps():
    plan = decompose_gates(2, [('H', 1)])
    a, b = [matrix(m.logical_pauli) for m in plan.measurements if m.role == 'joint']
    assert np.allclose(a @ b, -b @ a)
    assert all(m.to_dict()['execution'] == 'merge_measure_split' for m in plan.measurements)


def test_gate_alias_and_dictionary():
    plan = decompose_gates(3, [{'gate':'h', 'target':1}, {'name':'s','qubits':[2]}, ('CX', 2, 1)])
    assert [g.name for g in plan.gates] == ['H', 'S', 'CNOT']
