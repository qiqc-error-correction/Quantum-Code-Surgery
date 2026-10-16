"""One reusable logical ancilla; ideal Pauli measurement identities.

Data stays in its original logical slots.  Each measurement is a complete
non-destructive merge/measure/split primitive, not another check added to the
previous deformed code.  Corrections are interpreted in an effective Pauli
frame by models.GatePlan.resolve.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class BitExpr:
    """An affine XOR expression in effective measurement outcome bits."""
    terms: tuple[str, ...] = ()
    constant: int = 0

    def __post_init__(self):
        c = integer(self.constant, 'bit constant')
        if c > 1:
            raise ValueError('bit constant must be 0 or 1')
        terms = set()
        for term in self.terms:
            if not isinstance(term, str) or not term:
                raise ValueError('BitExpr terms must be nonempty outcome IDs')
            if term in terms:
                terms.remove(term)
            else:
                terms.add(term)
        object.__setattr__(self, 'terms', tuple(sorted(terms)))
        object.__setattr__(self, 'constant', c)

    def __xor__(self, other):
        if isinstance(other, Integral):
            other = BitExpr(constant=other)
        if not isinstance(other, BitExpr):
            return NotImplemented
        return BitExpr(self.terms + other.terms, self.constant ^ other.constant)

    def evaluate(self, outcomes):
        result = self.constant
        for term in self.terms:
            value = integer(outcomes[term], f'outcome {term}')
            if value > 1:
                raise ValueError('outcome bits must be 0 or 1')
            result ^= value
        return result

    def to_dict(self):
        return dict(xor=list(self.terms), constant=self.constant)

    def __str__(self):
        return ' ^ '.join(([str(self.constant)] if self.constant else []) + list(self.terms)) or '0'


@dataclass(frozen=True)
class GateSpec:
    name: str
    qubits: tuple[int, ...]

    def to_dict(self):
        return dict(gate=self.name, qubits=list(self.qubits))


@dataclass(frozen=True)
class MeasurementStep:
    id: str
    logical_pauli: str
    gate_index: int
    role: str

    def to_dict(self):
        return dict(kind='measure', id=self.id, logical_pauli=self.logical_pauli,
                    gate_index=self.gate_index, role=self.role,
                    execution='merge_measure_split', outcome_convention='eigenvalue=(-1)^bit')


@dataclass(frozen=True)
class FrameUpdate:
    logical_pauli: str
    exponent: BitExpr
    gate_index: int
    reason: str

    def to_dict(self):
        return dict(kind='frame_update', logical_pauli=self.logical_pauli,
                    exponent=self.exponent.to_dict(), gate_index=self.gate_index,
                    reason=self.reason, outcome_basis='effective')


def _gate(value, k, ancilla):
    if isinstance(value, GateSpec):
        name, qubits = value.name, value.qubits
    elif isinstance(value, Mapping):
        unknown = set(value) - {'gate', 'name', 'qubits', 'target', 'control'}
        if unknown:
            raise ValueError(f'Unknown gate fields: {sorted(unknown)}')
        if 'name' in value and 'gate' in value:
            raise ValueError('Use gate or name, not both')
        name = value.get('gate', value.get('name'))
        if 'qubits' in value:
            if 'target' in value or 'control' in value:
                raise ValueError('Use qubits or target/control, not both')
            qubits = value['qubits']
        elif str(name).upper() in ('CNOT', 'CX'):
            if 'control' not in value or 'target' not in value:
                raise ValueError('CNOT needs control and target')
            qubits = (value['control'], value['target'])
        else:
            if 'control' in value or 'target' not in value:
                raise ValueError('H/S need target and no control')
            qubits = (value['target'],)
    elif isinstance(value, (tuple, list)) and value:
        name, *qubits = value
    else:
        raise ValueError('Each gate must be (name, qubits...) or a gate dictionary')
    if not isinstance(name, str):
        raise ValueError('Gate name must be H, S, or CNOT')
    name = name.upper()
    if name == 'CX':
        name = 'CNOT'
    if name not in ('H', 'S', 'CNOT'):
        raise ValueError(f'Unsupported gate {name}; choose H, S, or CNOT')
    if not isinstance(qubits, (tuple, list)):
        raise ValueError('Gate qubits must be a list/tuple of zero-based indices')
    qubits = tuple(integer(q, 'logical qubit') for q in qubits)
    if len(qubits) != (2 if name == 'CNOT' else 1):
        raise ValueError(f'{name} has the wrong number of operands')
    if len(set(qubits)) != len(qubits):
        raise ValueError('CNOT control and target must be distinct')
    if any(q >= k for q in qubits):
        raise ValueError(f'Logical indices must be in [0, {k})')
    if ancilla in qubits:
        raise ValueError(f'Logical qubit {ancilla} is reserved for the reusable ancilla')
    return GateSpec(name, qubits)


def pauli_at(k, axes):
    text = ['I'] * k
    for q, axis in axes.items():
        text[q] = axis
    return ''.join(text)


def decompose(k, gates, *, ancilla=0):
    k = integer(k, 'k', 2)
    ancilla = integer(ancilla, 'ancilla')
    if ancilla >= k:
        raise ValueError('ancilla must be an original logical index in [0,k)')
    if isinstance(gates, (str, bytes, Mapping)):
        raise ValueError('gates must be a sequence of gate specifications')
    try:
        specs = tuple(_gate(g, k, ancilla) for g in gates)
    except TypeError as error:
        raise ValueError('gates must be a sequence of gate specifications') from error
    if not specs:
        raise ValueError('Supply at least one gate')
    events = []
    for i, spec in enumerate(specs):
        def measure(label, axes, role):
            identifier = f'g{i}_{label}'
            events.append(MeasurementStep(identifier, pauli_at(k, axes), i, role))
            return BitExpr((identifier,))

        def correct(axes, exponent, reason):
            events.append(FrameUpdate(pauli_at(k, axes), exponent, i, reason))

        prep_axis = 'Z' if spec.name == 'H' else 'X'
        p = measure('prep', {ancilla: prep_axis}, 'prepare')
        correct({ancilla: 'X' if prep_axis == 'Z' else 'Z'}, p,
                f'prepare ancilla in +{prep_axis} eigenstate')
        if spec.name == 'H':
            d, = spec.qubits
            b1 = measure('m1', {d: 'X', ancilla: 'X'}, 'joint')
            b2 = measure('m2', {d: 'Z', ancilla: 'X'}, 'joint')
            b3 = measure('m3', {ancilla: 'Z'}, 'readout')
            correct({d: 'X'}, b1 ^ b2, 'H data byproduct X')
            correct({d: 'Z'}, b1 ^ b2 ^ b3 ^ 1, 'H data byproduct Z')
            last = b3
        elif spec.name == 'S':
            d, = spec.qubits
            b1 = measure('m1', {d: 'Z', ancilla: 'Y'}, 'joint')
            b2 = measure('m2', {ancilla: 'Z'}, 'readout')
            correct({d: 'Z'}, b1 ^ b2, 'S data byproduct Z')
            last = b2
        else:
            control, target = spec.qubits
            b1 = measure('m1', {control: 'Z', ancilla: 'Z'}, 'joint')
            b2 = measure('m2', {ancilla: 'X', target: 'X'}, 'joint')
            b3 = measure('m3', {ancilla: 'Z'}, 'readout')
            correct({control: 'Z'}, b2, 'CNOT control byproduct Z')
            correct({target: 'X'}, b1 ^ b3, 'CNOT target byproduct X')
            last = b3
        correct({ancilla: 'X'}, last, 'reset ancilla to +Z in the corrected frame')
    return k, ancilla, specs, tuple(events)


SOURCES = [
    dict(title='Zhang and Li, Time-Efficient Logical Operations, Appendix I.3',
         url='https://arxiv.org/html/2408.01339v3#A9.SS3',
         scope='Pauli measurement computation; their H teleports the data. The same-slot H here is an independently verified projector identity.'),
    dict(title='Horsman et al., Surface code quantum computing by lattice surgery',
         url='https://arxiv.org/abs/1111.4022',
         scope='Logical parity measurements and measurement-mediated CNOT.'),
    dict(title='Krishna and Poulin, Fault-tolerant gates on hypergraph product codes, Lemma 21',
         url='https://arxiv.org/html/1909.07424#S5.SS2',
         scope='One-ancilla measurement framework for Clifford gates; H/S below are verified concrete specializations.'),
    dict(title='Litinski and von Oppen, Lattice Surgery with a Twist, Fig. 3',
         url='https://arxiv.org/html/1709.02318#S3',
         scope='Ancilla |+>, ZZ, XX, Z measurement implementation of CNOT.'),
]
