"""Bounded-degree ancillas for high-rate surgery (arXiv:2510.08523).

``B = boundary`` has auxiliary qubits as rows and new X checks as columns;
``C = cycles`` has auxiliary Z checks as rows.  Thus ``C @ B == 0``.  Memory
checks are dressed according to ``gamma0`` and their anticommutation with the
attachments must equal ``gamma0 @ B``.  The X terminology refers to the
auxiliary qubits: the memory attachments can have arbitrary local Pauli axes.

Appendix E's intralayer, copying and layer-extension moves are implemented as
chain-complex updates.  A candidate is accepted only after the caller's full
merged-code degree check.  An incomplete cycle basis deliberately leaves gauge
degrees of freedom; neither initialization nor successful growth is a distance
certificate.  Random kernel candidates use *column* permutations, since row
permutations alone do not change a canonical nullspace computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np

from ..gf2 import binary_matrix, binary_vector, in_rowspace, nullspace, rank, row_basis


class ConstructionError(ValueError):
    """No valid construction was found within the requested degree budget."""


@dataclass(frozen=True)
class AncillaLayer:
    """Global auxiliary-qubit indices and global new-check coordinates."""

    internal_qubits: tuple[int, ...]
    check_slice: slice


@dataclass(frozen=True)
class AncillaState:
    boundary: np.ndarray
    cycles: np.ndarray
    gamma0: np.ndarray
    attachment_x: np.ndarray
    attachment_z: np.ndarray
    layers: tuple[AncillaLayer, ...]
    bridges: dict[tuple[int, int], tuple[int, ...]]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        b = binary_matrix(self.boundary, name="boundary")
        c = binary_matrix(self.cycles, n_cols=b.shape[0], name="cycles")
        t = binary_matrix(self.gamma0, n_cols=b.shape[0], name="gamma0")
        px = binary_matrix(self.attachment_x, name="attachment_x")
        pz = binary_matrix(self.attachment_z, n_cols=px.shape[1], name="attachment_z")
        if px.shape[0] != b.shape[1] or pz.shape[0] != b.shape[1]:
            raise ValueError("attachments must have one row per boundary column")
        if np.any(c @ b % 2):
            raise ValueError("ancilla Z checks do not commute with the boundary")
        layers = tuple(self.layers)
        bridges = {tuple(k): tuple(v) for k, v in self.bridges.items()}
        if not layers:
            raise ValueError("an ancilla must contain at least one layer")
        check_start = 0
        used_qubits: list[int] = []
        layer_width = None
        for layer in layers:
            span = layer.check_slice
            if (span.step not in (None, 1) or span.start != check_start
                    or span.stop is None or span.stop <= span.start):
                raise ValueError("layer check slices must be consecutive and nonempty")
            width = span.stop - span.start
            if layer_width is not None and width != layer_width:
                raise ValueError("every layer must have the same check-coordinate width")
            layer_width = width
            check_start = span.stop
            for qubit in layer.internal_qubits:
                if not isinstance(qubit, (int, np.integer)) or not 0 <= qubit < len(b):
                    raise ValueError("layer contains an invalid auxiliary-qubit index")
                if np.any(b[qubit, :span.start]) or np.any(b[qubit, span.stop:]):
                    raise ValueError("internal qubits may attach only to their own layer")
                used_qubits.append(int(qubit))
        if check_start != b.shape[1]:
            raise ValueError("layer check slices must cover every boundary column")
        if set(bridges) != {(i, i + 1) for i in range(len(layers) - 1)}:
            raise ValueError("bridges must connect each consecutive pair of layers")
        for (left, right), qubits in bridges.items():
            if len(qubits) != layer_width:
                raise ValueError("each bridge has one qubit per layer check coordinate")
            for coordinate, qubit in enumerate(qubits):
                if not isinstance(qubit, (int, np.integer)) or not 0 <= qubit < len(b):
                    raise ValueError("bridge contains an invalid auxiliary-qubit index")
                expected = np.zeros(b.shape[1], dtype=np.uint8)
                expected[layers[left].check_slice.start + coordinate] = 1
                expected[layers[right].check_slice.start + coordinate] = 1
                if not np.array_equal(b[qubit], expected):
                    raise ValueError("bridge boundary must join matching layer coordinates")
                used_qubits.append(int(qubit))
        if sorted(used_qubits) != list(range(len(b))):
            raise ValueError("layers and bridges must partition the auxiliary qubits")
        # Own all arrays: a new state's construction never aliases mutable input.
        for name, value in (("boundary", b), ("cycles", c), ("gamma0", t),
                            ("attachment_x", px), ("attachment_z", pz)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "bridges", bridges)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_qubits(self) -> int:
        return self.boundary.shape[0]

    @property
    def n_checks(self) -> int:
        return self.boundary.shape[1]


Admissible = Callable[[AncillaState], bool]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class _RowSpace:
    """Incremental echelon basis without recomputing an RREF for every trial."""

    def __init__(self, matrix: np.ndarray):
        self.pivots: dict[int, np.ndarray] = {}
        for row in matrix:
            self.add(row)

    def remainder(self, value: np.ndarray) -> np.ndarray:
        value = value.copy()
        for pivot in sorted(self.pivots):
            if value[pivot]:
                value ^= self.pivots[pivot]
        return value

    def add(self, value: np.ndarray) -> bool:
        reduced = self.remainder(value)
        nonzero = np.flatnonzero(reduced)
        if not nonzero.size:
            return False
        self.pivots[int(nonzero[0])] = reduced
        return True

    def contains(self, value: np.ndarray) -> bool:
        return not np.any(self.remainder(value))

    def __len__(self) -> int:
        return len(self.pivots)


def _kernel_candidates(matrix: np.ndarray, rng: np.random.Generator,
                       trials: int) -> list[np.ndarray]:
    """Several kernel bases and short row combinations, deduplicated by value."""
    width = matrix.shape[1]
    pool: dict[bytes, np.ndarray] = {}

    def add(row: np.ndarray) -> None:
        if np.any(row):
            pool.setdefault(row.tobytes(), row.copy())

    for iteration in range(trials):
        permutation = np.arange(width) if iteration == 0 else rng.permutation(width)
        permuted_basis = nullspace(matrix[:, permutation])
        basis = np.zeros_like(permuted_basis)
        basis[:, permutation] = permuted_basis
        for row in basis:
            add(row)
        if len(basis) < 2:
            continue
        # Pair combinations frequently turn a canonical long cycle into a short one.
        if iteration == 0 and len(basis) <= 24:
            pairs = combinations(range(len(basis)), 2)
        else:
            pairs = (rng.choice(len(basis), size=2, replace=False)
                     for _ in range(min(2 * len(basis), 128)))
        for i, j in pairs:
            add(basis[i] ^ basis[j])
        for _ in range(min(len(basis), 64)):
            size = int(rng.integers(2, min(len(basis), 6) + 1))
            indices = rng.choice(len(basis), size=size, replace=False)
            add(np.bitwise_xor.reduce(basis[indices], axis=0))
    return sorted(pool.values(), key=lambda row: (int(row.sum()), row.tobytes()))


def _within_bounds(state: AncillaState, max_check_weight: int | None,
                   max_qubit_degree: int | None) -> bool:
    """Necessary local bounds; the caller supplies full memory-code bounds."""
    if max_check_weight is not None:
        max_check_weight = _positive_integer(max_check_weight, "max_check_weight")
        x_weights = (state.attachment_x | state.attachment_z).sum(axis=1)
        x_weights = x_weights + state.boundary.sum(axis=0)
        if np.any(x_weights > max_check_weight) or np.any(state.cycles.sum(axis=1) > max_check_weight):
            return False
        if np.any(state.gamma0.sum(axis=1) > max_check_weight):
            return False
    if max_qubit_degree is not None:
        max_qubit_degree = _positive_integer(max_qubit_degree, "max_qubit_degree")
        auxiliary_degrees = (state.boundary.sum(axis=1) + state.cycles.sum(axis=0)
                             + state.gamma0.sum(axis=0))
        if np.any(auxiliary_degrees > max_qubit_degree):
            return False
        if np.any((state.attachment_x | state.attachment_z).sum(axis=0) > max_qubit_degree):
            return False
    return True


def _accepted(state: AncillaState, max_check_weight: int | None,
              max_qubit_degree: int | None, admissible: Admissible | None) -> bool:
    return (_within_bounds(state, max_check_weight, max_qubit_degree)
            and (admissible is None or bool(admissible(state))))


def _require_accepted(state: AncillaState, max_check_weight: int | None,
                      max_qubit_degree: int | None, admissible: Admissible | None) -> AncillaState:
    if not _accepted(state, max_check_weight, max_qubit_degree, admissible):
        raise ConstructionError("ancilla update exceeds the full merged-code row/column budget")
    return state


def _with_cycles(state: AncillaState, cycles: np.ndarray, **metadata: object) -> AncillaState:
    return AncillaState(state.boundary, cycles, state.gamma0,
                        state.attachment_x, state.attachment_z, state.layers,
                        state.bridges, {**state.metadata, **metadata})


def initialize(
    F: object,
    target_coordinates: object,
    attachment_x: object,
    attachment_z: object,
    *,
    max_check_weight: int,
    max_qubit_degree: int,
    seed: int | None = 0,
    trials: int = 32,
    admissible: Admissible | None = None,
    allowed_kernel: object | None = None,
    fallback_to_targets: bool = True,
    add_cycles: bool = True,
) -> AncillaState:
    """Initialize ``F = gamma0 @ boundary`` and eliminate unwanted readout.

    Rows of ``target_coordinates`` are combinations of the memory attachments
    that represent the requested measurements.  ``allowed_kernel`` may include
    additional combinations already known by the caller to be memory stabilizers.
    It must contain the target span and be a subspace of ``ker(F)``.  Separation
    rows are selected from its orthogonal complement until ``ker(B)`` equals
    that space.  If bounded-degree separation fails, a second search can retain
    only the target span; this choice is reported in ``metadata``.

    An admissibility callback must account for original memory check weights
    and memory qubit degrees.  Local bounds alone cannot check those quantities.
    Failure is a bounded search failure, not a nonexistence proof.
    """
    maximum_weight = _positive_integer(max_check_weight, "max_check_weight")
    maximum_degree = _positive_integer(max_qubit_degree, "max_qubit_degree")
    trials = _positive_integer(trials, "trials")
    f = binary_matrix(F, name="F")
    m, s = f.shape
    if s == 0:
        raise ValueError("there must be at least one attachment coordinate")
    targets = row_basis(binary_matrix(target_coordinates, n_cols=s, name="target_coordinates"))
    if len(targets) == 0:
        raise ValueError("target_coordinates must contain a nonzero target")
    if np.any(f @ targets.T % 2):
        raise ValueError("target_coordinates must commute with every source check")
    px = binary_matrix(attachment_x, name="attachment_x")
    pz = binary_matrix(attachment_z, n_cols=px.shape[1], name="attachment_z")
    if px.shape[0] != s or pz.shape[0] != s:
        raise ValueError("attachments must have one row per F column")
    allowed = targets if allowed_kernel is None else row_basis(
        binary_matrix(allowed_kernel, n_cols=s, name="allowed_kernel"))
    if not in_rowspace(targets, allowed) or np.any(f @ allowed.T % 2):
        raise ValueError("allowed_kernel must contain the targets and lie in ker(F)")

    active = np.flatnonzero(np.any(f, axis=1))
    initial_b = f[active].copy()
    initial_t = np.zeros((m, len(active)), dtype=np.uint8)
    initial_t[active, np.arange(len(active))] = 1
    base_metadata = {"seed": seed, "initial_source_rows": active.tolist(),
                     "requested_kernel_dimension": len(allowed),
                     "target_dimension": len(targets),
                     "kernel_fallback_to_targets": False,
                     "construction_method": "bounded_randomized_search",
                     "convergence_guarantee": False,
                     "hgp_fallback_implemented": False}
    base = AncillaState(initial_b, np.zeros((0, len(active)), dtype=np.uint8),
                        initial_t, px, pz,
                        (AncillaLayer(tuple(range(len(active))), slice(0, s)),),
                        {}, base_metadata)
    if not _accepted(base, maximum_weight, maximum_degree, admissible):
        raise ConstructionError(
            "the source-induced initial boundary already exceeds the row/column budget; "
            "increase the budget or change the attachment representatives")
    rng = np.random.default_rng(seed)
    spaces = [allowed]
    if fallback_to_targets and len(allowed) != len(targets):
        spaces.append(targets)
    selected: AncillaState | None = None
    initial_rank = rank(base.boundary)
    for space_index, safe_space in enumerate(spaces):
        required_rank = s - len(safe_space)
        candidates = ([] if initial_rank == required_rank
                      else _kernel_candidates(safe_space, rng, trials))
        candidates = [v for v in candidates if int(v.sum()) <= maximum_degree]
        for attempt in range(trials):
            state = base
            span = _RowSpace(state.boundary)
            if len(span) < required_rank:
                if attempt == 0:
                    order = candidates
                else:
                    # Keep low weight first and vary ties to avoid one greedy packing.
                    order = sorted(candidates, key=lambda v: (int(v.sum()), rng.random()))
                for vector in order:
                    if span.contains(vector):
                        continue
                    b = np.vstack([state.boundary, vector])
                    t = np.pad(state.gamma0, ((0, 0), (0, 1)))
                    candidate = AncillaState(
                        b, np.zeros((0, len(b)), dtype=np.uint8), t, px, pz,
                        (AncillaLayer(tuple(range(len(b))), slice(0, s)),), {},
                        {**base_metadata, "kernel_fallback_to_targets": bool(space_index)})
                    if _accepted(candidate, maximum_weight, maximum_degree, admissible):
                        state = candidate
                        span.add(vector)
                        if len(span) == required_rank:
                            break
            if len(span) == required_rank:
                selected = _with_cycles(
                    state, state.cycles, kernel_dimension=len(safe_space),
                    kernel_fallback_to_targets=bool(space_index),
                    separation_rows=state.n_qubits - len(active),
                    initialization_attempt=attempt + 1)
                break
        if selected is not None:
            break
    if selected is None:
        raise ConstructionError(
            "no bounded-degree separating boundary was found; unwanted logical readout "
            "cannot be accepted. Increase trials or the row/column budget, or use a "
            "different gauging/branching construction")

    cycle_dimension = selected.n_qubits - rank(selected.boundary)
    if add_cycles and cycle_dimension:
        cycle_span = _RowSpace(selected.cycles)
        candidates = _kernel_candidates(selected.boundary.T, rng, trials)
        for cycle in candidates:
            if int(cycle.sum()) > maximum_weight or cycle_span.contains(cycle):
                continue
            candidate = _with_cycles(selected, np.vstack([selected.cycles, cycle]))
            if _accepted(candidate, maximum_weight, maximum_degree, admissible):
                selected = candidate
                cycle_span.add(cycle)
                if len(cycle_span) == cycle_dimension:
                    break
    selected = _with_cycles(
        selected, selected.cycles, cycle_dimension=cycle_dimension,
        cycle_rank=rank(selected.cycles),
        unfilled_cycle_dimension=cycle_dimension - rank(selected.cycles))
    if not np.array_equal(selected.gamma0 @ selected.boundary % 2, f):
        raise AssertionError("internal error: initialization broke the chain map")
    return selected


def _layer(state: AncillaState, index: int) -> AncillaLayer:
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(state.layers):
        raise ValueError("layer_index is out of range")
    return state.layers[int(index)]


def _append_internal(state: AncillaState, layer_index: int, boundary_row: np.ndarray,
                     old_cycle: np.ndarray, move: str) -> AncillaState:
    b = np.vstack([state.boundary, boundary_row])
    c = np.pad(state.cycles, ((0, 0), (0, 1)))
    c = np.vstack([c, np.append(old_cycle, np.uint8(1))])
    t = np.pad(state.gamma0, ((0, 0), (0, 1)))
    layers = list(state.layers)
    old_layer = layers[layer_index]
    layers[layer_index] = AncillaLayer(old_layer.internal_qubits + (state.n_qubits,),
                                       old_layer.check_slice)
    cycle_dimension, cycle_rank = len(b) - rank(b), rank(c)
    return AncillaState(b, c, t, state.attachment_x, state.attachment_z,
                        tuple(layers), state.bridges,
                        {**state.metadata, f"{move}_moves": state.metadata.get(f"{move}_moves", 0) + 1,
                         "cycle_dimension": cycle_dimension,
                         "cycle_rank": cycle_rank,
                         "unfilled_cycle_dimension": cycle_dimension - cycle_rank})


def append_intralayer(
    state: AncillaState, layer_index: int, v: object, *,
    max_check_weight: int | None = None, max_qubit_degree: int | None = None,
    admissible: Admissible | None = None,
) -> AncillaState:
    """Append row ``w=v B_i`` and the Z relation ``(v,1)`` (Appendix E2).

    ``v`` is indexed by that layer's ``internal_qubits``, not by X checks.
    It can alternatively be a global auxiliary vector with zero outside them.
    """
    layer = _layer(state, layer_index)
    vector = binary_vector(v, name="intralayer coefficients")
    indices = list(layer.internal_qubits)
    if len(vector) == state.n_qubits and len(vector) != len(indices):
        outside = vector.copy()
        outside[indices] = 0
        if np.any(outside):
            raise ValueError("intralayer coefficients must vanish outside the selected layer")
        vector = vector[indices]
    if len(vector) != len(indices) or not np.any(vector):
        raise ValueError("intralayer coefficients must be nonzero and indexed by internal qubits")
    row = vector @ state.boundary[indices] % 2
    old_cycle = np.zeros(state.n_qubits, dtype=np.uint8)
    old_cycle[indices] = vector
    candidate = _append_internal(state, int(layer_index), row, old_cycle, "intralayer")
    return _require_accepted(candidate, max_check_weight, max_qubit_degree, admissible)


def append_copy(
    state: AncillaState, from_layer: int, from_qubit: int, to_layer: int, *,
    max_check_weight: int | None = None, max_qubit_degree: int | None = None,
    admissible: Admissible | None = None,
) -> AncillaState:
    """Copy an internal qubit to an adjacent layer using the bridge Z relation.

    ``from_qubit`` is a global auxiliary-qubit index.  If its boundary is ``w``,
    the new Z check is source + bridge(w) + copy, of weight ``2+wt(w)``.
    """
    source = _layer(state, from_layer)
    target = _layer(state, to_layer)
    if abs(from_layer - to_layer) != 1:
        raise ValueError("copying requires adjacent layers")
    if from_qubit not in source.internal_qubits:
        raise ValueError("from_qubit must be an internal qubit of from_layer")
    w = state.boundary[from_qubit, source.check_slice]
    new_row = np.zeros(state.n_checks, dtype=np.uint8)
    new_row[target.check_slice] = w
    old_cycle = np.zeros(state.n_qubits, dtype=np.uint8)
    old_cycle[from_qubit] = 1
    bridge = state.bridges[tuple(sorted((int(from_layer), int(to_layer))))]
    old_cycle[list(bridge)] = w
    candidate = _append_internal(state, int(to_layer), new_row, old_cycle, "copy")
    return _require_accepted(candidate, max_check_weight, max_qubit_degree, admissible)


def append_layer(
    state: AncillaState, *, max_check_weight: int | None = None,
    max_qubit_degree: int | None = None, admissible: Admissible | None = None,
) -> AncillaState:
    """Append matching-coordinate bridges and an empty layer (Appendix E5).

    In qubit-by-check orientation the new block is ``[[B,0],[P_last,I]]``.
    This is the qubit-by-check orientation of the layer-extension map.
    This move alone supplies no new Z checks.
    """
    last = state.layers[-1]
    width = last.check_slice.stop - last.check_slice.start
    b = np.pad(state.boundary, ((0, width), (0, width)))
    qubits = tuple(range(state.n_qubits, state.n_qubits + width))
    b[np.array(qubits), np.arange(last.check_slice.start, last.check_slice.stop)] = 1
    b[np.array(qubits), np.arange(state.n_checks, state.n_checks + width)] = 1
    c = np.pad(state.cycles, ((0, 0), (0, width)))
    t = np.pad(state.gamma0, ((0, 0), (0, width)))
    px = np.pad(state.attachment_x, ((0, width), (0, 0)))
    pz = np.pad(state.attachment_z, ((0, width), (0, 0)))
    layers = state.layers + (AncillaLayer((), slice(state.n_checks, state.n_checks + width)),)
    bridges = {**state.bridges, (len(state.layers) - 1, len(state.layers)): qubits}
    candidate = AncillaState(b, c, t, px, pz, layers, bridges,
                             {**state.metadata, "layer_moves": state.metadata.get("layer_moves", 0) + 1})
    return _require_accepted(candidate, max_check_weight, max_qubit_degree, admissible)


def _complete_layer_kernels(state: AncillaState) -> tuple[bool, ...]:
    """Eq. E7: each complete layer has the initial layer's boundary rowspace."""
    first = state.layers[0]
    reference = state.boundary[list(first.internal_qubits), first.check_slice]
    reference_rank = rank(reference)
    complete = []
    for layer in state.layers:
        local = state.boundary[list(layer.internal_qubits), layer.check_slice]
        complete.append(rank(local) == reference_rank and in_rowspace(local, reference))
    return tuple(complete)


def _last_layer_check_weights(state: AncillaState) -> np.ndarray:
    span = state.layers[-1].check_slice
    return ((state.attachment_x[span] | state.attachment_z[span]).sum(axis=1)
            + state.boundary[:, span].sum(axis=0))


def propose_growth(
    state: AncillaState, low_weight_x_witnesses: object, *,
    max_check_weight: int, max_qubit_degree: int,
    seed: int | None = 0, trials: int = 128,
    admissible: Admissible | None = None, allow_layer: bool = True,
    reserve_layer_capacity: bool = True,
    require_complete_layer_kernel: bool = True,
) -> tuple[AncillaState | None, dict]:
    """Choose a degree-admissible new Z check that detects supplied X witnesses.

    Witness rows contain only current auxiliary X coordinates.  Their new-qubit
    coordinates are initially zero.  The score is the number anticommuting with
    the proposed *Z check*, not with its boundary row.  A new distance search is
    necessary after each update: errors can acquire new auxiliary support.
    When further layers are allowed, reserve one check-weight slot on the last
    layer for its next bridge.  This reservation can be explicitly disabled.

    By default intralayer moves require Eq. E7's complete layer kernel.  If no
    detecting move fits, try a rank-increasing copy to finish an incomplete last
    layer, even when its immediate detection score is zero.  Only then consider
    appending another layer.  This implements necessary structural safeguards,
    not a constructive HGP fallback or a convergence/distance guarantee.
    Return ``(None, record)`` when no move fits.
    """
    maximum_weight = _positive_integer(max_check_weight, "max_check_weight")
    maximum_degree = _positive_integer(max_qubit_degree, "max_qubit_degree")
    trials = _positive_integer(trials, "trials")
    raw = np.asarray(low_weight_x_witnesses)
    if raw.ndim == 1 and raw.size:
        raw = raw[None, :]
    witnesses = binary_matrix(raw, n_cols=state.n_qubits, name="low_weight_x_witnesses")
    if not len(witnesses) or not np.any(witnesses):
        return None, {"move": None, "reason": "no_witness_with_auxiliary_X_support", "score": 0}
    rng = np.random.default_rng(seed)
    reservation_active = bool(reserve_layer_capacity and allow_layer)
    complete = _complete_layer_kernels(state)
    reservation_rejections = 0

    def selection_admissible(candidate: AncillaState) -> bool:
        nonlocal reservation_rejections
        if reservation_active and np.any(_last_layer_check_weights(candidate) > maximum_weight - 1):
            reservation_rejections += 1
            return False
        return admissible is None or bool(admissible(candidate))

    def finish(candidate: AncillaState, record: dict) -> tuple[AncillaState, dict]:
        candidate = _with_cycles(
            candidate, candidate.cycles,
            convergence_guarantee=False, hgp_fallback_implemented=False,
            last_layer_capacity_reserved=reservation_active,
            e7_kernel_complete_intralayer_only=bool(require_complete_layer_kernel),
            kernel_complete_layers=list(_complete_layer_kernels(candidate)))
        return candidate, {**record, "reserved_layer_capacity": reservation_active,
                           "e7_kernel_complete_intralayer_only": bool(require_complete_layer_kernel),
                           "reservation_rejections": reservation_rejections}

    proposals: list[tuple[tuple, str, dict]] = []
    for layer_index, layer in enumerate(state.layers):
        indices = list(layer.internal_qubits)
        if not indices:
            continue
        pool: dict[bytes, np.ndarray] = {}
        if complete[layer_index] or not require_complete_layer_kernel:
            for coordinate in range(len(indices)):
                v = np.zeros(len(indices), dtype=np.uint8)
                v[coordinate] = 1
                pool[v.tobytes()] = v
            maximum_size = min(len(indices), maximum_weight - 1)
            if maximum_size >= 2:
                for _ in range(trials):
                    size = int(rng.integers(2, maximum_size + 1))
                    v = np.zeros(len(indices), dtype=np.uint8)
                    v[rng.choice(len(indices), size=size, replace=False)] = 1
                    pool.setdefault(v.tobytes(), v)
        for v in pool.values():
            score = int(np.count_nonzero(witnesses[:, indices] @ v % 2))
            w = v @ state.boundary[indices] % 2
            if not score or int(v.sum()) + 1 > maximum_weight or int(w.sum()) + 1 > maximum_degree:
                continue
            proposals.append(((-score, int(v.sum()) + 1, int(w.sum()), rng.random()),
                              "intralayer", {"layer_index": layer_index, "v": v, "score": score}))
        for neighbor in (layer_index - 1, layer_index + 1):
            if not 0 <= neighbor < len(state.layers):
                continue
            bridge = state.bridges[tuple(sorted((layer_index, neighbor)))]
            for qubit in indices:
                w = state.boundary[qubit, layer.check_slice]
                if int(w.sum()) + 2 > maximum_weight or int(w.sum()) + 1 > maximum_degree:
                    continue
                parity = (witnesses[:, qubit] + witnesses[:, list(bridge)] @ w) % 2
                score = int(np.count_nonzero(parity))
                if score:
                    proposals.append(((-score, int(w.sum()) + 2, int(w.sum()), rng.random()),
                                      "copy", {"from_layer": layer_index, "from_qubit": qubit,
                                               "to_layer": neighbor, "score": score}))
    checked = 0
    for _, move, details in sorted(proposals, key=lambda proposal: proposal[0]):
        checked += 1
        try:
            if move == "intralayer":
                result = append_intralayer(
                    state, details["layer_index"], details["v"],
                    max_check_weight=maximum_weight, max_qubit_degree=maximum_degree,
                    admissible=selection_admissible)
                source_indices = state.layers[details["layer_index"]].internal_qubits
                record = {"layer_index": details["layer_index"],
                          "source_qubits": [source_indices[i] for i in np.flatnonzero(details["v"])],
                          "score": details["score"]}
            else:
                result = append_copy(
                    state, details["from_layer"], details["from_qubit"], details["to_layer"],
                    max_check_weight=maximum_weight, max_qubit_degree=maximum_degree,
                    admissible=selection_admissible)
                record = details
            return finish(result, {"move": move, **record, "candidates_checked": checked})
        except ConstructionError:
            continue

    # Complete the current last layer before using E7 intralayer moves or
    # creating another layer.  These copies can be necessary even if they do
    # not detect the particular witnesses returned by this distance search.
    if require_complete_layer_kernel and len(state.layers) > 1 and not complete[-1]:
        destination_index = len(state.layers) - 1
        destination = state.layers[-1]
        source = state.layers[-2]
        local_span = _RowSpace(state.boundary[list(destination.internal_qubits), destination.check_slice])
        copies = []
        for qubit in source.internal_qubits:
            w = state.boundary[qubit, source.check_slice]
            if not local_span.contains(w):
                copies.append((int(w.sum()), rng.random(), qubit))
        for _, _, qubit in sorted(copies):
            checked += 1
            try:
                result = append_copy(
                    state, destination_index - 1, qubit, destination_index,
                    max_check_weight=maximum_weight, max_qubit_degree=maximum_degree,
                    admissible=selection_admissible)
                score = int(np.count_nonzero(witnesses @ result.cycles[-1, :-1] % 2))
                return finish(result, {"move": "copy", "from_layer": destination_index - 1,
                                       "from_qubit": qubit, "to_layer": destination_index,
                                       "score": score, "candidates_checked": checked,
                                       "reason": "complete_last_layer_kernel"})
            except ConstructionError:
                continue

    can_extend = complete[-1] if require_complete_layer_kernel else bool(state.layers[-1].internal_qubits)
    if allow_layer and can_extend:
        try:
            result = append_layer(state, max_check_weight=maximum_weight,
                                  max_qubit_degree=maximum_degree, admissible=selection_admissible)
            return finish(result, {"move": "layer", "score": 0, "candidates_checked": checked,
                                   "reason": "no_detecting_check_fits_current_layers"})
        except ConstructionError:
            pass
    reason = "no_degree_admissible_growth_found"
    if require_complete_layer_kernel and not complete[-1]:
        reason = "last_layer_kernel_incomplete_and_no_admissible_completion"
    if reservation_active and np.any(_last_layer_check_weights(state) >= maximum_weight):
        reason = "last_layer_capacity_already_exhausted"
    return None, {"move": None, "score": 0, "candidates_checked": checked,
                  "reason": reason, "reservation_rejections": reservation_rejections,
                  "reserved_layer_capacity": reservation_active,
                  "kernel_complete_layers": list(complete)}


def hgp_ancilla(HB: object, HD: object) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's HGP boundary/cycle blocks (not a distance claim).

    For ``HB[r,s]`` and ``HD[t,u]``, ``B`` has shape ``(s*t+r*u, s*u)``
    and ``C`` has shape ``(r*t, s*t+r*u)``.  In particular ``C B=0`` and
    ``ker(B)=ker(HB) tensor ker(HD)``.
    """
    hb = binary_matrix(HB, name="HB")
    hd = binary_matrix(HD, name="HD")
    r, s = hb.shape
    t, u = hd.shape
    b = np.vstack([np.kron(np.eye(s, dtype=np.uint8), hd),
                   np.kron(hb, np.eye(u, dtype=np.uint8))])
    c = np.hstack([np.kron(hb, np.eye(t, dtype=np.uint8)),
                   np.kron(np.eye(r, dtype=np.uint8), hd)])
    return b.astype(np.uint8), c.astype(np.uint8)


__all__ = ["AncillaLayer", "AncillaState", "ConstructionError", "initialize",
           "append_intralayer", "append_copy", "append_layer", "propose_growth", "hgp_ancilla"]
