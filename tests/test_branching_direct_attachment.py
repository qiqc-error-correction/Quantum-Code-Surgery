"""Direct attachment budgets are measured from actual Pauli check matrices."""
import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.parallel_branching.direct_attachment import direct_loads, preview_direct_attachment
from codesurgery.parallel_branching.construction import _signed_check_rank
from codesurgery.parallel_branching.targets import prepare_targets
from codesurgery.symplectic import commutation_matrix, parse_pauli, pauli_product


def words(items, n=None):
    if not items:
        return np.zeros((0, n), np.uint8), np.zeros((0, n), np.uint8), np.zeros(0, np.uint8)
    n = len(items[0].lstrip('+-'))
    rows = [parse_pauli(item, n) for item in items]
    return tuple(np.asarray([row[i] for row in rows], np.uint8) for i in range(3))


def build(checks, targets, n=None, **kwargs):
    x, z, signs = words(checks, n=n)
    tx, tz, ts = words(targets, n=x.shape[1])
    return preview_direct_attachment(x, z, tx, tz, check_signs=signs, target_signs=ts, **kwargs)


def audit(result, checks):
    x, z, signs = words(checks, n=result['old_n'])
    ox, oz, os = (result[key] for key in ('check_x', 'check_z', 'check_signs'))
    m, n = x.shape
    assert not np.any(commutation_matrix(ox, oz))
    new_rank = _signed_check_rank(ox, oz, os)
    assert result['k'] == ox.shape[1] - new_rank
    assert result['k'] == n - rank(np.hstack((x, z))) - result['target_logical_rank']
    np.testing.assert_array_equal(result['memory_row_growth'],
                                  (ox | oz).sum(axis=1)[:m] - (x | z).sum(axis=1))
    np.testing.assert_array_equal(result['memory_column_growth'],
                                  (ox | oz).sum(axis=0)[:n] - (x | z).sum(axis=0))
    np.testing.assert_array_equal(result['memory_row_growth'], result['target_memory_row_growth'].sum(axis=0))
    np.testing.assert_array_equal(result['memory_column_growth'], result['target_memory_column_growth'].sum(axis=0))
    for i, graph in enumerate(result['graphs']):
        f, b, matching, cycles = (graph[key] for key in ('F', 'B', 'M', 'cycle_basis'))
        np.testing.assert_array_equal((matching @ b) % 2, f)
        assert np.all(f.sum(axis=1) % 2 == 0)
        np.testing.assert_array_equal(matching.sum(axis=1), f.sum(axis=1) // 2)
        assert rank(b) == len(graph['support']) - 1
        assert rank(cycles) == len(b) - rank(b)
        assert not np.any((cycles @ b) % 2)
        xx, zz, phase = pauli_product(ox, oz, os, graph['vertex_rows'])
        np.testing.assert_array_equal(xx[:n], result['target_x'][i])
        np.testing.assert_array_equal(zz[:n], result['target_z'][i])
        assert not np.any(xx[n:] | zz[n:])
        assert phase == 2 * int(result['target_signs'][i] ^ result['target_outcome_signs'][i])
    # Every memory-only readout belongs to the requested span modulo the old
    # stabilizers; no support irreducibility assumption is used.
    coeff = nullspace(np.hstack((ox[:, n:], oz[:, n:])).T)
    memory = (coeff @ np.hstack((ox[:, :n], oz[:, :n]))) % 2
    requested_span = np.vstack((np.hstack((x, z)),
                                np.hstack((result['target_x'], result['target_z']))))
    assert in_rowspace(memory, requested_span)


def test_two_overlapping_targets_have_exact_memory_growth():
    result = build(['XXXX', 'ZZZZ'], ['ZZII', 'ZIZI'])
    audit(result, ['XXXX', 'ZZZZ'])
    assert result['n'] == 6 and result['k'] == 0
    np.testing.assert_array_equal(result['memory_row_growth'], [2, 0])
    np.testing.assert_array_equal(result['memory_column_growth'], [2, 1, 1, 0])
    assert result['distance_certificate'] is None
    assert result['expansion_certificate'] is None


@pytest.mark.parametrize('targets', [['ZZII', 'ZIZI'], ['YYII'], ['ZZII', '-ZZII'], []])
def test_load_helper_matches_full_preview_per_target(targets):
    x, z, _ = words(['XXXX', 'ZZZZ'])
    tx, tz, _ = words(targets, n=4)
    rows, columns = direct_loads(x, z, tx, tz)
    result = build(['XXXX', 'ZZZZ'], targets)
    assert rows.shape == (len(targets), 2)
    assert columns.shape == (len(targets), 4)
    np.testing.assert_array_equal(rows, result['target_memory_row_growth'])
    np.testing.assert_array_equal(columns, result['target_memory_column_growth'])


@pytest.mark.parametrize('checks,targets,pattern', [
    (['X', 'Z'], ['Y'], 'commute'),
    (['XXXX', 'ZZZZ'], ['IIII'], 'trivial'),
    (['XXXX', 'ZZZZ'], ['ZIII'], 'commute'),
    (['XXXXXXXX', 'ZZZZZZZZ'], ['XXXXIIII', 'ZZIIZZII'], 'qubit-wise'),
])
def test_load_helper_validates_structural_requirements(checks, targets, pattern):
    x, z, _ = words(checks)
    tx, tz, _ = words(targets)
    with pytest.raises(ValueError, match=pattern):
        direct_loads(x, z, tx, tz)


def test_weight_four_restriction_paired_without_rejecting_support():
    result = build(['XXXXII'], ['ZZZZII'])
    audit(result, ['XXXXII'])
    graph = result['graphs'][0]
    np.testing.assert_array_equal(graph['edges'], [[0, 1], [2, 3], [0, 2]])
    assert graph['components_before_connecting'] == 2
    np.testing.assert_array_equal(result['memory_row_growth'], [2])
    assert len(graph['connecting_edges']) == 1
    assert graph['cycle_basis'].shape == (0, 3)


def test_cycle_checks_complete_dependent_old_checks():
    checks = ['XXI', 'IXX', 'XIX']
    result = build(checks, ['ZZZ'])
    audit(result, checks)
    graph = result['graphs'][0]
    assert graph['cycle_basis'].shape == (1, 3)
    np.testing.assert_array_equal(graph['cycle_basis'], [[1, 1, 1]])
    # Product of the dressed old checks is exactly the new +X cycle check.
    xx, zz, phase = pauli_product(result['check_x'], result['check_z'],
                                  result['check_signs'], [0, 1, 2])
    np.testing.assert_array_equal(xx, result['check_x'][graph['cycle_rows'][0]])
    np.testing.assert_array_equal(zz, result['check_z'][graph['cycle_rows'][0]])
    assert phase == 0


def test_distinct_check_owners_keep_parallel_edges_and_two_edge_cycles():
    checks = ['XXI', 'XXI']
    result = build(checks, ['ZZI'])
    audit(result, checks)
    graph = result['graphs'][0]
    np.testing.assert_array_equal(graph['edges'], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(graph['edge_owners'], [0, 1])
    np.testing.assert_array_equal(graph['M'], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(graph['cycle_basis'], [[1, 1]])
    assert result['n'] == 5


@pytest.mark.parametrize('target', ['ZZII', 'YYII', 'XXXX'])
def test_graph_incidence_and_ownership_match_existing_paired_graph(target):
    from codesurgery.gauging_measurement.api import paired_graph
    from codesurgery.models import CSSCode
    code = CSSCode([[1, 1, 0, 0], [1, 1, 0, 0]], [])
    px, pz, sign = parse_pauli(target, code.n)
    b, matching, _ = paired_graph(code, (px, pz, sign))
    x, z = np.vstack((code.hx, np.zeros_like(code.hz))), np.vstack((np.zeros_like(code.hx), code.hz))
    result = preview_direct_attachment(x, z, [px], [pz])
    np.testing.assert_array_equal(result['graphs'][0]['B'], b)
    np.testing.assert_array_equal(result['graphs'][0]['M'], matching)


def test_component_chain_matches_default_least_degree_representatives():
    from codesurgery.gauging_measurement.api import paired_graph
    from codesurgery.models import CSSCode
    code = CSSCode([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0]], [])
    px, pz, sign = parse_pauli('ZZZZZZ', 6)
    b, matching, _ = paired_graph(code, (px, pz, sign))
    result = preview_direct_attachment(code.hx, np.zeros_like(code.hx), [px], [pz])
    graph = result['graphs'][0]
    np.testing.assert_array_equal(graph['B'], b)
    np.testing.assert_array_equal(graph['M'], matching)
    np.testing.assert_array_equal(graph['edges'][-2:], [[1, 3], [3, 5]])
    np.testing.assert_array_equal(graph['edge_owners'][-2:], [-1, -1])


@pytest.mark.parametrize('logical', [['YI'], ['YI', 'IY'], ['XI', 'IZ'], ['YI', 'IX', 'YX']])
def test_signed_mixed_axes_use_actual_pauli_support(logical):
    p = prepare_targets(four_qubit_code(), logical_paulis=logical)
    result = preview_direct_attachment(p['source_check_x'], p['source_check_z'],
                                       p['target_x'], p['target_z'],
                                       check_signs=p['source_check_signs'],
                                       target_signs=p['target_signs'])
    audit(result, ['XXXX', 'ZZZZ'])
    assert result['target_logical_rank'] <= 2


def test_opposite_signed_duplicates_get_consistent_outcomes():
    result = build(['XXXX', 'ZZZZ'], ['ZZII', '-ZZII', 'ZZII'])
    audit(result, ['XXXX', 'ZZZZ'])
    np.testing.assert_array_equal(result['target_outcome_signs'], [0, 1, 0])
    assert result['target_logical_rank'] == 1
    np.testing.assert_array_equal(result['memory_row_growth'], [3, 0])
    assert result['outcome_sector'] == 'arbitrary_consistent_preview_assignment'


def test_signed_stabilizer_equivalent_targets_have_forced_relative_outcomes():
    result = build(['XXXX', '-ZZZZ'], ['ZZII', 'IIZZ'])
    audit(result, ['XXXX', '-ZZZZ'])
    np.testing.assert_array_equal(result['target_outcome_signs'], [0, 1])


def test_signed_noncss_input_dependencies_are_validated():
    checks = ['XXII', 'ZZII', '-YYII']
    result = build(checks, ['IIYX', '-IIYI', 'IIIX'])
    audit(result, checks)
    np.testing.assert_array_equal(result['target_outcome_signs'], [0, 0, 1])
    assert result['target_logical_rank'] == 2


def test_single_site_and_empty_targets_are_supported():
    result = build([], ['-Y'], n=1)
    audit(result, [])
    assert result['n'] == 1 and result['k'] == 0
    assert result['graphs'][0]['B'].shape == (0, 1)
    result = build(['XXXX', 'ZZZZ'], [])
    audit(result, ['XXXX', 'ZZZZ'])
    assert result['n'] == 4 and result['k'] == 2 and result['graphs'] == []


@pytest.mark.parametrize('checks,targets,pattern', [
    (['X', 'Z'], ['Y'], 'commute'),
    (['XX', 'ZZ', 'YY'], ['XI'], 'contains -I'),
    (['XXXX', 'ZZZZ'], ['IIII'], 'trivial'),
    (['XXXX', 'ZZZZ'], ['XXXX'], 'trivial'),
    (['XXXX', 'ZZZZ'], ['ZIII'], 'commute'),
    (['XXXXXXXX', 'ZZZZZZZZ'], ['XXXXIIII', 'ZZIIZZII'], 'qubit-wise'),
])
def test_invalid_inputs_rejected(checks, targets, pattern):
    with pytest.raises(ValueError, match=pattern):
        build(checks, targets)


@pytest.mark.parametrize('budget', [True, 0, -1, 1.2, None])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match='max_entries'):
        build(['XXXX', 'ZZZZ'], ['ZZII'], max_entries=budget)


def test_budget_prevents_output_matrix_allocation(monkeypatch):
    import codesurgery.parallel_branching.direct_attachment as module
    original = module.np.zeros
    allocations = []
    def guarded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2:
            allocations.append(shape)
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(module.np, 'zeros', guarded)
    with pytest.raises(ValueError, match='dense entries'):
        build(['XXXX', 'ZZZZ'], ['ZZII'], max_entries=1)
    assert allocations == []


def test_inputs_are_not_mutated_and_nonbinary_data_rejected():
    x, z, s = words(['XXXX', 'ZZZZ'])
    tx, tz, ts = words(['ZZII'])
    saved = [v.copy() for v in (x, z, s, tx, tz, ts)]
    preview_direct_attachment(x, z, tx, tz, check_signs=s, target_signs=ts)
    for value, old in zip((x, z, s, tx, tz, ts), saved):
        np.testing.assert_array_equal(value, old)
    tx[0, 0] = 2
    with pytest.raises(ValueError, match='0 and 1'):
        preview_direct_attachment(x, z, tx, tz)
