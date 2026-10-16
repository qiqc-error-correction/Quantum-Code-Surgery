"""Independent high-rate shared-ancilla surgery (Zheng--Jiang--Xu, 2025).

The randomized route implements the finite-size layered construction of
Appendix E. It does not promise constant overhead for arbitrary inputs, and
distance search exhaustion is never treated as a proof of distance.
"""
from __future__ import annotations
from copy import deepcopy
from numbers import Real
from time import perf_counter
import numpy as np

from ..models import CSSCode
from ..gf2 import binary_matrix
from .targets import prepare_targets
from .validation import (merged_checks, weight_report, inspect_readout,
                         consistent_check_signs, readout_witnesses, check_chain_maps)
from .models import HighRateCode, physical_checks
from .ancilla import initialize, propose_growth
from .distance import build_css_protected_basis, css_dressed_distance, symplectic_gauge_free_distance


class DistanceCertificationError(ValueError):
    """A correct sparse candidate exists, but its requested distance is unproved."""
    def __init__(self, result):
        self.result = result
        super().__init__('Requested protected distance was not certified. '
                         'The commuting, readout-validated candidate is available as error.result; '
                         f"status={result.metadata.get('status')}")


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _result(code, p, state, row_limit, col_limit, metadata):
    ancilla_h = not np.any(p['target_x'])  # Pure-Z dual: all ancilla axes swapped offline.
    check_chain_maps(p['source_x'], p['source_z'], state)
    x, z = physical_checks(p, state, ancilla_h)
    readout = inspect_readout(code, p['basis'], x, z, p['target_signatures'])
    signs = consistent_check_signs(code, p['source_x'], p['source_z'], x, z,
                                   readout, state.boundary.shape[1])
    witness, corrections = readout_witnesses(x, z, signs, p['target_x'], p['target_z'], p['target_signs'])
    result = HighRateCode(code, state, p, x, z, signs, witness, corrections,
                         row_limit, col_limit, ancilla_h, dict(metadata))
    result.metadata['measurement_validation'] = readout
    if not np.any(p['target_z']):
        internal = build_css_protected_basis(code.hx, code.hz, p['basis']['jx'], p['basis']['jz'],
            result.hx, result.hz, state.boundary, state.attachment_x.T,
            p['target_signatures'][:, :code.k])
        result.subsystem = internal
        distance_args = (result.hx, result.hz, internal['bare_x'], internal['bare_z'])
        result.metadata['growth_axis'] = 'X'
    elif not np.any(p['target_x']):
        internal = build_css_protected_basis(code.hz, code.hx, p['basis']['jz'], p['basis']['jx'],
            result.hz, result.hx, state.boundary, state.attachment_z.T,
            p['target_signatures'][:, code.k:])
        result.subsystem = dict(internal, gauge_x=internal['gauge_z'], gauge_z=internal['gauge_x'],
            bare_x=internal['bare_z'], bare_z=internal['bare_x'], convention='zero_original_logical_X_signature')
        distance_args = (result.hz, result.hx, internal['bare_x'], internal['bare_z'])
        result.metadata['growth_axis'] = 'Z'
    else:
        # With no extra gauge qubits the existing full-symplectic stabilizer
        # distance is EXACTLY the protected static distance. Do not use this
        # shortcut when any ancilla gauge degree of freedom remains.
        distance_args = 'mixed' if result.gauge_qubits == 0 else None
        result.metadata['growth_axis'] = 'mixed'
    return result, distance_args


def _public_distance(report, result):
    report = deepcopy(report)
    if result.metadata['growth_axis'] == 'Z':
        if 'x' in report and 'z' in report:
            report['x'], report['z'] = report['z'], report['x']
        if 'low_weight_x_witnesses' in report:
            report['low_weight_x_witnesses'], report['low_weight_z_witnesses'] = (
                report['low_weight_z_witnesses'], report['low_weight_x_witnesses'])
        if report.get('witness_type') in ('X','Z'):
            report['witness_type'] = 'Z' if report['witness_type'] == 'X' else 'X'
    report['axis_convention'] = 'physical output X and Z'
    report['measurement_distance_certified'] = False
    return report


def _merge_bounds(old, new):
    """Combine runs on the SAME candidate; never carry bounds across growth."""
    if 'x' not in old or 'x' not in new:
        merged = deepcopy(new)
        if old.get('proven_lower_bound') is None or new.get('proven_lower_bound') is None:
            return merged
        if old['proven_lower_bound'] > new['proven_lower_bound']:
            merged['lower_bound_source_run'] = deepcopy(old)
        merged['proven_lower_bound'] = merged['lower_bound'] = max(old['proven_lower_bound'],new['proven_lower_bound'])
        if old.get('upper_bound') is not None and (new.get('upper_bound') is None or old['upper_bound']<new['upper_bound']):
            for key in ['upper_bound','witness','witness_type','witness_validated']:
                if key in old: merged[key]=old[key]
        merged['distance'] = merged['upper_bound'] if merged.get('upper_bound')==merged['proven_lower_bound'] else None
        merged['status'] = 'exact' if merged['distance'] is not None else 'bounded'
        merged['combined_same_candidate_runs'] = True
        return merged
    merged = deepcopy(new)
    for axis in ['x', 'z']:
        a, b = old[axis], merged[axis]
        if a['proven_lower_bound'] > b['proven_lower_bound']:
            b['lower_bound_certificate'] = a.get('lower_bound_certificate')
            b['lower_bound_source_run'] = deepcopy(a)
        b['proven_lower_bound'] = b['lower_bound'] = max(a['proven_lower_bound'], b['proven_lower_bound'])
        if a.get('upper_bound') is not None and (b.get('upper_bound') is None or a['upper_bound'] < b['upper_bound']):
            for key in ['upper_bound', 'witness', 'witness_validated']:
                b[key] = a[key]
        b['distance'] = b['upper_bound'] if b['upper_bound'] == b['lower_bound'] else None
        b['status'] = 'exact' if b['distance'] is not None else 'bounded'
    merged['lower_bound'] = merged['proven_lower_bound'] = min(merged['x']['lower_bound'], merged['z']['lower_bound'])
    upper = [merged[q]['upper_bound'] for q in ['x', 'z'] if merged[q]['upper_bound'] is not None]
    merged['upper_bound'] = min(upper) if upper else None
    merged['distance'] = merged['upper_bound'] if merged['upper_bound'] == merged['lower_bound'] else None
    merged['status'] = 'exact' if merged['distance'] is not None else 'bounded'
    selected = min((q for q in ['x','z'] if merged[q]['upper_bound'] is not None),
                   key=lambda q: merged[q]['upper_bound'], default=None)
    if selected is not None:
        merged['witness'] = merged[selected]['witness']
        merged['witness_type'] = selected.upper()
        merged['witness_validated'] = merged[selected]['witness_validated']
    merged['combined_same_candidate_runs'] = True
    return merged


def _distance(result, args, *, method, time_limit, max_weight, seed, trials):
    if not isinstance(args, str):
        return css_dressed_distance(*args,method=method,time_limit=time_limit,
                                   max_weight=max_weight,seed=seed,trials=trials)
    if result.protected_k == 0:
        return dict(status='no_protected_logicals',proven_lower_bound=None,upper_bound=None,
                    low_weight_x_witnesses=[],low_weight_z_witnesses=[])
    return symplectic_gauge_free_distance(result.check_x,result.check_z,method=method,
        time_limit=time_limit,max_weight=max_weight,max_candidates=100_000,
        trials=trials,seed=seed)


def high_rate_deform(code, targets=None, *, pauli='Z', logical_indices=None,
                     logical_basis=None, logical_paulis=None, optimize_support=True,
                     ensure_distance=True, target_distance=None, distance_time_limit=60,
                     distance_method='auto', max_check_weight=None, max_qubit_degree=None,
                     seed=0, max_steps=100, max_layers=20, strict=False,
                     trials=64, max_entries=50_000_000):
    """Measure a commuting target space with one shared, layered ancilla.

    ``targets`` accepts physical signed Pauli strings/dictionaries; alternatively
    ``logical_paulis`` gives strings of length k in the supplied or standard
    basis. Each list entry is one requested observable: ['XX'] is a single
    product, ['XI','IX'] requests two separate outcomes. The final logical span
    is checked exactly, including at t=1 and after every accepted growth step.

    Complete Pauli row/column weights have independent limits, by default the
    source maxima plus two. Construction cannot exceed these limits. Pure X/Z
    targets receive a protected subsystem and rigorous bounded-distance search;
    QWC mixed targets use a full-symplectic distance check when gauge-free;
    mixed candidates with extra gauge qubits have no distance claim.
    ``distance_time_limit`` is one shared wall-clock budget for distance and
    growth, after initialization (default 60 seconds). ``strict`` raises with
    ``error.result`` when the requested protected distance was not certified.
    """
    if not isinstance(code, CSSCode):
        raise TypeError('code must be a CSSCode')
    for value, name in [(optimize_support,'optimize_support'), (ensure_distance,'ensure_distance'), (strict,'strict')]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean')
    seed, max_steps = _integer(seed,'seed'), _integer(max_steps,'max_steps')
    max_layers, trials = _integer(max_layers,'max_layers',1), _integer(trials,'trials',1)
    max_entries = _integer(max_entries,'max_entries',1)
    if (isinstance(distance_time_limit,(bool,np.bool_)) or not isinstance(distance_time_limit,Real)
            or not np.isfinite(distance_time_limit) or distance_time_limit <= 0):
        raise ValueError('distance_time_limit must be positive and finite')
    if distance_method not in ('auto','gurobi','enum','heuristic'):
        raise ValueError('distance_method must be auto, gurobi, enum, or heuristic')
    if target_distance is None:
        target_distance = code.distance
    if target_distance is not None:
        target_distance = _integer(target_distance,'target_distance',1)
    p = prepare_targets(code, targets, logical_paulis=logical_paulis, pauli=pauli,
        logical_indices=logical_indices, logical_basis=logical_basis,
        optimize_support=optimize_support, seed=seed, optimization_trials=trials)
    original_weights = weight_report(p['source_x'], p['source_z'])
    row_limit = original_weights['max_check_weight'] + 2 if max_check_weight is None else _integer(max_check_weight,'max_check_weight',1)
    col_limit = original_weights['max_qubit_degree'] + 2 if max_qubit_degree is None else _integer(max_qubit_degree,'max_qubit_degree',1)
    if row_limit < original_weights['max_check_weight'] or col_limit < original_weights['max_qubit_degree']:
        raise ValueError('Weight limits are already smaller than the original code')

    def admissible(state):
        count = len(p['source_x']) + state.boundary.shape[1] + len(state.cycles)
        width = code.n + len(state.boundary)
        if 2*count*width + count*count > max_entries:
            return False
        x,z = merged_checks(p['source_x'],p['source_z'],state)
        w = weight_report(x,z)
        return w['max_check_weight'] <= row_limit and w['max_qubit_degree'] <= col_limit

    state = initialize(p['restriction'], p['coordinates'], p['attachment_x'], p['attachment_z'],
        max_check_weight=row_limit, max_qubit_degree=col_limit, seed=seed,
        trials=trials, admissible=admissible, allowed_kernel=p['allowed_kernel'])
    metadata = dict(algorithm='Appendix E layered shared-ancilla randomized construction',
                    seed=seed, target_distance=target_distance, source_distance=code.distance,
                    source_distance_is_input_assumption=True, optimization=p['optimization'],
                    degree_policy='separate complete-matrix row and column maxima, source + 2 by default',
                    source_max_check_weight=original_weights['max_check_weight'],
                    source_max_qubit_degree=original_weights['max_qubit_degree'],
                    steps=[], protected_distance_certified=False,
                    measurement_fault_distance_certified=False,
                    distance_time_limit=float(distance_time_limit), max_steps=max_steps, max_layers=max_layers)
    result,args = _result(code,p,state,row_limit,col_limit,metadata)
    if not ensure_distance:
        report = dict(status='disabled', proven_lower_bound=None, upper_bound=None)
        status = 'constructed_distance_disabled'
    elif args is None:
        report = dict(status='unsupported_mixed_subsystem', proven_lower_bound=None, upper_bound=None,
            note='QWC mixed construction/readout is verified, but extra gauge qubits remain; a general symplectic protected-subsystem distance backend is not implemented.')
        status = 'constructed_mixed_distance_unverified'
    else:
        started = perf_counter()
        deadline = started + float(distance_time_limit)
        history = metadata['steps']
        report = None
        for step in range(max_steps+1):
            remaining = deadline-perf_counter()
            if remaining <= 0:
                status = 'distance_time_limit'
                break
            budget = remaining if target_distance is None else min(5.0,remaining)
            report = _distance(result,args, method=distance_method, time_limit=max(1e-6,budget),
                max_weight=None if target_distance is None else target_distance-1,
                seed=seed+step, trials=trials)
            if report['status'] == 'no_protected_logicals':
                status = 'constructed_all_logicals_measured_readout_distance_unverified'
                break
            if target_distance is None:
                status = 'constructed_no_distance_target'
                break
            if report['proven_lower_bound'] >= target_distance:
                status = 'protected_distance_certified'
                break
            witnesses = [w for w in report.get('low_weight_x_witnesses',[]) if sum(w)<target_distance]
            if step == max_steps or not witnesses or deadline-perf_counter() <= 0:
                status = 'step_limit' if step == max_steps else 'distance_inconclusive'
                break
            auxiliary = np.asarray(witnesses,np.uint8)[:, code.n:]
            state_new, record = propose_growth(state, auxiliary,
                max_check_weight=row_limit,max_qubit_degree=col_limit,
                seed=seed+step,trials=trials,admissible=admissible,
                allow_layer=len(state.layers)<max_layers)
            if state_new is None:
                status = 'no_admissible_growth'
                break
            history.append(dict(step=step, previous_distance_lower=report['proven_lower_bound'],
                                previous_distance_upper=report['upper_bound'], **record))
            previous_gauge_count=result.gauge_qubits
            state = state_new
            result,args = _result(code,p,state,row_limit,col_limit,metadata)
            if result.gauge_qubits != previous_gauge_count:
                raise RuntimeError('A layered growth move changed the gauge count')
            report = None  # Bounds belong to the previous matrix, not this one.
        remaining = deadline-perf_counter()
        if report is None:
            report = dict(status='unknown',proven_lower_bound=1,upper_bound=None,
                          stop_reason='deadline_after_growth')
        if (target_distance is not None and report.get('proven_lower_bound') is not None
                and report['proven_lower_bound'] < target_distance and remaining > 0.01):
            refined = _distance(result,args, method=distance_method,time_limit=remaining,
                max_weight=target_distance-1,seed=seed+max_steps+1,trials=trials)
            report = _merge_bounds(report,refined)
            if report['proven_lower_bound'] is not None and report['proven_lower_bound'] >= target_distance:
                status = 'protected_distance_certified'
        metadata['distance_and_growth_seconds'] = perf_counter()-started
    result.metadata.update(metadata, status=status,
        distance=_public_distance(report,result),
        protected_distance_certified=status=='protected_distance_certified')
    if result.subsystem is not None:
        result.metadata['gauge_partition_convention'] = result.subsystem['convention']
    result.validate()
    if strict and ensure_distance and not result.metadata['protected_distance_certified']:
        raise DistanceCertificationError(result)
    return result


def high_rate_from_ancilla(code, boundary, cycles, gamma0, gamma1, targets=None, *,
                           pauli='X', logical_paulis=None, logical_indices=None,
                           logical_basis=None, ensure_distance=True,
                           target_distance=None, distance_method='auto',
                           distance_time_limit=60, max_check_weight=None,
                           max_qubit_degree=None, strict=False, seed=0):
    """Attach a supplied algebraic ancilla and chain maps for pure X or Z.

    This is the matrix entry point for HGP ancillas from ``hgp_ancilla``.
    Gamma1 has shape (data qubits, ancilla attachment checks), and Gamma0
    has shape (opposite-type original checks, ancilla qubits). The caller
    supplies the maps: arbitrary HGP factors do not determine a valid
    addressable coupling to an arbitrary memory code. Invalid maps, extra
    logical readout and overweight constructions raise ValueError.
    """
    from .ancilla import AncillaLayer, AncillaState
    if not isinstance(code,CSSCode):
        raise TypeError('code must be a CSSCode')
    if not isinstance(pauli,str) or pauli.upper() not in ('X','Z'):
        raise ValueError('The algebraic chain-map entry point requires pure X or Z')
    pauli=pauli.upper()
    for value,name in [(ensure_distance,'ensure_distance'),(strict,'strict')]:
        if not isinstance(value,(bool,np.bool_)): raise ValueError(f'{name} must be boolean')
    if not isinstance(distance_time_limit,Real) or isinstance(distance_time_limit,bool) or not np.isfinite(distance_time_limit) or distance_time_limit<=0:
        raise ValueError('distance_time_limit must be positive and finite')
    seed=_integer(seed,'seed')
    if distance_method not in ('auto','gurobi','enum','heuristic'):
        raise ValueError('Invalid distance_method')
    d=code.distance if target_distance is None else _integer(target_distance,'target_distance',1)
    p=prepare_targets(code,targets,logical_paulis=logical_paulis,pauli=pauli,
                      logical_indices=logical_indices,logical_basis=logical_basis,optimize_support=False)
    if np.any(p['target_z'] if pauli=='X' else p['target_x']):
        raise ValueError('Requested targets have the wrong Pauli type for the supplied maps')
    b=binary_matrix(boundary,name='boundary')
    c=binary_matrix(cycles,n_cols=len(b),name='cycles')
    r=binary_matrix(gamma1,n_cols=b.shape[1],name='gamma1')
    if len(r)!=code.n: raise ValueError('gamma1 must have one row per data qubit')
    t=binary_matrix(gamma0,n_cols=len(b),name='gamma0')
    conjugate_count=len(code.hz) if pauli=='X' else len(code.hx)
    if len(t)!=conjugate_count: raise ValueError('gamma0 must have one row per opposite-type original check')
    full_t=np.zeros((len(code.hx)+len(code.hz),len(b)),np.uint8)
    full_t[len(code.hx):] = t if pauli=='X' else full_t[len(code.hx):]
    if pauli=='Z': full_t[:len(code.hx)]=t
    px=r.T if pauli=='X' else np.zeros_like(r.T)
    pz=r.T if pauli=='Z' else np.zeros_like(r.T)
    state=AncillaState(b,c,full_t,px,pz,(AncillaLayer(tuple(range(len(b))),slice(0,b.shape[1])),),{},
                       dict(construction='supplied_algebraic_ancilla'))
    original=weight_report(p['source_x'],p['source_z'])
    row=original['max_check_weight']+2 if max_check_weight is None else _integer(max_check_weight,'max_check_weight',1)
    col=original['max_qubit_degree']+2 if max_qubit_degree is None else _integer(max_qubit_degree,'max_qubit_degree',1)
    metadata=dict(algorithm='algebraic ancilla with supplied chain maps',seed=seed,target_distance=d,steps=[],
                  protected_distance_certified=False,measurement_fault_distance_certified=False)
    result,args=_result(code,p,state,row,col,metadata)
    result.validate()
    if ensure_distance:
        report=_distance(result,args,method=distance_method,time_limit=distance_time_limit,
                         max_weight=None if d is None else d-1,seed=seed,trials=64)
        certified=(d is not None and result.protected_k>0 and report.get('proven_lower_bound') is not None
                   and report['proven_lower_bound']>=d)
        result.metadata.update(distance=_public_distance(report,result),protected_distance_certified=certified,
            status='protected_distance_certified' if certified else 'constructed_algebraic_distance_unverified')
    else:
        result.metadata.update(distance=dict(status='disabled',proven_lower_bound=None,upper_bound=None),
                               status='constructed_distance_disabled')
    if strict and ensure_distance and not result.metadata['protected_distance_certified']:
        raise DistanceCertificationError(result)
    return result
