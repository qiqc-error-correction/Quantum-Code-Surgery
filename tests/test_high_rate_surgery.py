"""End-to-end invariants of the independent shared-ancilla construction."""
import json
from dataclasses import replace
import numpy as np
import pytest
from codesurgery import CSSCode
from codesurgery.examples import four_qubit_code, steane_code, direct_sum, ibm_bb90_code
from codesurgery.gf2 import rank
from codesurgery.high_rate_surgery import high_rate_deform, DistanceCertificationError
from codesurgery.symplectic import commutation_matrix, pauli_product


@pytest.mark.parametrize('target',['XI','ZI','XX','ZZ','YI','XZ','YY','-YI'])
def test_joint_observable_is_exactly_one_readout(target):
    r=high_rate_deform(four_qubit_code(),logical_paulis=[target],ensure_distance=False)
    assert r.validate()
    assert r.prepared['target_rank']==1
    assert r.metadata['measurement_validation']['logical_rank']==1
    assert r.protected_k==1
    assert not np.any(commutation_matrix(r.check_x,r.check_z))
    assert r.summary()['max_check_weight']<=6
    assert r.summary()['max_qubit_degree']<=4


def test_separate_vs_product_and_signed_redundant_inputs():
    code=four_qubit_code()
    a=high_rate_deform(code,logical_paulis=['ZI','IZ'],ensure_distance=False)
    b=high_rate_deform(code,logical_paulis=['ZZ'],ensure_distance=False)
    assert (a.protected_k,b.protected_k)==(0,1)
    repeated=high_rate_deform(code,logical_paulis=['ZI','-ZI'],ensure_distance=False)
    assert repeated.prepared['target_rank']==1
    assert repeated.readout_sign_corrections[0]^repeated.readout_sign_corrections[1]==1


def test_single_reducible_input_support_extras_are_removed():
    # In this high-rate method low-weight logical checks remove extra readout
    # directions; it must not simply measure all logicals inside the support.
    code=CSSCode([[1]*6],[[1]*6],distance=2)
    r=high_rate_deform(code,targets=['XXXXII'],optimize_support=False,ensure_distance=False,max_qubit_degree=6)
    assert len(r.state.boundary)>1
    assert r.metadata['measurement_validation']['logical_rank']==1


@pytest.mark.parametrize('targets',[['XI','ZI'],['XX','ZZ']])
def test_incompatible_measurement_sets_are_rejected(targets):
    with pytest.raises(ValueError,match='commut|QWC'):
        high_rate_deform(four_qubit_code(),logical_paulis=targets,optimize_support=False)


def test_css_x_z_distance_and_no_protected_convention():
    for target in ['XI','ZI']:
        r=high_rate_deform(four_qubit_code(),logical_paulis=[target],distance_method='enum',strict=True)
        assert r.metadata['distance']['distance']==2
        assert r.metadata['protected_distance_certified']
        assert not r.summary()['measurement_fault_distance_certified']
    r=high_rate_deform(four_qubit_code(),logical_paulis=['ZI','IZ'],distance_method='enum')
    assert r.metadata['distance']['status']=='no_protected_logicals'
    assert not r.metadata['protected_distance_certified']


def test_small_distance_three_source_and_explicit_basis():
    code=direct_sum(steane_code(),steane_code())
    b=code.standard_logicals()
    r=high_rate_deform(code,logical_paulis=['ZI'],logical_basis=dict(jx=b.jx,jz=b.jz),
                      distance_method='enum',distance_time_limit=2)
    assert r.metadata['distance']['distance']==3
    assert r.prepared['basis']['origin']=='user supplied paired CSS basis'


def test_mixed_gauge_free_distance_is_full_symplectic():
    r=high_rate_deform(four_qubit_code(),logical_paulis=['YI'],distance_method='enum',strict=True)
    assert not r.is_css
    assert r.metadata['distance']['scope']=='protected_mixed_stabilizer_gauge_free'
    assert r.metadata['distance']['distance']==2
    with pytest.raises(DistanceCertificationError) as exc:
        high_rate_deform(four_qubit_code(),logical_paulis=['YI'],strict=True,target_distance=3,
                         max_steps=0,distance_method='enum',distance_time_limit=.1)
    assert exc.value.result.validate()


def test_save_json_npz_and_validate_tampering(tmp_path):
    r=high_rate_deform(four_qubit_code(),logical_paulis=['YI'],ensure_distance=False)
    r.save(tmp_path/'result.json'); r.save(tmp_path/'result.npz')
    data=json.loads((tmp_path/'result.json').read_text())
    assert data['summary']['scheme']=='high_rate_surgery'
    with np.load(tmp_path/'result.npz',allow_pickle=False) as data:
        assert np.array_equal(data['boundary'],r.state.boundary)
        assert json.loads(str(data['information']))['summary']['logical_measurement_rank']==1
    r.readout_sign_corrections[0]^=1
    with pytest.raises(ValueError,match='witness'):
        r.validate()


def test_hard_bounds_never_silently_relaxed():
    with pytest.raises(ValueError):
        high_rate_deform(steane_code(),logical_paulis=['Y'],ensure_distance=False)
    r=high_rate_deform(steane_code(),logical_paulis=['Y'],max_check_weight=8,ensure_distance=False)
    assert r.validate()


@pytest.mark.parametrize('targets',[['ZIIIIIII'],['ZIIIIIII','IZIIIIII'],['ZZIIIIII'],['XZIIIIII'],['YIIIIIII']])
def test_bb90_matrix_and_complete_readout(targets):
    r=high_rate_deform(ibm_bb90_code(),logical_paulis=targets,ensure_distance=False,trials=16)
    assert r.validate()
    assert r.prepared['target_rank']==len(targets)
    assert r.summary()['max_check_weight']<=8
    assert r.summary()['max_qubit_degree']<=8


@pytest.mark.parametrize('pauli',['X','Z'])
def test_algebraic_ancilla_attachment_round_trip(pauli):
    from codesurgery.high_rate_surgery import high_rate_from_ancilla
    code=four_qubit_code()
    r=high_rate_deform(code,logical_paulis=[pauli+'I'],optimize_support=False,ensure_distance=False)
    t=r.state.gamma0[len(code.hx):] if pauli=='X' else r.state.gamma0[:len(code.hx)]
    g=r.state.attachment_x.T if pauli=='X' else r.state.attachment_z.T
    a=high_rate_from_ancilla(code,r.state.boundary,r.state.cycles,t,g,
                            logical_paulis=[pauli+'I'],pauli=pauli,distance_method='enum',strict=True)
    assert a.validate()
    assert a.metadata['distance']['distance']==2
    assert np.array_equal(a.check_x,r.check_x)
    assert np.array_equal(a.check_z,r.check_z)
    with pytest.raises(ValueError,match='Chain map'):
        high_rate_from_ancilla(code,r.state.boundary,r.state.cycles,np.zeros_like(t),g,
                              logical_paulis=[pauli+'I'],pauli=pauli,ensure_distance=False)
