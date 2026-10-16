"""Signed logical targets and exact readout through surgery dispatch."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from codesurgery import CSSCode
from codesurgery.examples import four_qubit_code
from codesurgery.gadget_common import memory_checks, prepare, signature
from codesurgery.gf2 import nullspace, row_basis
from codesurgery.pauli_gates.backends import (
    BackendMeasurementError, SUPPORTED_BACKENDS, build_measurement, describe_backend,
)
from codesurgery.symplectic import commutation_matrix, parse_pauli, pauli_product


OPTIONS = {
    "gauging": {"require_distance": False},
    "explicit": {"require_distance": False},
    "ckbb": {"layers": 2},
    "devised_sticking": {"d_R": 2},
    "high_rate": {"ensure_distance": False},
}


def independently_assert_readout(code, result, text, logical_basis=None):
    a, b, sign = parse_pauli(text, code.k)
    _, target, basis = prepare(code, dict(logical_x=a, logical_z=b, sign=sign),
                                logical_basis=logical_basis)
    assert not np.any(commutation_matrix(result.check_x, result.check_z))
    cancel = nullspace(np.hstack((result.check_x[:, code.n:],
                                  result.check_z[:, code.n:])).T)
    rx = cancel @ result.check_x[:, :code.n] % 2
    rz = cancel @ result.check_z[:, :code.n] % 2
    actual = row_basis(signature(rx, rz, basis))
    assert np.array_equal(actual, signature(target[0], target[1], basis)[None, :])
    assert result.protected_k == code.k - 1
    assert result.validate()


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
@pytest.mark.parametrize("target", ["XI", "ZI", "XX", "ZZ", "-XI", "-ZZ"])
def test_all_backends_preserve_one_complete_pure_target(backend, target):
    code = four_qubit_code()
    result = build_measurement(code, target, backend=backend, backend_options=OPTIONS[backend])
    independently_assert_readout(code, result, target)


@pytest.mark.parametrize("backend", ["gauging", "explicit", "ckbb", "high_rate"])
@pytest.mark.parametrize("target", ["YI", "XZ", "YY", "-YI", "-XZ"])
def test_mixed_target_is_one_joint_measurement(backend, target):
    code = four_qubit_code()
    if backend == "explicit" and target.lstrip("-") == "XZ":
        # The raw restriction has two independent logical directions. This
        # route must reject instead of measuring X and Z separately; an
        # adapter must be requested through a separate explicit construction.
        with pytest.raises(BackendMeasurementError, match="unintended"):
            build_measurement(code, target, backend=backend, backend_options=OPTIONS[backend])
        return
    result = build_measurement(code, target, backend=backend, backend_options=OPTIONS[backend])
    independently_assert_readout(code, result, target)


def test_logical_y_three_physical_overlaps_has_negative_physical_sign():
    # i XXX ZZZ = -YYY in a different representative of the same CSS logical.
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]], distance=1)
    basis = dict(jx=[[1, 1, 1]], jz=[[1, 1, 1]])
    result = build_measurement(code, "Y", logical_basis=basis,
                               backend_options={"require_distance": False})
    assert np.array_equal(result.measurement_x, [1, 1, 1])
    assert np.array_equal(result.measurement_z, [1, 1, 1])
    assert result.measurement_sign == 1
    independently_assert_readout(code, result, "Y", basis)
    negative = build_measurement(code, "-Y", logical_basis=basis,
                                 backend_options={"require_distance": False})
    assert negative.measurement_sign == 0


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_custom_permuted_logical_basis_applies_to_every_backend(backend):
    code = four_qubit_code()
    original = code.standard_logicals()
    basis = dict(jx=original.jx[::-1].copy(), jz=original.jz[::-1].copy())
    result = build_measurement(code, "XI", logical_basis=basis,
                               backend=backend, backend_options=OPTIONS[backend])
    independently_assert_readout(code, result, "XI", basis)


@pytest.mark.parametrize("target", ["YI", "XZ"])
def test_devised_rejects_mixed_without_fallback(target):
    with pytest.raises(BackendMeasurementError, match="pure X or pure Z") as exc:
        build_measurement(four_qubit_code(), target, backend="devised")
    assert exc.value.backend == "devised_sticking"
    assert exc.value.logical_pauli == target


@pytest.mark.parametrize("backend", ["branching", "parallel_branching"])
def test_branching_is_not_silently_treated_as_measurement(backend):
    with pytest.raises(BackendMeasurementError, match="not a complete"):
        build_measurement(four_qubit_code(), "XI", backend=backend)


@pytest.mark.parametrize("target", ["II", "X", "XXX", "iXI", "AX", 0])
def test_invalid_logical_target_rejected_with_context(target):
    with pytest.raises(BackendMeasurementError) as exc:
        build_measurement(four_qubit_code(), target)
    assert exc.value.backend == "gauging"
    assert exc.value.logical_pauli == target
    assert exc.value.cause is not None


@pytest.mark.parametrize("option", ["logical", "logical_paulis", "targets", "logical_basis",
                                     "pauli", "logical_factors"])
def test_backend_options_cannot_replace_measurement(option):
    with pytest.raises(BackendMeasurementError, match="cannot override"):
        build_measurement(four_qubit_code(), "XI", backend_options={option: None})


def test_gauging_default_and_passthrough_preserve_distance_defaults(monkeypatch):
    from codesurgery.gauging_measurement import api
    calls = []
    original = api.gauging_deform

    def capture(code, target, **kwargs):
        calls.append(deepcopy(kwargs))
        # This mock only suppresses solving inside the test; wrapper must not.
        kwargs["require_distance"] = False
        return original(code, target, **kwargs)

    monkeypatch.setattr(api, "gauging_deform", capture)
    build_measurement(four_qubit_code(), "XI")
    assert "require_distance" not in calls[0]
    assert "distance_options" not in calls[0]
    options = {"require_distance": False, "exact_limit": 7}
    before = deepcopy(options)
    build_measurement(four_qubit_code(), "ZI", backend_options=options)
    assert options == before
    assert calls[1]["require_distance"] is False
    assert calls[1]["exact_limit"] == 7


def test_explicit_support_rejection_does_not_fall_back_to_gauging():
    # XXXXII contains several independent logical directions for this code.
    code = CSSCode([[1] * 6], [[1] * 6], distance=2)
    basis = code.standard_logicals()
    x = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
    a = x @ basis.jz.T % 2
    text = "".join("X" if bit else "I" for bit in a)
    # Standard representative may differ from x: request a matching paired
    # basis whose first generator is x instead for a deterministic fixture.
    from codesurgery.gf2 import solve
    transform = np.eye(code.k, dtype=np.uint8)
    index = int(np.flatnonzero(a)[0])
    transform[index] = a
    inverse = solve(transform, np.eye(code.k, dtype=np.uint8))
    custom = dict(jx=transform @ basis.jx % 2, jz=inverse.T @ basis.jz % 2)
    custom["jx"][index] = x
    text = "".join("X" if j == index else "I" for j in range(code.k))
    with pytest.raises(BackendMeasurementError, match="unintended") as exc:
        build_measurement(code, text, logical_basis=custom, backend="explicit",
                          backend_options={"require_distance": False})
    assert exc.value.backend == "explicit"


def test_independent_validation_catches_wrong_target_returned_by_backend(monkeypatch):
    from codesurgery.gauging_measurement import api
    wrong = api.gauging_deform(four_qubit_code(), pauli="Z", require_distance=False)
    monkeypatch.setattr(api, "gauging_deform", lambda *args, **kwargs: wrong)
    with pytest.raises(BackendMeasurementError, match="unintended"):
        build_measurement(four_qubit_code(), "XI")


def test_independent_validation_catches_changed_sign_even_if_backend_validator_lies(monkeypatch):
    from codesurgery.gauging_measurement import api
    r = api.gauging_deform(four_qubit_code(), pauli="X", require_distance=False)
    # Copy presentation into an unsealed object to exercise wrapper checks.
    fake = SimpleNamespace(**r.__dict__)
    fake.validate = lambda: True
    fake.measurement_sign ^= 1
    monkeypatch.setattr(api, "gauging_deform", lambda *args, **kwargs: fake)
    with pytest.raises(BackendMeasurementError, match="signed logical class"):
        build_measurement(four_qubit_code(), "XI")


def test_independent_validation_catches_extra_logical_readout(monkeypatch):
    from codesurgery.gauging_measurement import api
    r = api.gauging_deform(four_qubit_code(), pauli="X", require_distance=False)
    fake = SimpleNamespace(**r.__dict__)
    fake.validate = lambda: True
    extra_x = np.pad(four_qubit_code().logical("X", 1), (0, r.n - r.original_n))
    # This particular extra operator can fail commutation too; either failure
    # protects the protocol before it uses the backend result.
    fake.check_x = np.vstack((r.check_x, extra_x))
    fake.check_z = np.vstack((r.check_z, np.zeros(r.n, dtype=np.uint8)))
    fake.check_signs = np.append(r.check_signs, 0)
    monkeypatch.setattr(api, "gauging_deform", lambda *args, **kwargs: fake)
    with pytest.raises(BackendMeasurementError, match="commute|unintended"):
        build_measurement(four_qubit_code(), "XI")


def test_backend_distance_exception_candidate_is_preserved(monkeypatch):
    from codesurgery.high_rate_surgery import api
    candidate = object()
    original_error = ValueError("distance not certified")
    original_error.result = candidate

    def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(api, "high_rate_deform", fail)
    with pytest.raises(BackendMeasurementError, match="distance not certified") as exc:
        build_measurement(four_qubit_code(), "XI", backend="high_rate", backend_options={"strict": True})
    assert exc.value.cause is original_error
    assert exc.value.result is candidate


def test_backend_capabilities_and_aliases():
    assert describe_backend()["name"] == "gauging"
    assert describe_backend("high-rate-surgery")["name"] == "high_rate"
    assert describe_backend("explicit_gadgets")["mixed_pauli_supported"]
    assert not describe_backend("devised")["mixed_pauli_supported"]
    with pytest.raises(ValueError, match="Unknown backend"):
        describe_backend("nonexistent")

