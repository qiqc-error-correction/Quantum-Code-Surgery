"""Compile one logical Pauli measurement with an existing surgery backend.

Each string is ONE complete observable in the supplied paired logical basis.
In particular, ``"XZ"`` is never split into separate X and Z measurements.
The returned object is the backend's original deformed-code result, including
its own distance certificates and explicit unverified statuses.
"""
from __future__ import annotations

import numpy as np

from ..gadget_common import memory_checks, prepare, validate_readout
from ..gf2 import binary_matrix, binary_vector, nullspace, solve
from ..models import CSSCode
from ..symplectic import commutation_matrix, parse_pauli, pauli_product, pauli_string


SUPPORTED_BACKENDS = ("gauging", "ckbb", "explicit", "devised_sticking", "high_rate")

_ALIASES = {
    "gauging_measurement": "gauging",
    "gauge": "gauging",
    "gauge_measurement": "gauging",
    "explicit_gadgets": "explicit",
    "gadget": "explicit",
    "devised": "devised_sticking",
    "high_rate_surgery": "high_rate",
}

_RESERVED = {
    "code", "logical", "logicals", "targets", "logical_paulis", "pauli",
    "logical_index", "logical_indices", "logical_basis", "logical_factors",
}


class BackendMeasurementError(ValueError):
    """A requested backend cannot construct or validate this measurement.

``cause`` preserves the original failure; ``result`` preserves a candidate
attached by a backend's distance-certification exception, when available.
"""

    def __init__(self, backend, logical_pauli, cause):
        self.backend = backend
        self.logical_pauli = logical_pauli
        self.cause = cause
        self.result = getattr(cause, "result", None)
        super().__init__(
            f"Pauli measurement backend {backend!r}, target {logical_pauli!r}: {cause}"
        )


def _canonical_backend(backend):
    if not isinstance(backend, str):
        raise ValueError("backend must be a name from " + ", ".join(SUPPORTED_BACKENDS))
    name = backend.lower().replace("-", "_")
    name = _ALIASES.get(name, name)
    if name in ("branching", "parallel_branching"):
        raise ValueError(
            "parallel_branching relocates logical support; it is not a complete "
            "Pauli-measurement backend. Choose a measurement construction explicitly."
        )
    if name not in SUPPORTED_BACKENDS:
        raise ValueError("Unknown backend; choose " + ", ".join(SUPPORTED_BACKENDS))
    return name


def describe_backend(backend="gauging"):
    """Return capabilities without building a gadget or invoking a solver."""
    name = _canonical_backend(backend)
    return {
        "name": name,
        "single_complete_observable": True,
        "mixed_pauli_supported": name != "devised_sticking",
        "restrictions": (
            "The existing devised sticker supports pure X or pure Z only."
            if name == "devised_sticking" else
            "Construction-specific support, size, and distance checks still apply."
        ),
        "distance_options": {
            "gauging": ("require_distance", "distance_options"),
            "explicit": ("require_distance", "distance_options"),
            "high_rate": ("ensure_distance", "distance_time_limit", "strict"),
            "ckbb": ("layers",),
            "devised_sticking": ("d_R",),
        }[name],
        "defaults": "Unchanged from the selected measurement backend.",
    }


def _output_target(result):
    """Normalize the existing single/multiple-result conventions locally."""
    if hasattr(result, "target_x"):
        xs, zs, signs = result.target_x, result.target_z, result.target_signs
        witnesses = result.measurement_witnesses
        corrections = result.readout_sign_corrections
    else:
        xs, zs = result.measurement_x, result.measurement_z
        if np.asarray(xs).ndim == 1:
            xs, zs = np.asarray(xs)[None, :], np.asarray(zs)[None, :]
            signs = [result.measurement_sign]
            witnesses = np.asarray(result.measurement_witness)[None, :]
        else:
            signs, witnesses = result.measurement_signs, result.measurement_witnesses
        corrections = np.zeros(len(xs), dtype=np.uint8)
    xs = binary_matrix(xs, n_cols=result.original_n, name="output target X")
    zs = binary_matrix(zs, n_cols=result.original_n, name="output target Z")
    signs = binary_vector(signs, length=1, name="output target sign")
    corrections = binary_vector(corrections, length=1, name="readout sign correction")
    witnesses = binary_matrix(witnesses, n_cols=len(result.check_x), name="measurement witnesses")
    if len(xs) != 1 or len(zs) != 1 or len(witnesses) != 1:
        raise ValueError("Backend must return exactly one requested observable")
    return xs[0], zs[0], int(signs[0]), witnesses[0], int(corrections[0])


def _verify_output(result, source, target, basis):
    """Independently check the complete output and its signed logical class."""
    result.validate()
    if result.original_n != source.n:
        raise ValueError("Backend changed the original memory-coordinate count")
    cx = binary_matrix(result.check_x, name="check_x")
    cz = binary_matrix(result.check_z, n_cols=cx.shape[1], name="check_z")
    if cx.shape != cz.shape or cx.shape[1] < source.n:
        raise ValueError("Invalid deformed check dimensions")
    signs = binary_vector(result.check_signs, length=len(cx), name="check_signs")
    if np.any(commutation_matrix(cx, cz)):
        raise ValueError("Deformed checks do not commute")
    # The entire memory image, not just a single witness, must have rank one.
    validate_readout(source, cx, cz, target, basis)
    if result.protected_k != source.k - 1:
        raise ValueError("Backend must preserve exactly k-1 protected logical qubits")

    ox, oz, os, witness, correction = _output_target(result)
    memory_x, memory_z = memory_checks(source)
    difference = np.concatenate((target[0] ^ ox, target[1] ^ oz))
    coefficients = solve(np.hstack((memory_x, memory_z)).T, difference)
    # Optimized support is allowed only inside the SAME signed stabilizer coset.
    px, pz, phase = pauli_product(
        np.vstack((target[0], memory_x)), np.vstack((target[1], memory_z)),
        [target[2], *([0] * len(memory_x))],
        [0, *(np.flatnonzero(coefficients) + 1).tolist()],
    )
    if not np.array_equal(px, ox) or not np.array_equal(pz, oz) or phase != 2 * os:
        raise ValueError("Backend changed the requested signed logical class")

    wx, wz, phase = pauli_product(cx, cz, signs, np.flatnonzero(witness))
    if (not np.array_equal(wx, np.pad(ox, (0, cx.shape[1] - source.n)))
            or not np.array_equal(wz, np.pad(oz, (0, cx.shape[1] - source.n)))
            or phase != 2 * (os ^ correction)):
        raise ValueError("Backend measurement witness has the wrong Pauli or sign")
    for dependency in nullspace(np.hstack((cx, cz)).T):
        _, _, phase = pauli_product(cx, cz, signs, np.flatnonzero(dependency))
        if phase:
            raise ValueError("Backend signed check group contains -I")


def build_measurement(code, logical_pauli: str, *, logical_basis=None,
                      backend="gauging", backend_options=None):
    """Build one signed logical Pauli measurement on the original CSS memory.

    ``logical_pauli`` has length ``code.k`` (plus an optional leading +/-).
    Logical Y means ``i Xbar Zbar``; its canonical physical Pauli sign is
    computed explicitly, including nonstandard supplied representatives.

    Backend options are passed through without disabling automatic distance
    processing. They cannot override the target or logical basis. There is no
    silent fallback to a different scheme and no factorization into separate
    measurements. A returned candidate's distance status must still be read
    from its native result; successful construction alone is not a circuit
    fault-distance certificate.
    """
    name = backend
    try:
        name = _canonical_backend(backend)
        if not isinstance(code, CSSCode):
            raise TypeError("code must be a CSSCode")
        if backend_options is not None and not isinstance(backend_options, dict):
            raise TypeError("backend_options must be a dictionary")
        options = dict(backend_options or {})
        conflict = _RESERVED.intersection(options)
        if conflict:
            raise ValueError("backend_options cannot override " + ", ".join(sorted(conflict)))
        a, b, sign = parse_pauli(logical_pauli, code.k)
        source, target, basis = prepare(
            code, dict(logical_x=a, logical_z=b, sign=sign),
            logical_basis=logical_basis,
            max_entries=options.get("max_entries", 50_000_000),
        )
        physical = dict(x=target[0], z=target[1], sign=target[2])
        if name == "gauging":
            from ..gauging_measurement.api import gauging_deform
            result = gauging_deform(source, physical, logical_basis=basis, **options)
        elif name == "explicit":
            from ..explicit_gadgets.api import gadget_deform
            result = gadget_deform(source, physical, logical_basis=basis, **options)
        elif name == "ckbb":
            from ..ckbb import ckbb_deform
            result = ckbb_deform(source, physical, **options)
        elif name == "high_rate":
            from ..high_rate_surgery.api import high_rate_deform
            result = high_rate_deform(source, targets=[physical], logical_basis=basis, **options)
        else:
            if np.any(target[0]) and np.any(target[1]):
                raise ValueError(
                    "devised_sticking supports pure X or pure Z measurements only; "
                    "its mixed-Pauli Appendix I protocol is not implemented. "
                    "Select a mixed-Pauli measurement backend such as gauging."
                )
            from ..devised_sticking.api import devised_deform
            pure = "X" if np.any(target[0]) else "Z"
            result = devised_deform(source, pauli_string(*target), pauli=pure, **options)
        _verify_output(result, source, target, basis)
        return result
    except BackendMeasurementError:
        raise
    except Exception as error:
        raise BackendMeasurementError(name, logical_pauli, error) from error


__all__ = ["SUPPORTED_BACKENDS", "BackendMeasurementError", "describe_backend", "build_measurement"]
