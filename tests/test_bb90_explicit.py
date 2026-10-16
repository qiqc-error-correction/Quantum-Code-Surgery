"""BB90 explicit Tanner gadget checks, without repeating exponential h search."""
from pathlib import Path

import numpy as np
import pytest

from codesurgery.examples import ibm_bb90_code
from codesurgery.explicit_gadgets import gadget_deform
from codesurgery.gf2 import rank
from test_bb90_gauging import _verify_actual_matrices


DATA = Path(__file__).resolve().parents[1] / "examples" / "bb90-explicit-data"


@pytest.fixture(scope="module")
def bb90_explicit_inputs():
    return (ibm_bb90_code(), np.load(DATA / "G.npy", allow_pickle=False),
            np.load(DATA / "C_sparse.npy", allow_pickle=False))


def test_bb90_explicit_preserves_raw_f_and_only_measures_z1(bb90_explicit_inputs):
    code, supplement, cycles = bb90_explicit_inputs
    result = gadget_deform(code, pauli="Z", extra_rows=supplement,
                           cycle_basis=cycles, exact_limit=1,
                           require_distance=False)
    assert result.validate()
    assert result.n == 148
    assert result.n - code.n == 58
    assert result.k_stabilizer == result.protected_k == 7
    assert result.is_css

    # Recompute the original restriction from the published memory checks;
    # neither of its weight-four rows may be split into weight-two edges.
    support = np.flatnonzero(code.standard_logicals().jz[0])
    expected_f = code.hx[:, support]
    expected_f = expected_f[np.any(expected_f, axis=1)]
    assert expected_f.shape == (34, 24)
    weights = expected_f.sum(axis=1)
    assert np.count_nonzero(weights == 4) == 2
    assert np.count_nonzero(weights == 2) == 32
    np.testing.assert_array_equal(result.metadata["raw_restriction_matrix"], expected_f)
    augmented = result.metadata["restriction_matrix"]
    np.testing.assert_array_equal(augmented, np.vstack((expected_f, supplement)))
    assert augmented.shape == (58, 24)
    assert supplement.shape == (24, 24)
    assert np.all(supplement.sum(axis=1) == 2)

    # The supplied low-weight dependencies form the entire closure space,
    # not merely a commuting subset that leaves extra auxiliary logicals.
    actual_cycles = result.metadata["cycle_basis"]
    np.testing.assert_array_equal(actual_cycles, cycles)
    assert cycles.shape == (35, 58)
    assert rank(cycles) == len(augmented) - rank(augmented) == 35
    assert not np.any((cycles.astype(np.int64) @ augmented.astype(np.int64)) % 2)
    assert int(cycles.sum(axis=1).max()) == 5
    assert int(cycles.sum(axis=0).max()) == 4

    # Recheck the actual complete matrices, readout signatures, and phases.
    _verify_actual_matrices(code, vars(result), "Z")
    full_support = np.bitwise_or(result.check_x, result.check_z)
    assert full_support.shape == (149, 148)
    assert int(full_support.sum(axis=1).max()) == 7
    assert int(full_support.sum(axis=0).max()) == 7


def test_bb90_explicit_rejects_incomplete_closure(bb90_explicit_inputs):
    code, supplement, cycles = bb90_explicit_inputs
    with pytest.raises(ValueError, match="cycle_basis.*complete.*kernel"):
        gadget_deform(code, pauli="Z", extra_rows=supplement,
                      cycle_basis=cycles[:-1], exact_limit=1,
                      require_distance=False)


def test_bb90_explicit_rejects_odd_supplementary_row(bb90_explicit_inputs):
    code, supplement, cycles = bb90_explicit_inputs
    malformed = supplement.copy()
    malformed[0, 0] ^= 1
    with pytest.raises(ValueError, match="extra_rows.*even weight"):
        gadget_deform(code, pauli="Z", extra_rows=malformed,
                      cycle_basis=cycles, exact_limit=1,
                      require_distance=False)
