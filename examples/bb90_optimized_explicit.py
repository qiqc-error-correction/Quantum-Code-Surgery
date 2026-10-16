"""BB90 Z1: reduce its fixed stabilizer coset, then rebuild the WSC gadget.

The heuristic finds weight 12; this is not a proof that 12 is optimal in
the requested logical class. The resulting gadget's static distance 10
is certified separately by exact expansion and a weight-10 logical witness.
"""
from pathlib import Path
import argparse

import numpy as np

from bb90_gauging import audit_export, write_json
from codesurgery.examples import ibm_bb90_code
from codesurgery.explicit_gadgets import gadget_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "bb90-optimized")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = ibm_bb90_code()
    basis = source.standard_logicals()
    result = gadget_deform(source, pauli="Z", logical_index=0,
        optimize_logical=True,
        logical_optimization_options=dict(method="heuristic", seed=0),
        require_distance=True)
    meta = result.metadata
    reduction = meta["logical_optimization"]
    assert reduction["threshold_rule"] == "2d" and reduction["threshold"] == 20
    audit = audit_export(result, out / "bb90-optimized-z1-deformed")
    original = basis.jz[0]
    coefficients = np.asarray(reduction["hz_coefficients"], dtype=np.uint8)
    z = result.measurement_z
    np.testing.assert_array_equal(z, original ^ (coefficients @ source.hz % 2))
    np.testing.assert_array_equal(basis.jx.astype(np.int64) @ z % 2,
                                  [1, 0, 0, 0, 0, 0, 0, 0])
    assert reduction["before_weight"] == 24 and reduction["after_weight"] == 12
    assert not reduction["optimal"]
    assert result.n == 108 and result.k_stabilizer == 7
    assert meta["cheeger_constant"] == 1 and meta["added_edge_qubits"] == 0

    # Derive the entire logical readout directly from the exported check rows.
    cx, cz = result.check_x.astype(np.int64), result.check_z.astype(np.int64)
    cancellation = nullspace(np.hstack((cx[:, 90:], cz[:, 90:])).T).astype(np.int64)
    readx, readz = cancellation @ cx[:, :90] % 2, cancellation @ cz[:, :90] % 2
    signatures = np.hstack((readx @ basis.jz.T % 2, readz @ basis.jx.T % 2))
    expected = np.zeros((1, 16), dtype=np.uint8)
    expected[0, 8] = 1
    assert rank(signatures) == 1 and in_rowspace(signatures, expected)
    search = meta["distance_calculation"]
    witness = search["witness"]
    wx, wz = np.asarray(witness["x"]), np.asarray(witness["z"])
    assert not np.any((cx @ wz + cz @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack((cx, cz)))
    assert np.count_nonzero(wx | wz) == meta["distance_lower_bound"] == 10

    f = meta["raw_restriction_matrix"]
    assert f.shape == (18, 12) and rank(f) == 11
    np.testing.assert_array_equal(f, meta["restriction_matrix"])
    for name, matrix in dict(F=f, C=meta["cycle_basis"], S=meta["port_selector"],
                             HX_deformed=result.hx, HZ_deformed=result.hz).items():
        np.save(out / f"{name}.npy", matrix)
        np.savetxt(out / f"{name}.csv", matrix, fmt="%d", delimiter=",")
    write_json(out / "logical-reduction.json", reduction)
    report = dict(source=dict(n=90, k=8, published_static_distance=10),
        logical_target="same standard-form Z1 modulo original positive stabilizers",
        reduction=reduction, optimized_support_one_based=np.flatnonzero(z) + 1,
        multiplied_hz_rows_one_based=np.flatnonzero(coefficients) + 1,
        deformed=audit, f_shape=list(f.shape), hx_shape=list(result.hx.shape),
        hz_shape=list(result.hz.shape), cheeger=meta["expansion"],
        cycle_basis_shape=list(meta["cycle_basis"].shape),
        sparsity=meta["distance_postprocessing"]["sparsity_certificate"],
        actual_readout_verified_against_original_logical=True,
        distance=dict(exact_static_distance=10,
            lower_bound_source="published source distance 10 and exact h(F)=1",
            upper_bound_source="independently validated weight-10 non-stabilizer witness",
            numerical_search=search),
        references=["https://arxiv.org/abs/2511.15989", "https://arxiv.org/html/2308.07915v2#S4.T3"])
    write_json(out / "bb90-optimized-verification.json", report)
    print("Same Z1: weight 24 -> 12 (coset optimum not proven)")
    print("Rebuilt gadget: [[108,7,10]], F=18x12, h(F)=1, extra G rows=0")
    print("Commutator=0; only Z1 readout; max row/column weights=7/8")
    print(out / "bb90-optimized-verification.json")
    return report


if __name__ == "__main__":
    main()
