"""Certified WSC explicit gadget for the standard-form Z_1 of IBM BB90.

The supplied Hamilton G and sparse closure basis are instance-specific
candidates. The builder rechecks their algebra, all parity cuts, sole target
readout, and full check weights; it does not trust a stored certificate.
"""
from pathlib import Path
import argparse

import numpy as np

from bb90_gauging import audit_export, write_json
from codesurgery.examples import ibm_bb90_code
from codesurgery.explicit_gadgets import gadget_deform
from codesurgery.explicit_gadgets.expansion import exact_cheeger
from codesurgery.distance_preservation import code_distance
from codesurgery.gf2 import in_rowspace, rank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "bb90-explicit")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = Path(__file__).resolve().parent / "bb90-explicit-data"
    code = ibm_bb90_code()
    basis = code.standard_logicals()
    g = np.load(data / "G.npy", allow_pickle=False)
    cycles = np.load(data / "C_sparse.npy", allow_pickle=False)

    # These arrays are tied to this precise standard representative/column order.
    basic = gadget_deform(code, pauli="Z", logical_index=0,
                          require_distance=False, exact_limit=1)
    f = basic.metadata["raw_restriction_matrix"]
    a = np.vstack((f, g))
    assert f.shape == (34, 24) and g.shape == (24, 24)
    assert np.all(g.sum(0) == 2) and np.all(g.sum(1) == 2)
    assert not np.any(cycles.astype(np.int64) @ a.astype(np.int64) % 2)
    assert rank(cycles) == len(a) - rank(a) == 35
    base_report = audit_export(basic, out / "bb90-explicit-z1-basic")
    base_report["distance_search"] = code_distance(basic, method="ris", ris_trials=256, seed=0)

    # w=24 requires a larger exact-enumeration allocation budget than the
    # library default. This does not change the fixed row-weight limit of 7.
    exact_options = dict(exact_limit=24, max_entries=250_000_000)
    base_report["expansion"] = exact_cheeger(f, **exact_options)
    result = gadget_deform(code, pauli="Z", logical_index=0,
                           extra_rows=g, cycle_basis=cycles,
                           require_distance=True, **exact_options)
    final_report = audit_export(result, out / "bb90-explicit-z1-certified")
    meta = result.metadata
    np.testing.assert_array_equal(meta["restriction_matrix"], a)
    np.testing.assert_array_equal(meta["cycle_basis"], cycles)
    assert meta["incidence_kind"] == "even_tanner"
    assert meta["cheeger_constant"] == 1
    assert result.n == 148 and result.k_stabilizer == 7
    assert final_report["max_check_weight"] == final_report["max_qubit_degree"] == 7

    # Recheck the numerical upper-bound witness independently of its report.
    search = meta["distance_calculation"]
    witness = search["witness"]
    wx, wz = np.asarray(witness["x"]), np.asarray(witness["z"])
    assert not np.any((result.check_x.astype(np.int64) @ wz
                       + result.check_z.astype(np.int64) @ wx) % 2)
    assert not in_rowspace(np.r_[wx, wz], np.hstack((result.check_x, result.check_z)))
    upper = int(np.count_nonzero(wx | wz))
    lower = int(meta["distance_lower_bound"])
    assert lower == upper == 10
    final_report.update(
        hx_shape=list(result.hx.shape), hz_shape=list(result.hz.shape),
        expansion=meta["boundary_expansion"],
        sparsity=meta["distance_postprocessing"]["sparsity_certificate"],
        distance_search=search,
        combined_distance_certificate=dict(
            distance=10, proven_lower_bound=lower, verified_upper_bound=upper,
            lower_bound_source="Exact h([F;G])=1 and the published source distance 10",
            upper_bound_source="Explicit commuting non-stabilizer Pauli of weight 10",
            source_distance_assumption="IBM BB90 static distance 10 from Table 3",
            numerical_search_alone_exact=search["distance"] is not None,
            distance_kind="static stabilizer distance"))
    support = np.flatnonzero(basis.jz[0])
    selector = np.eye(code.n, dtype=np.uint8)[support]
    p = meta["pasting_matrix"][:len(code.hx)]
    hx = np.block([[code.hx, p], [np.zeros((35, 90), dtype=np.uint8), cycles]])
    hz = np.block([[code.hz, np.zeros((45, 58), dtype=np.uint8)], [selector, a.T]])
    np.testing.assert_array_equal(result.hx, hx)
    np.testing.assert_array_equal(result.hz, hz)
    assert not np.any(hx.astype(np.int64) @ hz.astype(np.int64).T % 2)
    matrices = dict(F=f, G=g, F_augmented=a, C=cycles, S=selector, P=p,
                    HX_deformed=hx, HZ_deformed=hz)
    for name, matrix in matrices.items():
        np.save(out / f"{name}.npy", matrix)
        np.savetxt(out / f"{name}.csv", matrix, fmt="%d", delimiter=",")
    write_json(out / "bb90-explicit-matrix-layout.json", dict(
        column_groups=dict(memory=[0, 90], F_ancillas=[90, 124], G_ancillas=[124, 148]),
        intervals="zero-based, right endpoint excluded",
        support_one_based=support + 1,
        active_hx_rows_one_based=meta["touching_memory_check_rows"] + 1,
        shapes={name: list(value.shape) for name, value in matrices.items()},
        HX_formula="[[HX, P], [0, C]]", HZ_formula="[[HZ, 0], [S, F_augmented.T]]"))
    write_json(out / "bb90-input.json", dict(hx=code.hx, hz=code.hz, distance=10,
               name=code.name, distance_provenance="Published static distance, not recomputed here"))
    write_json(out / "bb90-standard-basis.json", vars(basis))
    report = dict(
        scheme="Webster--Smith--Cohen explicit gadget", target="standard-form Z_1",
        source=dict(n=90, k=8, published_static_distance=10),
        basic=base_report, certified=final_report,
        candidate_provenance="G is a selected Hamilton cycle; C is a complete low-weight kernel basis. These are instance-specific implementation choices, not a paper-prescribed Hamilton algorithm.",
        references=["https://arxiv.org/abs/2511.15989", "https://arxiv.org/html/2308.07915v2#S4.T3"])
    write_json(out / "bb90-explicit-verification.json", report)
    print("BB90 Z1 explicit: [[148,7,10]], h=1, max row/column weight=7, commutator=0")
    print(out / "bb90-explicit-verification.json")
    return report


if __name__ == "__main__":
    main()
