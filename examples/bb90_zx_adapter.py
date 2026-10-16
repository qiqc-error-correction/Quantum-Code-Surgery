"""Joint Z_2 X_5 on one IBM BB90 block using a 10-qubit universal adapter.

Reduce each factor only above 2d, verify disjoint support, then construct
the signed mixed-Pauli code. A cyclic relabeling of the right SkipTree
keeps all full check rows within the fixed dressed-memory limit of seven.
"""
from pathlib import Path
import argparse

import numpy as np

from bb90_gauging import audit_export, write_json
from codesurgery.examples import ibm_bb90_code
from codesurgery.explicit_gadgets import gadget_pair_deform
from codesurgery.gf2 import in_rowspace, nullspace, rank
from codesurgery.logical_optimization import reduce_logical_weight
from codesurgery.symplectic import pauli_product


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results") / "bb90-zx")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    code = ibm_bb90_code()
    basis = code.standard_logicals()
    reductions = [reduce_logical_weight(code, pauli=p, logical_index=i, method="heuristic")
                  for p, i in (("Z", 1), ("X", 4))]
    factors = [{k: reduction[k] for k in ("x", "z", "sign")} for reduction in reductions]
    supports = [np.flatnonzero(np.asarray(t["x"]) | np.asarray(t["z"])) for t in factors]
    assert not np.intersect1d(*supports).size
    assert [len(s) for s in supports] == [10, 10]
    np.testing.assert_array_equal(np.asarray(factors[0]["z"]) ^ basis.jz[1],
        np.asarray(reductions[0]["hz_coefficients"]) @ code.hz % 2)
    np.testing.assert_array_equal(factors[1]["x"], basis.jx[4])

    result = gadget_pair_deform(code, *factors, adapter_size=10, adapter_shift=6,
        require_distance=True,
        distance_options=dict(distance_method="ris", distance_ris_trials=128))
    audit = audit_export(result, out / "bb90-z2x5-deformed")
    meta, connector = result.metadata, result.metadata["adapter"]
    assert not meta["distance_postprocessing"]["changed"]
    assert (result.n, result.k_stabilizer, result.check_x.shape) == (130, 7, (132, 130))
    assert (meta["max_check_weight"], meta["max_qubit_degree"]) == (7, 11)
    assert connector["adapter_size"] == 10 and connector["adapter_shift"] == 6

    cx, cz = result.check_x.astype(np.int64), result.check_z.astype(np.int64)
    checks = np.hstack((cx, cz))
    assert rank(checks) == 123
    cancel = nullspace(np.hstack((cx[:, 90:], cz[:, 90:])).T).astype(np.int64)
    readx, readz = cancel @ cx[:, :90] % 2, cancel @ cz[:, :90] % 2
    signatures = np.hstack((readx @ basis.jz.T % 2, readz @ basis.jx.T % 2))
    target_signature = np.zeros((1, 16), dtype=np.uint8)
    target_signature[0, [4, 9]] = 1  # logical X_5 and Z_2 coefficients.
    assert rank(signatures) == 1 and in_rowspace(signatures, target_signature)
    for factor in factors:
        sig = np.r_[np.asarray(factor["x"]) @ basis.jz.T % 2,
                    np.asarray(factor["z"]) @ basis.jx.T % 2]
        assert not in_rowspace(sig, signatures)

    bridge_columns = 90 + np.asarray(connector["edge_layout"]["adapter"])
    vertex_rows = np.asarray(meta["row_layout"]["vertex"])
    branch_products = []
    for factor, rows in zip(factors, (vertex_rows[:10], vertex_rows[10:])):
        x, z, phase = pauli_product(cx, cz, result.check_signs, rows)
        expected_x, expected_z = np.zeros(130, dtype=np.uint8), np.zeros(130, dtype=np.uint8)
        expected_x[:90], expected_z[:90] = factor["x"], factor["z"]
        expected_x[bridge_columns] = 1
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(z, expected_z)
        assert phase % 2 == 0
        branch_products.append(dict(x=x, z=z, sign=phase // 2))
    assert branch_products[0]["sign"] ^ branch_products[1]["sign"] == result.measurement_sign

    # X on all 10 adapter qubits is a surviving logical: it equals either
    # factor modulo the vertex stabilizers, while only their product is fixed.
    witness_x = np.zeros(130, dtype=np.uint8)
    witness_x[bridge_columns] = 1
    witness_z = np.zeros(130, dtype=np.uint8)
    assert not np.any(cz @ witness_x % 2)
    assert not in_rowspace(np.r_[witness_x, witness_z], checks)
    assert int(witness_x.sum()) == meta["distance_lower_bound"] == 10

    fz, fx = [item["restriction_matrix"] for item in meta["factors"]]
    for f in (fz, fx):
        assert f.shape == (15, 10) and rank(f) == 9 and np.all(f.sum(axis=1) == 2)
    matrices = dict(F_Z2=fz, F_X5=fx, B_joint=meta["graph_incidence"],
                    C_joint=meta["cycle_basis"], M_joint=meta["matching_matrix"],
                    C_X=result.check_x, C_Z=result.check_z,
                    T_left=connector["T_left"], T_right=connector["T_right"],
                    P_left=connector["P_left"], P_right=connector["P_right"], H_C=connector["H_C"])
    for name, matrix in matrices.items():
        np.save(out / f"{name}.npy", matrix)
        np.savetxt(out / f"{name}.csv", matrix, fmt="%d", delimiter=",")
    connections = [dict(adapter_qubit_one_based=int(q + 1),
                        left_memory_qubit_one_based=int(supports[0][u] + 1),
                        right_memory_qubit_one_based=int(supports[1][v] + 1))
                   for q, (u, v) in zip(bridge_columns, connector["matching"])]
    report = dict(target="Z_2 X_5 in the original standard logical basis", source_nkd=[90, 8, 10],
        factors=reductions, supports_one_based=[s + 1 for s in supports], physical_overlap=[],
        adapter_size=10, adapter_shift=6, adapter_connections=connections,
        connector_max_weight=connector["connector_max_weight"],
        branch_vertex_products=branch_products,
        target_signature=target_signature[0], actual_logical_readout_rank=rank(signatures),
        signature_order="X_1..X_8, Z_1..Z_8", neither_factor_separately_measured=True,
        deformed=audit, auxiliary_counts=dict(left=15, adapter=10, right=15),
        full_commutation="C_X C_Z.T + C_Z C_X.T = 0 mod 2",
        cycle_rank=rank(meta["cycle_basis"]), adapter_distance_proof=meta["adapter_distance_proof"],
        sparsity=meta["distance_postprocessing"]["sparsity_certificate"],
        static_distance=dict(distance=10, lower_bound=10, upper_bound=10,
            lower_bound_source="Published memory d=10; both factors beta_10=1; all adapter vertices are ports; adapter size 10 (Universal adapters Lemma 9)",
            witness=dict(x=witness_x, z=witness_z, weight=10, centralizer_verified=True, nonstabilizer_verified=True),
            numerical_search=meta["distance_calculation"]),
        references=["https://arxiv.org/html/2410.03628v4#S4", "https://arxiv.org/abs/2511.15989"])
    write_json(out / "bb90-zx-verification.json", report)
    write_json(out / "bb90-zx-input.json", dict(hx=code.hx, hz=code.hz, distance=10,
        logicals=factors, adapter_size=10, adapter_shift=6, require_distance=True,
        distance_options=dict(distance_method="ris", distance_ris_trials=128)))
    print("Joint Z2 X5 only: [[130,7,10]], disjoint weight-10 factors, 10 adapter qubits")
    print("Full commutator=0; max row/column weights=7/11; connector max weight=6")
    print(out / "bb90-zx-verification.json")
    return report


if __name__ == "__main__":
    main()
