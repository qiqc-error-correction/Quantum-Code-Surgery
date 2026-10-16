"""IBM BB [[90,8,10]] gauging, full matrix exports, and commutation audit.

Run directly, or use --output-dir PATH. By default test standard-form Z_1
and Y_1 and attempt the existing automatic distance/sparsity pipeline.
The basic commuting codes are saved first, even if certification exceeds
the chosen resources. No failed certification is presented as preserved d=10.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from codesurgery.examples import ibm_bb90_code
from codesurgery.gauging_measurement import gauging_deform
from codesurgery.distance_preservation import code_distance, ensure_distance
from codesurgery.gf2 import rank, in_rowspace
from codesurgery.models import _jsonable
from codesurgery.symplectic import pauli_product, pauli_string


def write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def audit_export(result, prefix):
    """Reload actual exported arrays, then compute full integer commutation."""
    result.validate()
    result.save(prefix.with_suffix(".json"))
    result.save(prefix.with_suffix(".npz"))
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as disk:
        cx, cz = disk["check_x"].astype(np.int64), disk["check_z"].astype(np.int64)
        signs, witness = disk["check_signs"], disk["measurement_witness"]
    commutator = (cx @ cz.T + cz @ cx.T) % 2
    assert not np.any(commutator), "Exported stabilizers fail symplectic commutation"
    payload = json.loads(prefix.with_suffix(".json").read_text())
    assert np.array_equal(cx, payload["check_x"])
    assert np.array_equal(cz, payload["check_z"])
    assert np.array_equal(signs, payload["check_signs"])
    ox, oz, phase = pauli_product(cx, cz, signs, np.flatnonzero(witness))
    tx = np.pad(result.measurement_x, (0, result.n - result.original_n))
    tz = np.pad(result.measurement_z, (0, result.n - result.original_n))
    assert np.array_equal(ox, tx) and np.array_equal(oz, tz)
    assert phase == 2 * result.measurement_sign
    rows = [pauli_string(x, z, int(s)) for x, z, s in zip(cx, cz, signs)]
    prefix.with_name(prefix.name + "-stabilizers.txt").write_text("\n".join(rows) + "\n")
    np.save(prefix.with_name(prefix.name + "-commutator.npy"), commutator.astype(np.uint8))
    weights, degrees = (cx | cz).sum(1), (cx | cz).sum(0)
    return dict(n=result.n, k=result.k_stabilizer, ancillas=result.n-result.original_n,
        checks=len(cx), check_matrix_shape=list(cx.shape),
        stabilizer_rank=rank(np.hstack((cx, cz))), is_css=result.is_css,
        commutator_shape=list(commutator.shape), commutator_nonzeros=int(np.count_nonzero(commutator)),
        full_validation_passed=True, json_npz_equal=True, signed_readout_verified=True,
        logical_measurement_rank=result.metadata["logical_measurement_rank"],
        measurement_validation=result.metadata["measurement_validation"],
        target_pauli=pauli_string(result.measurement_x,result.measurement_z,result.measurement_sign),
        target_weight=int((result.measurement_x | result.measurement_z).sum()),
        max_check_weight=int(weights.max(initial=0)),max_qubit_degree=int(degrees.max(initial=0)),
        row_weights=weights.tolist(),column_degrees=degrees.tolist(),
        graph_shape=list(result.metadata["graph_incidence"].shape),
        cycle_checks=len(result.metadata["cycle_basis"]),
        distance_preservation_guaranteed=bool(result.metadata["distance_preservation_guaranteed"]),
        files={"json":prefix.with_suffix(".json").name,"npz":prefix.with_suffix(".npz").name,
               "stabilizers":prefix.name+"-stabilizers.txt","commutator":prefix.name+"-commutator.npy"})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("results") / "bb90-gauging")
    parser.add_argument("--paulis",nargs="+",choices=("X","Y","Z"),default=["Z","Y"])
    parser.add_argument("--logical-index",type=int,default=0,help="zero-based standard-form logical index")
    parser.add_argument("--basic-only",action="store_true",help="explicitly skip automatic distance/sparsity repair")
    parser.add_argument("--distance-method",choices=("auto","cluster","milp","enumeration"),default="auto")
    parser.add_argument("--distance-seconds",type=float,default=30.0)
    parser.add_argument("--max-candidates",type=int,default=2_000_000)
    parser.add_argument("--ris-trials",type=int,default=256)
    args=parser.parse_args(argv)
    if not 0 <= args.logical_index < 8:
        parser.error("logical-index must be between 0 and 7")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    code=ibm_bb90_code()
    basis=code.standard_logicals()
    assert (code.n,code.k,rank(code.hx),rank(code.hz)) == (90,8,41,41)
    assert not np.any((code.hx.astype(np.int64) @ code.hz.astype(np.int64).T)%2)
    source=dict(n=90,k=8,distance=10,hx=code.hx,hz=code.hz,name=code.name,
        definition=dict(ell=15,m=3,A="x^9+y+y^2",B="1+x^2+x^7",
            shift="S_r[i,(i+1) mod r]=1",column_order="left45,right45; local index 3*i+j"),
        distance_provenance="Published static code distance, not recomputed by the constructor",
        references=["https://arxiv.org/html/2308.07915v2#S4.T3",
                    "https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py"])
    write_json(out/"bb90-input.json",source)
    write_json(out/"bb90-standard-basis.json",vars(basis))
    report=dict(source=dict(n=90,k=8,published_distance=10,rank_hx=41,rank_hz=41,
                            hx_hz_commutator_nonzeros=0,hx_row_weight=6,hz_row_weight=6),
        logical_basis="paired CSS standard form", logical_index_zero_based=args.logical_index,
        distance_kind="static stabilizer distance; not circuit fault distance",cases={})
    for pauli in dict.fromkeys(args.paulis):
        label=f"{pauli.lower()}{args.logical_index+1}"
        basic=gauging_deform(code,pauli=pauli,logical_index=args.logical_index,require_distance=False)
        case=audit_export(basic,out/f"bb90-gauging-{label}-basic")
        report["cases"][label]={"basic":case}
        write_json(out/"bb90-verification.json",report)
        print(f"{label}: n={basic.n}, k={basic.k_stabilizer}, checks={len(basic.check_x)}, commutator nonzeros=0",flush=True)
        # Fast upper bound suffices to prove distance loss when a witness has
        # weight < 10. A separate complete solve is not needed for that claim.
        distance=code_distance(basic,method="ris",ris_trials=args.ris_trials,seed=0)
        w=distance["witness"]
        wx,wz=np.array(w["x"]),np.array(w["z"])
        assert not np.any((basic.check_x.astype(np.int64)@wz+basic.check_z.astype(np.int64)@wx)%2)
        assert not in_rowspace(np.r_[wx,wz],np.hstack((basic.check_x,basic.check_z)))
        case["distance_search"]=distance
        case["distance_witness_independently_verified"]=True
        print(f"{label}: distance bounds [{distance['proven_lower_bound']}, {distance['upper_bound']}]",flush=True)
        write_json(out/f"bb90-gauging-{label}-distance.json",distance)
        if args.basic_only:
            report["cases"][label]["automatic_processing"]={"attempted":False,"reason":"explicit --basic-only"}
        else:
            options=dict(distance_method=args.distance_method,distance_time_limit=args.distance_seconds,
                         distance_max_candidates=args.max_candidates,distance_ris_trials=args.ris_trials)
            start=time.monotonic()
            try:
                certified=ensure_distance(basic,**options)
            except ValueError as exc:
                report["cases"][label]["automatic_processing"]=dict(attempted=True,success=False,
                    error_type=type(exc).__name__,error=str(exc),
                    reason=getattr(exc,"reason",None),details=getattr(exc,"details",None),
                    elapsed_seconds=time.monotonic()-start,options=options)
                print(f"{label}: automatic processing stopped: {exc}",flush=True)
            else:
                report["cases"][label]["certified"]=audit_export(certified,out/f"bb90-gauging-{label}-certified")
                report["cases"][label]["automatic_processing"]=dict(attempted=True,success=True,
                    elapsed_seconds=time.monotonic()-start,options=options)
                print(f"{label}: certified result n={certified.n}",flush=True)
        write_json(out/"bb90-verification.json",report)
    print(out/"bb90-verification.json",flush=True)
    return report


if __name__=="__main__":
    main()
