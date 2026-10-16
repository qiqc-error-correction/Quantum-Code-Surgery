"""JSON input and JSON/NPZ deformed-code output."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .models import CSSCode, _jsonable
from .ckbb import ckbb_deform, ckbb_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build CKBB deformed-code matrices for a CSS code")
    parser.add_argument("input", type=Path, help="JSON with hx, hz, optional distance and logical")
    parser.add_argument("-o", "--output", type=Path, default=Path("deformed.npz"))
    parser.add_argument("--pauli", choices=["X", "Y", "Z"], default=None)
    parser.add_argument("--logical", help="Physical I/X/Y/Z string; overrides JSON logical")
    parser.add_argument("--logical-index", type=int, default=0, help="Zero-based standard logical index")
    parser.add_argument("--layers", type=int, help="Number of dual layers; default distance, or n")
    parser.add_argument("--all-logicals", action="store_true", help="One separate deformation per standard-form generator")
    parser.add_argument("--no-bridge", action="store_true", help="Disable explicit Fig. 3 bridges and disjoint mixed connector arms")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        hx, hz = config.get("hx", config.get("H_X")), config.get("hz", config.get("H_Z"))
        if hx is None or hz is None:
            raise ValueError("Input must contain hx and hz (or H_X and H_Z)")
        if config.get("n") is not None:
            if hx == []:
                hx = np.zeros((0, config["n"]), dtype=np.uint8)
            if hz == []:
                hz = np.zeros((0, config["n"]), dtype=np.uint8)
        code = CSSCode(hx, hz, config.get("distance"), config.get("name", "CSS code"))
        pauli = args.pauli or config.get("pauli", "Z")
        if not isinstance(pauli, str) or pauli.upper() not in ("X", "Y", "Z"):
            raise ValueError("pauli must be X, Y or Z")
        logical = args.logical if args.logical is not None else config.get("logical")
        logical_factors = config.get("logical_factors")
        layers = args.layers if args.layers is not None else config.get("layers")
        bridge = False if args.no_bridge else config.get("bridge", True)
        common = {"pauli": pauli, "layers": layers, "bridge": bridge}
        if args.all_logicals:
            if logical is not None or logical_factors is not None:
                raise ValueError("--all-logicals cannot be combined with explicit logical or logical_factors")
            results = ckbb_batch(code, **common)
            paths = []
            for index, result in enumerate(results):
                path = args.output.with_name(f"{args.output.stem}_{pauli.lower()}{index}{args.output.suffix}")
                result.save(path)
                paths.append(str(path))
            print(json.dumps({"independent_deformations": paths, "summaries": [r.summary() for r in results]}, ensure_ascii=False, indent=2))
        else:
            result = ckbb_deform(code, logical, logical_factors=logical_factors,
                                 logical_index=args.logical_index, **common)
            result.save(args.output)
            print(json.dumps({"output": str(args.output), **result.summary()}, ensure_ascii=False, indent=2))
        basis = code.standard_logicals()
        basis_path = args.output.with_name(args.output.stem + "_logical_basis.json")
        basis_path.write_text(json.dumps(_jsonable({"jx": basis.jx, "jz": basis.jz,
                             "jx_standard": basis.jx_standard, "jz_standard": basis.jz_standard,
                             "column_permutation": basis.column_permutation}), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
