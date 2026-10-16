"""Build a branching code; this command does not perform logical measurements."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..distance_preservation import code_distance
from ..models import CSSCode
from .api import branching_deform


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Branch targets only when direct measurement would exceed source check/qubit weight limits; no logical measurement is performed")
    parser.add_argument("input", type=Path,
        help="JSON with hx/hz; optionally targets or logical_paulis, logical_basis, and logical_indices")
    parser.add_argument("-o", "--output", type=Path, default=Path("branched.json"))
    parser.add_argument("--force-copy", action="store_true", default=None,
        help="Force a leaf copy for a single target even when direct measurement fits the limits")
    parser.add_argument("--branching-mode", choices=("adaptive", "always"),
        help="adaptive (default): branch only as needed for the source weight limits; always: build the complete balanced tree")
    parser.add_argument("--max-check-increase", type=int,
        help="Maximum increase for each original check's row weight (default: 2)")
    parser.add_argument("--max-qubit-increase", type=int,
        help="Maximum increase for each original physical qubit's check participation (default: 2)")
    parser.add_argument("--representatives", choices=("preserve", "auto", "canonical"))
    parser.add_argument("--optimize-logical", action=argparse.BooleanOptionalAction, default=None,
        help="Reduce target support in the same signed logical class before branching (default: enabled; threshold strictly above 2d)")
    parser.add_argument("--leaf-pauli", choices=("auto", "X", "Z"))
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--compute-distance", action="store_true", default=None,
        help="After validation, compute static distance with the existing backend (auto, 60 seconds by default)")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        hx, hz = config.get("hx", config.get("H_X")), config.get("hz", config.get("H_Z"))
        if hx is None or hz is None:
            raise ValueError("Input must contain hx and hz (or H_X and H_Z)")
        if config.get("n") is not None:
            n = config["n"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("n must be a positive integer")
            if hx == []:
                hx = np.zeros((0, n), dtype=np.uint8)
            if hz == []:
                hz = np.zeros((0, n), dtype=np.uint8)
        if config.get("targets") is not None and config.get("logical_paulis") is not None:
            raise ValueError("Specify targets or logical_paulis, not both")
        compute_distance = args.compute_distance if args.compute_distance is not None else config.get("compute_distance", False)
        if not isinstance(compute_distance, bool):
            raise ValueError("compute_distance must be true or false")
        distance_options = config.get("distance_options", {})
        if not isinstance(distance_options, dict):
            raise ValueError("distance_options must be a JSON object of code_distance keyword arguments")
        if distance_options and not compute_distance:
            raise ValueError("distance_options requires compute_distance=true or --compute-distance")
        optimize_logical = args.optimize_logical if args.optimize_logical is not None else config.get("optimize_logical", True)
        if not isinstance(optimize_logical, bool):
            raise ValueError("optimize_logical must be true or false")
        optimization_options = config.get("logical_optimization_options")
        if optimization_options is not None and not isinstance(optimization_options, dict):
            raise ValueError("logical_optimization_options must be a JSON object")
        if optimization_options is not None and not optimize_logical:
            raise ValueError("logical_optimization_options requires optimize_logical=true or --optimize-logical")
        source = CSSCode(hx, hz, config.get("distance"), config.get("name", "CSS code"))
        result = branching_deform(source, config.get("targets"),
            logical_paulis=config.get("logical_paulis"),
            pauli=config.get("pauli", "Z"),
            logical_indices=config.get("logical_indices"),
            logical_basis=config.get("logical_basis"),
            representatives=args.representatives or config.get("representatives", "auto"),
            optimize_logical=optimize_logical,
            logical_optimization_options=optimization_options,
            leaf_pauli=args.leaf_pauli or config.get("leaf_pauli", "auto"),
            force_copy=args.force_copy if args.force_copy is not None else config.get("force_copy", False),
            branching_mode=args.branching_mode or config.get("branching_mode", "adaptive"),
            max_check_increase=args.max_check_increase if args.max_check_increase is not None else config.get("max_check_increase", 2),
            max_qubit_increase=args.max_qubit_increase if args.max_qubit_increase is not None else config.get("max_qubit_increase", 2),
            max_entries=args.max_entries if args.max_entries is not None else config.get("max_entries", 50_000_000))
        result.validate()
        if compute_distance:
            result.metadata["distance_calculation"] = code_distance(result, **distance_options)
            result.validate()
        result.save(args.output)
        print(json.dumps({"output": str(args.output), **result.summary(),
            "stage": "branching_only", "logical_measurement_performed": False},
            ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
