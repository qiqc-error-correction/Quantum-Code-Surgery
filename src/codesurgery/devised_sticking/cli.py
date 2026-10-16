"""Separate CLI for devised sticking; the CKBB command remains unchanged."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..models import CSSCode
from .api import devised_deform


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build Zhang--Li finely devised measurement-sticker matrices")
    parser.add_argument("input", type=Path, help="JSON with hx/hz and optional logicals or logical")
    parser.add_argument("-o", "--output", type=Path, default=Path("devised-deformed.npz"))
    parser.add_argument("--pauli", choices=("X", "Z"))
    parser.add_argument("--logical", help="One physical pure X or Z Pauli string")
    parser.add_argument("--logical-index", type=int, default=None)
    parser.add_argument("--d-r", type=int, default=None, help="Repetition-code length d_R")
    parser.add_argument("--dense-glue", action="store_true", help="Keep the dressed glue before LDPC duplication")
    optimization = parser.add_mutually_exclusive_group()
    optimization.add_argument("--optimize-logical", dest="optimize_logical", action="store_true",
                              help="Reduce each logical representative before sticking (default threshold: weight > 2d)")
    optimization.add_argument("--no-optimize-logical", dest="optimize_logical", action="store_false",
                              help="Use the supplied physical representatives unchanged (default)")
    parser.set_defaults(optimize_logical=None)
    parser.add_argument("--optimize-always", action="store_true",
                        help="Enable logical optimization with no 2d threshold")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        hx = config.get("hx", config.get("H_X"))
        hz = config.get("hz", config.get("H_Z"))
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
        code = CSSCode(hx, hz, config.get("distance"), config.get("name", "CSS code"))
        if "logical" in config and "logicals" in config and args.logical is None:
            raise ValueError("Specify logical or logicals, not both")
        logicals = args.logical if args.logical is not None else config.get("logicals", config.get("logical"))
        optimize_logical = (args.optimize_logical if args.optimize_logical is not None
                            else config.get("optimize_logical", False))
        optimization_options = config.get("logical_optimization_options")
        if args.optimize_always:
            if args.optimize_logical is False:
                raise ValueError("--optimize-always cannot be combined with --no-optimize-logical")
            if optimization_options is not None and not isinstance(optimization_options, dict):
                raise ValueError("logical_optimization_options must be a dictionary")
            optimize_logical = True
            optimization_options = dict(optimization_options or {}, threshold=None)
        elif args.optimize_logical is False:
            # An explicit CLI disable overrides both optimization config fields.
            optimization_options = None
        result = devised_deform(
            code, logicals,
            pauli=args.pauli or config.get("pauli", "Z"),
            logical_index=args.logical_index if args.logical_index is not None else config.get("logical_index", 0),
            d_R=args.d_r if args.d_r is not None else config.get("d_R"),
            ldpc=False if args.dense_glue else config.get("ldpc", True),
            max_entries=config.get("max_entries", 50_000_000),
            optimize_logical=optimize_logical,
            logical_optimization_options=optimization_options,
        )
        result.save(args.output)
        print(json.dumps({"output": str(args.output), **result.summary()}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
