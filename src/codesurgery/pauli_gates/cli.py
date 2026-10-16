"""Compile logical Clifford gates into sequential Pauli-measurement gadgets."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..examples import direct_sum, four_qubit_code, ibm_bb90_code
from ..models import CSSCode, _jsonable
from .api import compile_gates
from .backends import describe_backend


def _eight():
    return direct_sum(four_qubit_code(), four_qubit_code())


_EXAMPLES = {"four": four_qubit_code, "eight": _eight, "bb90": ibm_bb90_code}


def _json_argument(value, name):
    """Read JSON literals, or an explicitly prefixed @path."""
    try:
        return json.loads(Path(value[1:]).read_text(encoding="utf-8")
                          if value.startswith("@") else value)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{name}: expected JSON or @path: {error}") from error


def _gate(value):
    if not isinstance(value, str):
        return value
    fields = value.split(":")
    name = fields[0].strip().upper()
    arity = {"H": 1, "S": 1, "CNOT": 2}.get(name)
    if arity is None or len(fields) != arity + 1:
        raise ValueError("Gate syntax is H:target, S:target, or CNOT:control:target")
    try:
        return (name, *(int(item) for item in fields[1:]))
    except ValueError as error:
        raise ValueError("Gate operands must be zero-based integer logical indices") from error


def _source(config, example):
    selected = example if example is not None else config.get("example")
    hx, hz = config.get("hx", config.get("H_X")), config.get("hz", config.get("H_Z"))
    if selected is not None:
        if hx is not None or hz is not None:
            raise ValueError("Supply example or hx/hz matrices, not both")
        if selected not in _EXAMPLES:
            raise ValueError("example must be four, eight, or bb90")
        code = _EXAMPLES[selected]()
        return CSSCode(code.hx, code.hz, config.get("distance", code.distance),
                       config.get("name", code.name))
    if hx is None or hz is None:
        raise ValueError("Supply hx/hz matrices, or choose --example")
    if "n" in config:
        n = config["n"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if hx == []:
            hx = np.zeros((0, n), dtype=np.uint8)
        if hz == []:
            hz = np.zeros((0, n), dtype=np.uint8)
    return CSSCode(hx, hz, config.get("distance"), config.get("name", "CSS code"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Compile H, S, and CNOT into sequential nondestructive logical Pauli measurements. "
        "Logical index 0 is reserved as a reusable ancilla by default."))
    parser.add_argument("input", nargs="?", type=Path,
                        help="JSON containing code matrices/example, gates, and optional compiler settings")
    parser.add_argument("--example", choices=tuple(_EXAMPLES))
    parser.add_argument("--gate", action="append", dest="gates",
                        help="H:target, S:target, or CNOT:control:target; repeat in execution order")
    parser.add_argument("--ancilla", type=int, default=None,
                        help="Reserved zero-based original logical index (default 0)")
    parser.add_argument("--backend", default=None,
                        help="Surgery backend (default gauging); supported names are checked by the API")
    parser.add_argument("--backend-options", default=None,
                        help="JSON object or @path forwarded to every backend call")
    parser.add_argument("--decompose-only", action="store_true",
                        help="Export the logical protocol without constructing deformed codes")
    parser.add_argument("--no-ensure-distance", action="store_true",
                        help="Explicitly disable distance processing for gauging/explicit/high_rate")
    parser.add_argument("--outcomes", default=None,
                        help="JSON list or ID-to-outcome object, or @path; no outcomes are simulated by default")
    parser.add_argument("--outcome-format", choices=("bits", "eigenvalues"), default=None,
                        help="bits: 0/+1, 1/-1 (default); eigenvalues: +1 or -1")
    parser.add_argument("--initial-frame", default=None,
                        help="JSON initial Pauli frame accepted by plan.resolve, or @path")
    parser.add_argument("--matrix-format", choices=("npz", "json"), default=None)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output directory (default pauli-gates-output)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing files from a previous export")
    args = parser.parse_args(argv)

    try:
        config = {} if args.input is None else json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        code = _source(config, args.example)
        gates = args.gates if args.gates is not None else config.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError("Supply a nonempty gates list or one or more --gate options")
        gates = [_gate(item) for item in gates]
        backend = args.backend if args.backend is not None else config.get("backend", "gauging")
        if not isinstance(backend, str):
            raise ValueError("backend must be a string")
        backend = describe_backend(backend)["name"]
        options = config.get("backend_options", {})
        if not isinstance(options, dict):
            raise ValueError("backend_options must be a JSON object")
        options = dict(options)
        if args.backend_options is not None:
            overrides = _json_argument(args.backend_options, "backend-options")
            if not isinstance(overrides, dict):
                raise ValueError("backend-options must be a JSON object")
            options.update(overrides)
        if args.no_ensure_distance:
            if backend in ("gauging", "explicit"):
                options["require_distance"] = False
            elif backend == "high_rate":
                options["ensure_distance"] = False
            else:
                raise ValueError("--no-ensure-distance supports gauging, explicit, and high_rate only; "
                                 "use the selected backend's explicit options")
        build = False if args.decompose_only else config.get("build", True)
        if not isinstance(build, bool):
            raise ValueError("build must be true or false")
        plan = compile_gates(code, gates,
                             ancilla=args.ancilla if args.ancilla is not None else config.get("ancilla", 0),
                             logical_basis=config.get("logical_basis"), backend=backend,
                             backend_options=options, build=build)
        outcomes = (_json_argument(args.outcomes, "outcomes") if args.outcomes is not None
                    else config.get("outcomes"))
        initial_frame = (_json_argument(args.initial_frame, "initial-frame")
                         if args.initial_frame is not None else config.get("initial_frame"))
        resolution = None
        if outcomes is not None:
            resolution = plan.resolve(outcomes,
                outcome_format=args.outcome_format or config.get("outcome_format", "bits"),
                initial_frame=initial_frame)
        elif initial_frame is not None:
            raise ValueError("initial_frame requires outcomes")
        output = args.output if args.output is not None else Path(config.get("output", "pauli-gates-output"))
        matrix_format = args.matrix_format or config.get("matrix_format", "npz")
        correction_path = output / "corrections.json"
        if correction_path.exists() and not args.overwrite:
            raise FileExistsError("Correction output exists; use another directory or --overwrite")
        plan_path = plan.save(output, matrix_format=matrix_format, overwrite=args.overwrite)
        report = {"output": str(plan_path), **plan.summary()}
        if resolution is not None:
            correction_path.write_text(json.dumps(_jsonable(resolution), ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
            report["corrections"] = _jsonable(resolution)
            report["corrections_file"] = str(correction_path)
        elif args.overwrite and correction_path.exists():
            # A previous run's outcomes must not appear to belong to this plan.
            correction_path.unlink()
        print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
        step = getattr(error, "measurement_id", None)
        location = f"measurement {step}: " if step is not None else ""
        parser.exit(2, f"error: {location}{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
