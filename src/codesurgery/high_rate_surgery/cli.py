"""Command line interface for shared-ancilla high-rate surgery."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ..examples import four_qubit_code, ibm_bb90_code, steane_code
from ..models import CSSCode, _jsonable
from .api import DistanceCertificationError, high_rate_deform


_EXAMPLES = {"bb90": ibm_bb90_code, "steane": steane_code, "four": four_qubit_code}


def _boolean(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _distance_choice(flag, config):
    """Ask only on an interactive, unspecified invocation."""
    if flag is not None:
        return flag
    if "ensure_distance" in config:
        return _boolean(config, "ensure_distance", True)
    if not sys.stdin.isatty():
        print("未指定是否保证码距：非交互模式默认开启；可使用 --no-ensure-distance 关闭。", file=sys.stderr)
        return True
    while True:
        print("是否需要保证码距？开启后将按需增强与分层，并报告认证状态 [Y/n]：", end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip().lower()
        except EOFError as error:
            raise ValueError("未收到选择；请指定 --ensure-distance 或 --no-ensure-distance") from error
        if answer in ("", "y", "yes", "是", "需要"):
            return True
        if answer in ("n", "no", "否", "不需要"):
            return False
        print("请输入 y 或 n。", file=sys.stderr)


def _source(config, example=None):
    example = example if example is not None else config.get("example")
    has_matrices = any(key in config for key in ("hx", "hz", "H_X", "H_Z"))
    if example is not None:
        if has_matrices:
            raise ValueError("Supply example or hx/hz matrices, not both")
        if example not in _EXAMPLES:
            raise ValueError("example must be bb90, steane, or four")
        code = _EXAMPLES[example]()
        return CSSCode(code.hx, code.hz, config.get("distance", code.distance),
                       config.get("name", code.name))
    hx = config.get("hx", config.get("H_X"))
    hz = config.get("hz", config.get("H_Z"))
    if hx is None or hz is None:
        raise ValueError("Supply hx and hz (or H_X and H_Z), or choose --example")
    if config.get("n") is not None:
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
        "Construct and validate a shared-ancilla high-rate surgery code; "
        "each target is a separately measured observable"))
    parser.add_argument("input", nargs="?", type=Path,
                        help="Input JSON with code matrices or example, and measurement options")
    parser.add_argument("--example", choices=tuple(_EXAMPLES),
                        help="Built-in memory code; may be combined with a target-only JSON file")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output .json or .npz (default high-rate-deformed.json); an explicit path also saves a valid strict-mode candidate before exit 2")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target", action="append", dest="targets",
                         help="Signed physical IXYZ string of length n; repeat for simultaneous measurements")
    targets.add_argument("--logical-pauli", action="append", dest="logical_paulis",
                         help="Signed logical IXYZ string of length k; repeat for simultaneous measurements")
    targets.add_argument("--logical-indices", nargs="+", type=int,
                         help="Zero-based basis indices; omitted targets default to all logicals of --pauli")
    parser.add_argument("--pauli", choices=("X", "Y", "Z"),
                        help="Basis target type, default Z")
    parser.add_argument("--optimize-support", action=argparse.BooleanOptionalAction, default=None,
                        help="Attempt support reduction before constructing the shared ancilla (default: enabled)")
    parser.add_argument("--ensure-distance", action=argparse.BooleanOptionalAction, default=None,
                        help="Attempt distance preservation; ask if unspecified in a terminal, otherwise default on")
    parser.add_argument("--target-distance", type=int,
                        help="Desired distance/lower bound; default from the source code when available")
    parser.add_argument("--distance-time-limit", type=float,
                        help="Shared distance/repair time budget in seconds after initialization (default: 60)")
    parser.add_argument("--distance-method", choices=("auto", "gurobi", "enum", "heuristic"),
                        help="Protected dressed-distance backend (default auto, using Gurobi when available)")
    parser.add_argument("--max-check-weight", type=int,
                        help="Full merged-code row-weight limit; default source maximum +2")
    parser.add_argument("--max-qubit-degree", type=int,
                        help="Full merged-code column-degree limit; default source maximum +2")
    parser.add_argument("--seed", type=int, help="Reproducible randomized-construction seed (default: 0)")
    parser.add_argument("--max-steps", type=int, help="Construction/repair step budget (default: 100)")
    parser.add_argument("--max-layers", type=int, help="Ancilla-layer budget (default: 20)")
    parser.add_argument("--trials", type=int, help="Randomized candidate trials per search (default: 64)")
    parser.add_argument("--max-entries", type=int, help="Dense-array construction budget (default: 50000000)")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None,
                        help="Fail if the requested construction certificates cannot be obtained")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else Path("high-rate-deformed.json")
    try:
        config = {} if args.input is None else json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        code = _source(config, args.example)

        # An explicit command-line target selection replaces the file selection.
        if args.targets is not None:
            physical, logical, indices = args.targets, None, None
        elif args.logical_paulis is not None:
            physical, logical, indices = None, args.logical_paulis, None
        elif args.logical_indices is not None:
            physical, logical, indices = None, None, args.logical_indices
        else:
            physical, logical, indices = (config.get("targets"), config.get("logical_paulis"),
                                           config.get("logical_indices"))
        if sum(value is not None for value in (physical, logical, indices)) > 1:
            raise ValueError("Choose targets, logical_paulis, or logical_indices, not multiple selectors")
        for key, value in (("targets", physical), ("logical_paulis", logical), ("logical_indices", indices)):
            if value is not None and not isinstance(value, list):
                raise ValueError(f"{key} must be a JSON list")

        options = dict(pauli=args.pauli or config.get("pauli", "Z"),
                       logical_indices=indices, logical_basis=config.get("logical_basis"),
                       logical_paulis=logical,
                       optimize_support=(args.optimize_support if args.optimize_support is not None
                                         else _boolean(config, "optimize_support", True)),
                       ensure_distance=_distance_choice(args.ensure_distance, config),
                       strict=args.strict if args.strict is not None else _boolean(config, "strict", False))
        defaults = dict(target_distance=None, distance_time_limit=60, distance_method="auto",
                        max_check_weight=None, max_qubit_degree=None, seed=0, max_steps=100, max_layers=20,
                        trials=64, max_entries=50_000_000)
        for key, default in defaults.items():
            cli_value = getattr(args, key)
            options[key] = cli_value if cli_value is not None else config.get(key, default)
        result = high_rate_deform(code, physical, **options)
        result.validate()
        result.save(output)
        print(json.dumps(_jsonable({"output": str(output), **result.summary()}),
                         ensure_ascii=False, indent=2))
        return 0
    except DistanceCertificationError as error:
        # Preserve a correct deformation only when the user selected an output
        # path. Exit 2 remains visible to scripts: saving is not certification.
        if args.output is not None:
            try:
                error.result.save(output)
            except (ValueError, TypeError, OSError) as save_error:
                parser.exit(2, f"error: {error}; could not save candidate: {save_error}\n")
            print(json.dumps(_jsonable({"output": str(output), **error.result.summary(),
                                       "strict_failed": True, "candidate_saved": True}),
                             ensure_ascii=False, indent=2))
            print(f"error: {error}\nValid but uncertified candidate saved to {output}", file=sys.stderr)
        else:
            print(f"error: {error}\nUse --output to retain a valid uncertified candidate.", file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
