"""Single/pair CLI for strict explicit gadgets and auxiliary-graph gauging."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from ..models import CSSCode
from .api import gadget_deform, gadget_pair_deform
from ..gauging_measurement import gauging_deform, gauging_pair_deform


def _weight_threshold(value):
    if value == "2d":
        return value
    try:
        threshold = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected '2d' or a nonnegative integer") from error
    if threshold < 0:
        raise argparse.ArgumentTypeError("Expected '2d' or a nonnegative integer")
    return threshold


def _distance_choice(flag, config):
    if flag is not None:
        return flag
    if "require_distance" in config:
        if not isinstance(config["require_distance"], bool):
            raise ValueError("require_distance must be true or false")
        return config["require_distance"]
    if not sys.stdin.isatty():
        print("未指定是否保证码距：非交互模式使用基础构造；可加 --require-distance 或 --no-require-distance。", file=sys.stderr)
        return False
    while True:
        try:
            answer = input("是否需要保证码距？需要时将按需增强扩展并稀疏化校验 [y/N]：").strip().lower()
        except EOFError as error:
            raise ValueError("未收到选择；请使用 --require-distance 或 --no-require-distance") from error
        if answer in ("", "n", "no", "否", "不需要"):
            return False
        if answer in ("y", "yes", "是", "需要"):
            return True
        print("请输入 y 或 n。", file=sys.stderr)


def main(argv=None, *, default_scheme="explicit"):
    parser = argparse.ArgumentParser(description="Build an explicit or gauging measurement, optionally with a universal adapter")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("deformed-gadget.json"))
    parser.add_argument("--scheme", choices=("explicit", "gauging"))
    parser.add_argument("--logical", help="One signed physical IXYZ target")
    parser.add_argument("--pauli", choices=("X", "Y", "Z"))
    parser.add_argument("--logical-index", type=int)
    reduction = parser.add_mutually_exclusive_group()
    reduction.add_argument("--optimize-logical", dest="optimize_logical", action="store_true",
                           help="Reduce a single target within its stabilizer coset before constructing F")
    reduction.add_argument("--no-optimize-logical", dest="optimize_logical", action="store_false")
    parser.set_defaults(optimize_logical=None)
    parser.add_argument("--logical-weight-threshold", type=_weight_threshold,
                        help="With --optimize-logical, use '2d' (default) or an integer; optimize only above it")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--require-distance", dest="require_distance", action="store_true",
                          help="Require certified static distance and sparse-check bounds; repair if needed")
    decision.add_argument("--no-require-distance", dest="require_distance", action="store_false",
                          help="Keep the basic construction and report its existing certificates")
    parser.set_defaults(require_distance=None)
    parser.add_argument("--distance", type=int, help="Assert the memory-code distance or trusted lower bound")
    parser.add_argument("--expansion", choices=("none", "auto", "star", "complete"),
                        help="Explicit single-target only; non-none is an optional implementation heuristic")
    parser.add_argument("--exact-limit", type=int)
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--component-connection", choices=("adapter", "chain"),
                        help="Gauging automatic graph: connect components with repetition adapters (default) or single edges")
    parser.add_argument("--component-adapter-size", type=int)
    parser.add_argument("--component-adapter-shift", type=int)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Input JSON must be an object")
        hx, hz = config.get("hx", config.get("H_X")), config.get("hz", config.get("H_Z"))
        if hx is None or hz is None:
            raise ValueError("Input must supply hx and hz")
        if config.get("n") is not None:
            n = config["n"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError("n must be a positive integer")
            if hx == []:
                hx = np.zeros((0, n), dtype=np.uint8)
            if hz == []:
                hz = np.zeros((0, n), dtype=np.uint8)
        require_distance = _distance_choice(args.require_distance, config)
        distance = args.distance if args.distance is not None else config.get("distance")
        source = CSSCode(hx, hz, distance, config.get("name", "CSS code"))
        scheme = args.scheme or config.get("scheme", default_scheme)
        if scheme not in ("explicit", "gauging"):
            raise ValueError("scheme must be explicit or gauging")
        common = dict(logical_basis=config.get("logical_basis"),
            require_distance=require_distance, distance_options=config.get("distance_options"),
            exact_limit=args.exact_limit if args.exact_limit is not None else config.get("exact_limit", 20),
            max_entries=args.max_entries if args.max_entries is not None else config.get("max_entries", 50_000_000))
        component_options = {}
        for key in ("component_connection", "component_adapter_size", "component_adapter_shift"):
            value = getattr(args, key)
            if value is not None or key in config:
                component_options[key] = value if value is not None else config[key]
        if component_options and scheme != "gauging":
            raise ValueError("component connection options are available only for gauging")
        if scheme == "gauging":
            common.update(component_options)
        optimize_logical = (args.optimize_logical if args.optimize_logical is not None
                            else config.get("optimize_logical", False))
        optimization_options = config.get("logical_optimization_options")
        if args.logical_weight_threshold is not None:
            if optimization_options is not None and not isinstance(optimization_options, dict):
                raise ValueError("logical_optimization_options must be a dictionary")
            optimization_options = dict(optimization_options or {}, threshold=args.logical_weight_threshold)
        if "logicals" in config:
            if optimize_logical or optimization_options is not None:
                raise ValueError("For adapter factors, call reduce_logical_weight separately and recheck disjoint supports")
            factors = config["logicals"]
            if not isinstance(factors, list) or len(factors) != 2:
                raise ValueError("logicals must contain exactly two factors of one product")
            if "logical" in config or args.logical is not None or args.pauli is not None or args.logical_index is not None:
                raise ValueError("For pairs, use logicals, paulis and logical_indices")
            if (args.expansion or config.get("expansion", "none")) != "none" or "extra_rows" in config:
                raise ValueError("Pair construction uses the paper adapter, not automatic expansion")
            common.update(paulis=config.get("paulis", ("Z", "Z")),
                logical_indices=config.get("logical_indices", (0, 1)),
                adapter_size=config.get("adapter_size"), adapter_vertices=config.get("adapter_vertices"),
                adapter_shift=config.get("adapter_shift", 0),
                cycle_bases=config.get("cycle_bases"))
            if scheme == "gauging":
                common.update(graphs=config.get("graphs"), ports=config.get("ports"), matchings=config.get("matchings"))
            builder = gadget_pair_deform if scheme == "explicit" else gauging_pair_deform
            result = builder(source, factors[0], factors[1], **common)
        else:
            common.update(pauli=args.pauli or config.get("pauli", "Z"),
                logical_index=args.logical_index if args.logical_index is not None else config.get("logical_index", 0),
                cycle_basis=config.get("cycle_basis"), optimize_logical=optimize_logical,
                logical_optimization_options=optimization_options)
            if scheme == "explicit":
                common.update(expansion=args.expansion or config.get("expansion", "none"), extra_rows=config.get("extra_rows"))
            else:
                if (args.expansion or config.get("expansion", "none")) != "none" or "extra_rows" in config:
                    raise ValueError("Gauging uses graph/ports/matching inputs, not expansion heuristics")
                common.update(graph=config.get("graph"), ports=config.get("ports"), matching=config.get("matching"))
            builder = gadget_deform if scheme == "explicit" else gauging_deform
            result = builder(source, args.logical if args.logical is not None else config.get("logical"), **common)
        result.save(args.output)
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    main()
