"""Compare surgery on supplied and stabilizer-equivalent reduced targets.

The same signed logical observables, basis and construction options are used
in both branches. No underlying builder may silently optimize the raw input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .gadget_common import prepare, signature
from .gf2 import nullspace, rank
from .logical_optimization import optimize_prepared_logical, verify_logical_optimization
from .models import CSSCode
from .symplectic import parse_pauli, pauli_string


_SCHEMES = ("ckbb", "devised_sticking", "explicit", "gauging", "high_rate", "branching")
_ALIASES = {"devised": "devised_sticking", "explicit_gadgets": "explicit",
            "gauging_measurement": "gauging", "high_rate_surgery": "high_rate",
            "parallel_branching": "branching"}
_RESERVED = {"code", "logical", "logicals", "targets", "logical_paulis", "pauli",
             "logical_index", "logical_indices", "logical_basis", "logical_factors",
             "optimize_logical", "logical_optimization_options", "optimize_support",
             "representatives"}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _target_record(target, basis):
    x, z, sign = target
    return dict(x=x.tolist(), z=z.tolist(), sign=int(sign),
                pauli=pauli_string(x, z, int(sign)),
                weight=int(np.count_nonzero(x | z)),
                support=np.flatnonzero(x | z).tolist(),
                logical_signature=signature(x, z, basis).tolist())


def _error_record(error):
    if error is None:
        return None
    record = dict(type=type(error).__name__, message=str(error))
    for name in ("reason", "details", "logical_rank", "measured_signatures", "target_signature"):
        value = getattr(error, name, None)
        if value is not None:
            record[name] = value
    return record


def _build(code, targets, basis, scheme, options):
    # Preserve literal input representatives in BOTH branches. Reduction is
    # performed once above the builders, so it cannot run again after comparison.
    physical = [dict(x=x.copy(), z=z.copy(), sign=int(s)) for x, z, s in targets]
    if scheme in ("ckbb", "explicit", "gauging"):
        if len(targets) != 1:
            raise ValueError(f"{scheme} comparison accepts one complete observable; "
                             "encode a product as one target, not a list of factors")
        if scheme == "ckbb":
            from .ckbb import ckbb_deform
            return ckbb_deform(code, physical[0], **options)
        if scheme == "explicit":
            from .explicit_gadgets import gadget_deform
            return gadget_deform(code, physical[0], logical_basis=basis,
                                 optimize_logical=False, **options)
        from .gauging_measurement import gauging_deform
        return gauging_deform(code, physical[0], logical_basis=basis,
                              optimize_logical=False, **options)
    if scheme == "devised_sticking":
        from .devised_sticking import devised_deform
        pure_x = all(not np.any(z) for _, z, _ in targets)
        pure_z = all(not np.any(x) for x, _, _ in targets)
        if not (pure_x or pure_z):
            raise ValueError("devised_sticking requires a same-type pure X or Z set; "
                             "use gauging for a single Y/XZ observable")
        strings = [pauli_string(*target) for target in targets]
        return devised_deform(code, strings, pauli="X" if pure_x else "Z",
                               optimize_logical=False, **options)
    if scheme == "high_rate":
        from .high_rate_surgery import high_rate_deform
        return high_rate_deform(code, targets=physical, logical_basis=basis,
                               optimize_support=False, **options)
    from .parallel_branching import branching_deform
    return branching_deform(code, targets=[pauli_string(*t) for t in targets],
                            logical_basis=basis, representatives="preserve",
                            optimize_logical=False, **options)


def _validate_construction(code, result, targets, basis, scheme):
    result.validate()
    expected_x = np.stack([t[0] for t in targets])
    expected_z = np.stack([t[1] for t in targets])
    expected_signs = np.asarray([t[2] for t in targets], dtype=np.uint8)
    if scheme == "branching":
        actual_x, actual_z = result.source["target_x"], result.source["target_z"]
        actual_signs = result.source["target_signs"]
    elif scheme == "high_rate":
        actual_x, actual_z, actual_signs = result.target_x, result.target_z, result.target_signs
    else:
        actual_x = np.asarray(result.measurement_x).reshape(-1, code.n)
        actual_z = np.asarray(result.measurement_z).reshape(-1, code.n)
        actual_signs = (result.measurement_signs if scheme == "devised_sticking"
                        else [result.measurement_sign])
    if (not np.array_equal(expected_x, actual_x)
            or not np.array_equal(expected_z, actual_z)
            or not np.array_equal(expected_signs, actual_signs)):
        raise ValueError("Builder changed the requested physical representative or its sign")

    # Recompute what the actual checks read after cancelling EVERY auxiliary
    # X/Z coordinate. A product such as X1 X2 is one row, hence one readout.
    x, z = result.check_x, result.check_z
    if result.original_n != code.n or result.n < code.n:
        raise ValueError("Builder output has the wrong memory block")
    if np.any((x @ z.T + z @ x.T) % 2):
        raise ValueError("Output checks do not commute")
    cancel = nullspace(np.hstack((x[:, code.n:], z[:, code.n:])).T)
    mx, mz = (cancel @ x[:, :code.n]) % 2, (cancel @ z[:, :code.n]) % 2
    if np.any((mx @ code.hz.T) % 2) or np.any((mz @ code.hx.T) % 2):
        raise ValueError("Output reads memory information outside the normalizer")
    actual = signature(mx, mz, basis)
    expected = (np.zeros((0, 2 * code.k), dtype=np.uint8) if scheme == "branching"
                else signature(expected_x, expected_z, basis))
    requested_rank = rank(expected)
    if rank(actual) != requested_rank or rank(np.vstack((actual, expected))) != requested_rank:
        raise ValueError("Output reads unintended logical information or misses requested targets")
    if result.protected_k != code.k - requested_rank:
        raise ValueError("Output has the wrong number of protected logical qubits")


@dataclass
class DeformationComparison:
    """Two native code results, plus signed representatives and audit reports.

    ``original``/``optimized`` are None for failed or unbuilt branches. Failures
    are recorded in ``errors``; they never select a different surgery scheme.
    Support indices in ``summary()`` are zero-based physical qubit indices.
    """
    scheme: str
    original: object | None
    optimized: object | None
    optimization: list[dict]
    errors: dict = field(default_factory=dict)
    _code: object = field(default=None, repr=False)
    _basis: dict = field(default_factory=dict, repr=False)
    _original_targets: list = field(default_factory=list, repr=False)
    _optimized_targets: list = field(default_factory=list, repr=False)
    _construction_requested: bool = field(default=True, repr=False)
    _optimization_completed: bool = field(default=True, repr=False)

    @property
    def original_targets(self):
        return [_target_record(t, self._basis) for t in self._original_targets]

    @property
    def optimized_targets(self):
        return [_target_record(t, self._basis) for t in self._optimized_targets]

    def overhead(self):
        """Compare the actual final code sizes before and after reduction.

        The primary overhead is ``ancilla_qubits = total_qubits - memory_qubits``.
        Counts include all auxiliary code qubits in each returned construction,
        including any completed repetition, expansion and sparsification steps.
        They exclude syndrome-extraction circuit ancillas, measurement rounds and
        spacetime cost. These are costs of the selected constructions, not minima.

        ``check_rows`` counts stored rows, including redundant and zero rows;
        ``check_row_delta`` is the net change from the supplied HX/HZ row count.
        It is not the number of newly introduced independent generators.
        Pauli support counts Y once in row weights and column degrees.

        ``savings`` is original minus optimized, so negative values mean increased
        cost. ``ancilla_percent`` uses the original ancilla count as denominator
        and is None when that count is zero. Failed/unbuilt costs and savings
        without two successful constructions are None, never inferred from weight.
        This report does not itself certify the code distance. Compare the native
        distance evidence in both results before claiming equal protection.
        ``construction_stage='branching_only'`` means no measurement gadget is
        included; its overhead is only that of the branching code.
        """
        memory_n = int(self._code.n)
        memory_rows = len(self._code.hx) + len(self._code.hz)
        branches = {}
        fields = ("total_qubits", "ancilla_qubits", "ancilla_fraction",
                  "total_qubit_factor", "check_rows", "check_row_delta",
                  "max_check_weight", "max_qubit_degree", "protected_k")
        for name in ("original", "optimized"):
            result = getattr(self, name)
            failed = name in self.errors
            record = dict.fromkeys(fields)
            record.update(status="failed" if failed else "constructed" if result is not None else "not_built",
                          memory_qubits=memory_n)
            if result is not None and not failed:
                total = int(result.n)
                ancilla = total - memory_n
                incidence = np.bitwise_or(result.check_x, result.check_z)
                rows = int(incidence.shape[0])
                row_weights = incidence.sum(axis=1)
                column_degrees = incidence.sum(axis=0)
                record.update(
                    total_qubits=total, ancilla_qubits=ancilla,
                    ancilla_fraction=ancilla / memory_n if memory_n else None,
                    total_qubit_factor=total / memory_n if memory_n else None,
                    check_rows=rows, check_row_delta=rows - memory_rows,
                    max_check_weight=int(row_weights.max()) if row_weights.size else 0,
                    max_qubit_degree=int(column_degrees.max()) if column_degrees.size else 0,
                    protected_k=int(result.protected_k),
                )
            branches[name] = record
        before, after = branches["original"], branches["optimized"]
        savings = None
        if before["status"] == after["status"] == "constructed":
            savings = {key: before[key] - after[key]
                       for key in ("ancilla_qubits", "total_qubits", "check_rows")}
            denominator = before["ancilla_qubits"]
            savings["ancilla_percent"] = (100.0 * savings["ancilla_qubits"] / denominator
                                          if denominator else None)
        return dict(
            metric="ancilla_qubits",
            construction_stage="branching_only" if self.scheme == "branching" else "deformed_code",
            scope="Final static code qubits; excludes syndrome-extraction circuit ancillas, "
                  "measurement rounds and spacetime cost. Branching alone is routing, not measurement.",
            input_check_rows=int(memory_rows), **branches, savings=savings,
        )

    def validate(self):
        if not self._optimization_completed:
            if (self.optimization or self._optimized_targets or self.optimized is not None
                    or "optimized" not in self.errors):
                raise ValueError("Failed reduction must retain an error and no fabricated optimized target")
        elif not (len(self.optimization) == len(self._original_targets) == len(self._optimized_targets)):
            raise ValueError("Comparison target/report counts disagree")
        for before, after, report in zip(self._original_targets, self._optimized_targets, self.optimization):
            verify_logical_optimization(self._code, after, self._basis, report)
            for actual, saved in zip(before, (report["original_x"], report["original_z"], report["original_sign"])):
                if not np.array_equal(actual, saved):
                    raise ValueError("Comparison report does not match the original input")
        for name, targets in (("original", self._original_targets), ("optimized", self._optimized_targets)):
            result = getattr(self, name)
            if result is not None:
                if name in self.errors:
                    raise ValueError("A failed construction cannot be presented as a valid result")
                _validate_construction(self._code, result, targets, self._basis, self.scheme)
            elif self._construction_requested and name not in self.errors:
                raise ValueError("A requested construction is missing without an error record")
        return True

    def summary(self):
        branches = {}
        for name in ("original", "optimized"):
            result = getattr(self, name)
            error = self.errors.get(name)
            branches[name] = dict(
                status="failed" if error else "constructed" if result is not None else "not_built",
                error=_error_record(error),
                targets=getattr(self, name + "_targets"),
                code=result.summary() if result is not None else None,
            )
        return dict(scheme=self.scheme, basis_origin=self._basis["origin"],
                    support_indexing="zero-based physical qubits", **branches,
                    overhead=self.overhead(),
                    optimization_status=([r["status"] for r in self.optimization]
                                         if self._optimization_completed else ["failed"] * len(self._original_targets)),
                    before_weights=[t["weight"] for t in self.original_targets],
                    after_weights=([t["weight"] for t in self.optimized_targets]
                                   if self._optimization_completed else None),
                    note="Logical-weight reduction searches a fixed signed stabilizer coset; "
                         "it is separate from distance calculation, graph expansion and sparsification.")

    def save(self, directory, *, matrix_format="npz"):
        """Write original/optimized matrices and comparison.json without overwriting."""
        if matrix_format not in ("npz", "json"):
            raise ValueError("matrix_format must be npz or json")
        self.validate()
        directory = Path(directory)
        names = ["comparison.json", f"original.{matrix_format}", f"optimized.{matrix_format}"]
        if any((directory / name).exists() for name in names):
            raise FileExistsError("Comparison output already exists; choose another directory")
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("original", "optimized"):
            result = getattr(self, name)
            if result is not None:
                path = directory / f"{name}.{matrix_format}"
                result.save(path)
                files[name] = path.name
        report = directory / "comparison.json"
        report.write_text(json.dumps(dict(summary=self.summary(), optimization=self.optimization,
                                         logical_basis=self._basis, code_files=files),
                                     default=_json_default, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        return report


class ComparisonConstructionError(ValueError):
    """Both branches were attempted; ``comparison`` retains successes and errors."""
    def __init__(self, comparison):
        self.comparison = comparison
        details = "; ".join(f"{name}: {type(error).__name__}: {error}"
                            for name, error in comparison.errors.items())
        super().__init__("Surgery comparison has failed construction(s): " + details)


def compare_deformations(code, logical=None, *, scheme="gauging", logicals=None,
                         logical_paulis=None, pauli="Z", logical_index=0,
                         logical_basis=None, optimization_options=None,
                         construction_options=None, build=True, strict=True):
    """Return original and reduced representatives and their separate code builds.

    ``logical`` accepts the same single-observable inputs as gauging_deform:
    physical Pauli string/vector or a physical/logical-coefficient dict.
    ``logicals`` is an explicit collection of such observables. Alternatively,
    ``logical_paulis`` supplies a string or list of strings of length code.k.
    Each list entry is one measurement: ['ZZ'] is a product; ['ZI','IZ'] is two
    separate measurements. With no targets, select one standard-form generator.

    Reduction follows the existing strict weight > 2d rule; unknown d skips it.
    Use optimization_options={'threshold': None} to attempt all input weights.
    Each target retains its signed logical class. Reduction need not succeed or
    be globally optimal, and reduced supports need not reduce gadget overhead.
    Multiple targets must also satisfy the selected scheme's compatibility rules.

    The returned comparison.overhead() gives the actual auxiliary/total qubits,
    stored check rows, row/column weights and savings from both final code builds.
    This report is also included in summary() and saved comparison.json files.

    Construction options (distance limits, repetition length, etc.) apply equally
    to both branches. Native distance processing defaults are retained. To compare
    basic gauging/explicit matrices set construction_options={'require_distance':
    False}; for high_rate use {'ensure_distance': False}. Branching preserves all
    logical qubits and is not a measurement. CKBB's implementation is unchanged.

    build=False returns representatives/reduction reports without any code build.
    By default a failure in either branch raises ComparisonConstructionError AFTER
    trying both; a reduction failure still allows the raw construction to run.
    error.comparison retains the other result. strict=False returns
    that same object with explicit errors. It never labels a failed result valid.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    if not isinstance(scheme, str):
        raise ValueError("scheme must name a supported surgery construction")
    scheme = _ALIASES.get(scheme.lower().replace("-", "_"), scheme.lower().replace("-", "_"))
    if scheme not in _SCHEMES:
        raise ValueError("Unknown scheme; choose " + ", ".join(_SCHEMES))
    for name, value in (("build", build), ("strict", strict)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    for name, value in (("optimization_options", optimization_options),
                        ("construction_options", construction_options)):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"{name} must be a dictionary")
    options = dict(construction_options or {})
    conflicts = _RESERVED.intersection(options)
    if conflicts:
        raise ValueError("construction_options cannot override " + ", ".join(sorted(conflicts)))
    if sum(value is not None for value in (logical, logicals, logical_paulis)) > 1:
        raise ValueError("Choose logical, logicals or logical_paulis, not more than one")
    if logical_paulis is not None:
        labels = [logical_paulis] if isinstance(logical_paulis, str) else list(logical_paulis)
        items = []
        for label in labels:
            a, b, sign = parse_pauli(label, code.k)
            items.append(dict(logical_x=a, logical_z=b, sign=sign))
    elif logicals is not None:
        if isinstance(logicals, (str, dict)):
            raise ValueError("logicals must be a collection; use logical for a single observable")
        items = list(logicals)
    else:
        items = [logical]
    if not items:
        raise ValueError("At least one logical observable is required")
    prepared = [prepare(code, item, pauli=pauli, logical_index=logical_index,
                        logical_basis=logical_basis,
                        max_entries=options.get("max_entries", 50_000_000)) for item in items]
    source, _, basis = prepared[0]
    original = [(x.copy(), z.copy(), int(s)) for _, (x, z, s), _ in prepared]
    comparison = DeformationComparison(scheme, None, None, [], _code=source,
        _basis=basis, _original_targets=original, _construction_requested=bool(build),
        _optimization_completed=False)
    try:
        optimized, reports = [], []
        for target in original:
            after, report = optimize_prepared_logical(source, target, basis, **(optimization_options or {}))
            optimized.append(after)
            reports.append(report)
        comparison._optimized_targets = optimized
        comparison.optimization = reports
        comparison._optimization_completed = True
    except Exception as error:
        comparison.errors["optimized"] = error
    if build:
        for name, targets in (("original", original), ("optimized", comparison._optimized_targets)):
            if name in comparison.errors:
                continue
            try:
                result = _build(source, targets, basis, scheme, options)
                _validate_construction(source, result, targets, basis, scheme)
                setattr(comparison, name, result)
            except Exception as error:
                comparison.errors[name] = error
    comparison.validate()
    if strict and comparison.errors:
        raise ComparisonConstructionError(comparison)
    return comparison


__all__ = ["compare_deformations", "DeformationComparison", "ComparisonConstructionError"]
