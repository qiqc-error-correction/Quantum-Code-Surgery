"""Matrix construction of quantum code surgery protocols."""
from .models import CSSCode, DeformedCode
from .logicals import LogicalBasis, standard_form_logicals
from .ckbb import ckbb_deform, ckbb_batch
from .measurement import UnintendedLogicalMeasurementError, validate_single_logical_measurement
from .logical_optimization import reduce_logical_weight
from .measurement_comparison import (compare_deformations, DeformationComparison,
                                     ComparisonConstructionError)


def deform(code, logical=None, *, scheme="ckbb", **kwargs):
    """Scheme-dispatch entry point, ready for additional constructions."""
    if scheme.lower() != "ckbb":
        raise ValueError(f"Unsupported surgery scheme: {scheme}")
    return ckbb_deform(code, logical, **kwargs)


__all__ = ["CSSCode", "DeformedCode", "LogicalBasis", "standard_form_logicals",
           "ckbb_deform", "ckbb_batch", "deform", "UnintendedLogicalMeasurementError",
           "validate_single_logical_measurement", "reduce_logical_weight",
           "compare_deformations", "DeformationComparison", "ComparisonConstructionError"]
