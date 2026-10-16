"""Independent branching construction from Parallel Logical Measurements.

Adaptive routing retains targets on memory when attachment budgets permit,
and transports the others to disjoint leaves. The core preserves logical
information; a separate paired-gauging preview audits projected weights.
"""
from .api import branching_deform
from .models import BranchedCode
from .construction import append_branch, lift_paulis, retract_paulis

__all__ = ["branching_deform", "BranchedCode", "append_branch", "lift_paulis", "retract_paulis"]
