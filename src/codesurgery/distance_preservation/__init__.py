"""Computed stabilizer distances and certified graph postprocessing."""

from .distance import code_distance, stabilizer_distance, verify_distance


def ensure_distance(*args, **kwargs):
    from .api import ensure_distance as implementation
    return implementation(*args, **kwargs)


__all__ = ["ensure_distance", "code_distance", "stabilizer_distance", "verify_distance"]
