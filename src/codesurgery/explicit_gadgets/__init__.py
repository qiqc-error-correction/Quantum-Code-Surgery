"""Webster--Smith--Cohen single-operator gadgets, independently of CKBB."""
from .expansion import exact_cheeger
from .models import GadgetCode, GadgetReadoutError
from ..gadget_common import GaugeMeasurementRequired


def gadget_deform(*args, **kwargs):
    from .api import gadget_deform as build
    return build(*args, **kwargs)


def gadget_pair_deform(*args, **kwargs):
    from .api import gadget_pair_deform as build
    return build(*args, **kwargs)

__all__ = ["gadget_deform", "gadget_pair_deform", "exact_cheeger", "GadgetCode", "GadgetReadoutError", "GaugeMeasurementRequired"]
