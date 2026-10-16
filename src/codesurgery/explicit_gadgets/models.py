"""Public result/error aliases shared by the strict graph-gadget routes."""
from ..gauging_measurement.models import GraphMeasurementCode as GadgetCode
from ..gadget_common import GaugeMeasurementRequired as GadgetReadoutError
from ..gadget_common import logical_basis_for, validate_readout as _validate_readout


def validate_readout(code, cx, cz, x, z):
    """Compatibility wrapper; full memory-only logical readout, standard basis."""
    return _validate_readout(code, cx, cz, (x, z, 0), logical_basis_for(code))
