"""Pavement profile and rut-depth domain algorithms."""

from .aggregate import aggregate_rutting, severity_from_rut_depth
from .cross_slope import fit_cross_slope
from .models import (
    CrossSlopeResult,
    LaneGeometry,
    ReducedProfile,
    RutAggregate,
    RutBarResult,
    TransverseProfile,
    WheelPathRut,
)
from .reduction import ReductionConfig, reduce_profile
from .rutbar import (
    DEFAULT_MEASUREMENT_WIDTH_INCHES,
    DEFAULT_RUT_BAR_LENGTH_INCHES,
    DEFAULT_RUT_PATH_HALF_WIDTH_INCHES,
    DEFAULT_RUT_PATH_WIDTH_INCHES,
    DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES,
    measure_profile_rutting,
    measure_rut_depth,
)
from .shoulder import remove_shoulders, shoulder_trim_indices

__all__ = [
    "DEFAULT_MEASUREMENT_WIDTH_INCHES",
    "DEFAULT_RUT_BAR_LENGTH_INCHES",
    "DEFAULT_RUT_PATH_HALF_WIDTH_INCHES",
    "DEFAULT_RUT_PATH_WIDTH_INCHES",
    "DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES",
    "CrossSlopeResult",
    "LaneGeometry",
    "ReducedProfile",
    "ReductionConfig",
    "RutAggregate",
    "RutBarResult",
    "TransverseProfile",
    "WheelPathRut",
    "aggregate_rutting",
    "fit_cross_slope",
    "measure_profile_rutting",
    "measure_rut_depth",
    "reduce_profile",
    "remove_shoulders",
    "severity_from_rut_depth",
    "shoulder_trim_indices",
]
