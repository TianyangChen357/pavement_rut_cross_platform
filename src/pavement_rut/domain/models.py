"""Small, dependency-light data models for transverse-profile processing.

All distances and elevations in the domain layer are expressed in inches.  A
positive profile slope means that elevation increases as ``x`` increases.  The
cross-slope reporting convention is documented separately in
``cross_slope.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
WheelSide = Literal["left", "right"]


def _as_readonly_1d(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    array = np.array(array, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _validate_coordinates(x: FloatArray, elevation: FloatArray) -> None:
    if x.size != elevation.size:
        raise ValueError("x_inches and elevation_inches must have the same length")
    if x.size == 0:
        raise ValueError("a transverse profile must contain at least one point")
    if not np.all(np.isfinite(x)):
        raise ValueError("x_inches must contain only finite values")
    if x.size > 1 and np.any(np.diff(x) <= 0.0):
        raise ValueError("x_inches must be strictly increasing")


@dataclass(frozen=True, eq=False)
class TransverseProfile:
    """A calibrated transverse profile before domain-level reduction.

    Non-finite elevations are allowed here because invalid-value handling is an
    explicit part of :func:`pavement_rut.domain.reduction.reduce_profile`.
    """

    x_inches: ArrayLike
    elevation_inches: ArrayLike
    profile_id: str | None = None

    def __post_init__(self) -> None:
        x = _as_readonly_1d(self.x_inches, "x_inches")
        elevation = _as_readonly_1d(self.elevation_inches, "elevation_inches")
        _validate_coordinates(x, elevation)
        object.__setattr__(self, "x_inches", x)
        object.__setattr__(self, "elevation_inches", elevation)

    @property
    def point_count(self) -> int:
        return int(self.x_inches.size)


@dataclass(frozen=True, eq=False)
class ReducedProfile:
    """A finite, cropped profile ready for cross-slope/rutting calculations."""

    x_inches: ArrayLike
    elevation_inches: ArrayLike
    is_noisy: bool = False
    original_point_count: int | None = None
    valid_fraction: float = 1.0
    profile_id: str | None = None
    compatibility_notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        x = _as_readonly_1d(self.x_inches, "x_inches")
        elevation = _as_readonly_1d(self.elevation_inches, "elevation_inches")
        _validate_coordinates(x, elevation)
        if not np.all(np.isfinite(elevation)):
            raise ValueError("a ReducedProfile cannot contain non-finite elevations")
        if not 0.0 <= self.valid_fraction <= 1.0:
            raise ValueError("valid_fraction must be between zero and one")
        if self.original_point_count is not None and self.original_point_count < x.size:
            raise ValueError("original_point_count cannot be smaller than the reduced profile")
        object.__setattr__(self, "x_inches", x)
        object.__setattr__(self, "elevation_inches", elevation)
        object.__setattr__(self, "compatibility_notes", tuple(self.compatibility_notes))

    @property
    def point_count(self) -> int:
        return int(self.x_inches.size)


@dataclass(frozen=True)
class LaneGeometry:
    """Lane boundaries and the two wheel-path zones.

    ``left`` always means the smaller-x half of the profile.  Wheel-path
    centers are placed symmetrically around the lane center.  The default
    offsets/widths reproduce the public constants exposed by the currently
    used PathView runtime, but no proprietary implementation is copied.
    """

    left_edge_inches: float
    right_edge_inches: float
    center_offset_inches: float = 0.0
    wheel_path_center_offset_inches: float = 34.448819
    wheel_path_width_inches: float = 44.29134

    def __post_init__(self) -> None:
        values = (
            self.left_edge_inches,
            self.right_edge_inches,
            self.center_offset_inches,
            self.wheel_path_center_offset_inches,
            self.wheel_path_width_inches,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("lane geometry values must be finite")
        if self.right_edge_inches <= self.left_edge_inches:
            raise ValueError("right_edge_inches must be greater than left_edge_inches")
        if self.wheel_path_center_offset_inches < 0.0:
            raise ValueError("wheel_path_center_offset_inches cannot be negative")
        if self.wheel_path_width_inches <= 0.0:
            raise ValueError("wheel_path_width_inches must be positive")
        if not self.left_edge_inches < self.center_inches < self.right_edge_inches:
            raise ValueError("the offset lane center must remain within the lane boundaries")

    @property
    def center_inches(self) -> float:
        return (self.left_edge_inches + self.right_edge_inches) / 2.0 + self.center_offset_inches

    @property
    def wheel_path_half_width_inches(self) -> float:
        return self.wheel_path_width_inches / 2.0

    def wheel_path_center(self, side: WheelSide) -> float:
        if side == "left":
            return self.center_inches - self.wheel_path_center_offset_inches
        if side == "right":
            return self.center_inches + self.wheel_path_center_offset_inches
        raise ValueError(f"unsupported wheel-path side: {side!r}")

    def wheel_path_bounds(self, side: WheelSide) -> tuple[float, float]:
        center = self.wheel_path_center(side)
        half_width = self.wheel_path_half_width_inches
        if side == "left":
            return (
                max(self.left_edge_inches, center - half_width),
                min(self.center_inches, center + half_width),
            )
        if side == "right":
            return (
                max(self.center_inches, center - half_width),
                min(self.right_edge_inches, center + half_width),
            )
        raise ValueError(f"unsupported wheel-path side: {side!r}")

    def half_lane_bounds(self, side: WheelSide) -> tuple[float, float]:
        if side == "left":
            return self.left_edge_inches, self.center_inches
        if side == "right":
            return self.center_inches, self.right_edge_inches
        raise ValueError(f"unsupported wheel-path side: {side!r}")


@dataclass(frozen=True)
class CrossSlopeResult:
    """Cross-slope estimate and diagnostic reference-line fields."""

    rise_per_run: float
    percent: float
    angle_degrees: float
    intercept_inches: float
    r_squared: float
    point_count: int


@dataclass(frozen=True)
class WheelPathRut:
    """Straightedge measurement for one wheel path.

    The rut coordinate is the mean surface coordinate under the measurement
    footprint.  The measurement coordinate is its perpendicular projection
    onto the supported bar.
    """

    side: WheelSide
    rut_depth_inches: float
    rut_x_inches: float
    rut_elevation_inches: float
    bar_elevation_inches: float
    bar_slope: float
    left_contact_x_inches: float
    left_contact_elevation_inches: float
    right_contact_x_inches: float
    right_contact_elevation_inches: float
    support_point_count: int
    measurement_point_count: int
    measurement_x_inches: float | None = None
    measurement_elevation_inches: float | None = None
    measurement_width_inches: float = 4.0


@dataclass(frozen=True)
class RutBarResult:
    """Two-wheel-path result for one transverse profile."""

    left: WheelPathRut | None
    right: WheelPathRut | None
    bar_length_inches: float
    lane_geometry: LaneGeometry

    @property
    def overall_rut_depth_inches(self) -> float:
        if self.left is None or self.right is None:
            return float("nan")
        left = self.left.rut_depth_inches
        right = self.right.rut_depth_inches
        if not np.isfinite(left) or not np.isfinite(right):
            return float("nan")
        return float((left + right) / 2.0)


@dataclass(frozen=True)
class RutAggregate:
    """File/batch-level rut-depth summary."""

    left_average_inches: float
    right_average_inches: float
    overall_average_inches: float
    severity: int
    profiles_total: int
    profiles_with_any_result: int
    left_count: int
    right_count: int
