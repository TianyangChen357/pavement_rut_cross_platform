"""Cross-slope estimation from a reduced transverse profile."""

from __future__ import annotations

import math

import numpy as np

from .models import CrossSlopeResult, LaneGeometry, ReducedProfile
from .shoulder import remove_shoulders


def fit_cross_slope(
    profile: ReducedProfile,
    lane_geometry: LaneGeometry | None = None,
    *,
    lane_left_inches: float | None = None,
    lane_right_inches: float | None = None,
    positive_downward: bool = True,
    remove_lane_shoulders: bool = True,
) -> CrossSlopeResult:
    """Estimate cross slope from the mean elevations of the lane halves.

    The default clean-room compatibility path first removes shoulders, then
    includes the lane-center point (when present) in both half-lane means. The
    rise is ``mean(left_y) - mean(right_y)`` and the run is half the retained
    profile span. ``rise_per_run`` and the default reported percent are thus
    positive when the surface falls toward increasing x. Pass
    ``positive_downward=False`` to reverse that reporting convention.

    Set ``remove_lane_shoulders=False`` only when the caller has already passed
    an identically trimmed sequence; lane bounds are still applied strictly.
    """

    if lane_geometry is not None and (lane_left_inches is not None or lane_right_inches is not None):
        raise ValueError("provide lane_geometry or explicit lane bounds, not both")
    if lane_geometry is not None:
        lane = lane_geometry
    else:
        left = float(profile.x_inches[0]) if lane_left_inches is None else lane_left_inches
        right = float(profile.x_inches[-1]) if lane_right_inches is None else lane_right_inches
        lane = LaneGeometry(left_edge_inches=left, right_edge_inches=right)
    if not np.isfinite(lane.left_edge_inches) or not np.isfinite(lane.right_edge_inches):
        raise ValueError("cross-slope lane bounds must be finite and increasing")

    selected = remove_shoulders(profile, lane) if remove_lane_shoulders else profile
    x = np.asarray(selected.x_inches, dtype=np.float64)
    elevation = np.asarray(selected.elevation_inches, dtype=np.float64)
    inside = (x > lane.left_edge_inches) & (x < lane.right_edge_inches)
    x = x[inside]
    elevation = elevation[inside]
    if x.size == 0:
        raise ValueError("at least one point inside the lane bounds is required")

    center = lane.center_inches
    left_mask = x <= center
    right_mask = x >= center
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("cross-slope calculation requires points in both lane halves")
    run = float((x[-1] - x[0]) / 2.0)
    rise = float(np.mean(elevation[left_mask]) - np.mean(elevation[right_mask]))
    if run == 0.0:
        # The observed public API includes a lone center point in both halves
        # and returns NaN rather than throwing when its retained span is zero.
        return CrossSlopeResult(
            rise_per_run=float("nan"),
            percent=float("nan"),
            angle_degrees=float("nan"),
            intercept_inches=float(elevation[0]),
            r_squared=float("nan"),
            point_count=1,
        )
    if run < 0.0:  # pragma: no cover - ReducedProfile guarantees increasing x
        raise ValueError("cross-slope calculation requires a non-negative retained span")
    slope = rise / run
    reported_slope = slope if positive_downward else -slope

    # Preserve the result model's general diagnostic fields.  The intercept is
    # the compatible two-half reference line at x=0; R-squared is evaluated
    # against that line and is informational, not part of the oracle contract.
    intercept = float(np.mean(elevation[left_mask]) + slope * np.mean(x[left_mask]))
    fitted = intercept - slope * x
    residual_sum_squares = float(np.sum(np.square(elevation - fitted)))
    centered_elevation = elevation - float(np.mean(elevation))
    total_sum_squares = float(np.dot(centered_elevation, centered_elevation))
    r_squared = 1.0 if total_sum_squares <= np.finfo(np.float64).eps else 1.0 - residual_sum_squares / total_sum_squares
    return CrossSlopeResult(
        rise_per_run=float(reported_slope),
        percent=float(100.0 * reported_slope),
        angle_degrees=float(math.degrees(math.atan(reported_slope))),
        intercept_inches=intercept,
        r_squared=float(r_squared),
        point_count=int(x.size),
    )


calculate_cross_slope = fit_cross_slope
