"""Clean-room 6-ft straightedge geometry for transverse rut measurements.

The implementation models a rigid bar centered on each nominal wheel path.  A
bar is lowered until it is supported by the upper concave envelope of the
available half-lane surface.  Rut depth is the greatest perpendicular gap from
that supported line to a 4-inch averaged surface footprint.  The complete
footprint must lie inside both supporting contacts and the corresponding
wheel-path zone.  This is a documented geometric interpretation of the public
straightedge method, not a reproduction of unpublished PathView code.
"""

from __future__ import annotations

import math

import numpy as np

from .models import LaneGeometry, ReducedProfile, RutBarResult, WheelPathRut, WheelSide
from .shoulder import remove_shoulders

DEFAULT_RUT_BAR_LENGTH_INCHES = 72.0
DEFAULT_RUT_PATH_WIDTH_INCHES = 44.29134
DEFAULT_RUT_PATH_HALF_WIDTH_INCHES = 22.14567
DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES = 34.448819
DEFAULT_MEASUREMENT_WIDTH_INCHES = 4.0


def _upper_concave_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the least concave majorant's vertices, left to right."""

    hull: list[int] = []
    for index in range(x.size):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (x[second] - x[first]) * (y[index] - y[second]) - (y[second] - y[first]) * (x[index] - x[second])
            scale = max(
                1.0,
                abs((x[second] - x[first]) * (y[index] - y[second])),
                abs((y[second] - y[first]) * (x[index] - x[second])),
            )
            # Concavity requires successively non-increasing segment slopes.
            # Removing collinear middle points gives deterministic end contacts.
            if cross >= -32.0 * np.finfo(np.float64).eps * scale:
                hull.pop()
            else:
                break
        hull.append(index)
    return np.asarray(hull, dtype=np.int64)


def _supporting_line_at(
    x: np.ndarray,
    y: np.ndarray,
    target_x: float,
    slope_hint: float,
) -> tuple[float, float, int, int]:
    """Return ``(slope, y_at_target, left_contact, right_contact)``."""

    hull = _upper_concave_hull_indices(x, y)
    hull_x = x[hull]
    if hull.size < 2 or target_x < hull_x[0] or target_x > hull_x[-1]:
        raise ValueError("straightedge center is not bracketed by support points")

    position = int(np.searchsorted(hull_x, target_x, side="left"))
    exact = position < hull.size and math.isclose(
        float(hull_x[position]),
        target_x,
        rel_tol=0.0,
        abs_tol=16.0 * np.finfo(np.float64).eps * max(1.0, abs(target_x)),
    )
    if not exact:
        right_hull_position = position
        left_hull_position = position - 1
        left_index = int(hull[left_hull_position])
        right_index = int(hull[right_hull_position])
        slope = float((y[right_index] - y[left_index]) / (x[right_index] - x[left_index]))
        y_at_target = float(y[left_index] + slope * (target_x - x[left_index]))
        return slope, y_at_target, left_index, right_index

    vertex_position = position
    vertex_index = int(hull[vertex_position])
    left_slope = float("inf")
    right_slope = float("-inf")
    if vertex_position > 0:
        previous_index = int(hull[vertex_position - 1])
        left_slope = float((y[vertex_index] - y[previous_index]) / (x[vertex_index] - x[previous_index]))
    if vertex_position + 1 < hull.size:
        next_index = int(hull[vertex_position + 1])
        right_slope = float((y[next_index] - y[vertex_index]) / (x[next_index] - x[vertex_index]))
    slope = float(np.clip(slope_hint, right_slope, left_slope))

    if math.isclose(slope, left_slope, rel_tol=1e-12, abs_tol=1e-12) and vertex_position > 0:
        left_index = int(hull[vertex_position - 1])
        right_index = vertex_index
    elif math.isclose(slope, right_slope, rel_tol=1e-12, abs_tol=1e-12) and vertex_position + 1 < hull.size:
        left_index = vertex_index
        right_index = int(hull[vertex_position + 1])
    else:
        # A tangent strictly between the adjacent hull slopes has a single
        # physical contact at the vertex; repeat it in the two contact fields.
        left_index = right_index = vertex_index
    return slope, float(y[vertex_index]), left_index, right_index


def _piecewise_linear_values(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise-linear profile without extrapolation."""

    if np.any(query < x[0]) or np.any(query > x[-1]):
        raise ValueError("piecewise-linear query falls outside the profile")
    return np.interp(query, x, y)


def _piecewise_linear_integral(
    x: np.ndarray,
    y: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Integral of a piecewise-linear profile from ``x[0]`` to each query."""

    if x.size < 2:
        raise ValueError("piecewise-linear integration requires at least two points")
    query = np.asarray(query, dtype=np.float64)
    if np.any(query < x[0]) or np.any(query > x[-1]):
        raise ValueError("piecewise-linear integration query falls outside the profile")
    segment_area = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
    prefix_area = np.concatenate(([0.0], np.cumsum(segment_area, dtype=np.float64)))
    indices = np.searchsorted(x, query, side="right") - 1
    indices = np.clip(indices, 0, x.size - 2)
    offset = query - x[indices]
    segment_slope = (y[indices + 1] - y[indices]) / (x[indices + 1] - x[indices])
    return prefix_area[indices] + y[indices] * offset + 0.5 * segment_slope * np.square(offset)


def _maximum_footprint_gap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    center_left: float,
    center_right: float,
    measurement_width_inches: float,
    bar_slope: float,
    bar_y_at_wheel_center: float,
    wheel_center: float,
) -> tuple[float, float, float, int]:
    """Maximize bar-to-surface gap for a continuous averaging footprint.

    The surface is linearly interpolated between samples.  Its centered moving
    average is therefore piecewise quadratic.  Candidate interval boundaries
    occur when either footprint edge crosses a profile knot; within each
    interval the derivative is linear, so all stationary points are found
    without a sampling-grid approximation.
    """

    half_width = measurement_width_inches / 2.0
    if center_right < center_left:
        raise ValueError("no legal center can contain the complete measurement footprint")

    shifted_breaks = np.concatenate((x - half_width, x + half_width))
    internal_breaks = shifted_breaks[(shifted_breaks > center_left) & (shifted_breaks < center_right)]
    breaks = np.unique(np.concatenate(([center_left], internal_breaks, [center_right])))

    left_edge_y = _piecewise_linear_values(x, y, breaks - half_width)
    right_edge_y = _piecewise_linear_values(x, y, breaks + half_width)
    derivative = bar_slope - (right_edge_y - left_edge_y) / measurement_width_inches

    interval_widths = np.diff(breaks)
    derivative_change = np.diff(derivative)
    root_mask = (
        (interval_widths > 0.0)
        & (derivative[:-1] * derivative[1:] <= 0.0)
        & (np.abs(derivative_change) > 64.0 * np.finfo(np.float64).eps)
    )
    roots = breaks[:-1][root_mask] - derivative[:-1][root_mask] * (
        interval_widths[root_mask] / derivative_change[root_mask]
    )
    roots = roots[(roots > breaks[:-1][root_mask]) & (roots < breaks[1:][root_mask])]
    candidates = np.unique(np.concatenate((breaks, roots)))

    right_integral = _piecewise_linear_integral(x, y, candidates + half_width)
    left_integral = _piecewise_linear_integral(x, y, candidates - half_width)
    average_y = (right_integral - left_integral) / measurement_width_inches
    bar_y = bar_y_at_wheel_center + bar_slope * (candidates - wheel_center)
    normalizer = math.sqrt(1.0 + bar_slope**2)
    gaps = np.maximum(0.0, (bar_y - average_y) / normalizer)
    deepest = int(np.argmax(gaps))
    rut_x = float(candidates[deepest])
    footprint_mask = (x >= rut_x - half_width) & (x <= rut_x + half_width)
    return (
        rut_x,
        float(average_y[deepest]),
        float(gaps[deepest]),
        int(np.count_nonzero(footprint_mask)),
    )


def _profile_slope_hint(profile: ReducedProfile, lane: LaneGeometry) -> float:
    mask = (profile.x_inches >= lane.left_edge_inches) & (profile.x_inches <= lane.right_edge_inches)
    x = np.asarray(profile.x_inches[mask], dtype=np.float64)
    y = np.asarray(profile.elevation_inches[mask], dtype=np.float64)
    if x.size < 2:
        return 0.0
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    return 0.0 if denominator == 0.0 else float(np.dot(centered, y - np.mean(y)) / denominator)


def measure_rut_depth(
    profile: ReducedProfile,
    side: WheelSide,
    lane_geometry: LaneGeometry,
    *,
    bar_length_inches: float = DEFAULT_RUT_BAR_LENGTH_INCHES,
    measurement_width_inches: float = DEFAULT_MEASUREMENT_WIDTH_INCHES,
    slope_hint: float | None = None,
) -> WheelPathRut | None:
    """Measure one wheel path with a physically 72-in straightedge by default.

    Support points are restricted to the wheel path's half-lane.  The physical
    bar may overhang the lane center, mirroring independent left/right rut
    measurements and preventing a feature in the opposite wheel path from
    becoming a support point.  Measurements are limited to the span between
    the two contacts; an overhanging part of the bar is not a rut reference.
    """

    if side not in {"left", "right"}:
        raise ValueError(f"unsupported wheel-path side: {side!r}")
    if not np.isfinite(bar_length_inches) or bar_length_inches <= 0.0:
        raise ValueError("bar_length_inches must be finite and positive")
    if not np.isfinite(measurement_width_inches) or measurement_width_inches <= 0.0:
        raise ValueError("measurement_width_inches must be finite and positive")
    wheel_center = lane_geometry.wheel_path_center(side)
    half_lane_left, half_lane_right = lane_geometry.half_lane_bounds(side)
    if not half_lane_left < wheel_center < half_lane_right:
        return None

    if slope_hint is None:
        slope_hint = _profile_slope_hint(profile, lane_geometry)
    slope = float(slope_hint)
    support_x = support_y = None
    left_contact = right_contact = 0
    y_at_center = float("nan")

    # The horizontal projection changes slightly as the physical bar tilts.
    # Iterate because the selected edge sample can in turn change the tilt.
    for _ in range(8):
        half_horizontal_span = (bar_length_inches / 2.0) / math.sqrt(1.0 + slope**2)
        support_left = max(half_lane_left, wheel_center - half_horizontal_span)
        support_right = min(half_lane_right, wheel_center + half_horizontal_span)
        support_mask = (profile.x_inches >= support_left) & (profile.x_inches <= support_right)
        support_x = np.asarray(profile.x_inches[support_mask], dtype=np.float64)
        support_y = np.asarray(profile.elevation_inches[support_mask], dtype=np.float64)
        if support_x.size < 2 or support_x[0] > wheel_center or support_x[-1] < wheel_center:
            return None
        new_slope, y_at_center, left_contact, right_contact = _supporting_line_at(
            support_x,
            support_y,
            wheel_center,
            slope,
        )
        if math.isclose(new_slope, slope, rel_tol=1e-12, abs_tol=1e-12):
            slope = new_slope
            break
        slope = new_slope

    assert support_x is not None and support_y is not None
    half_horizontal_span = (bar_length_inches / 2.0) / math.sqrt(1.0 + slope**2)
    reference_left = wheel_center - half_horizontal_span
    reference_right = wheel_center + half_horizontal_span
    zone_left, zone_right = lane_geometry.wheel_path_bounds(side)
    if measurement_width_inches > zone_right - zone_left:
        raise ValueError("measurement_width_inches cannot exceed the wheel-path zone width")
    footprint_half_width = measurement_width_inches / 2.0
    measurement_center_left = max(
        zone_left + footprint_half_width,
        reference_left + footprint_half_width,
        float(support_x[left_contact]) + footprint_half_width,
        float(profile.x_inches[0]) + footprint_half_width,
    )
    measurement_center_right = min(
        zone_right - footprint_half_width,
        reference_right - footprint_half_width,
        float(support_x[right_contact]) - footprint_half_width,
        float(profile.x_inches[-1]) - footprint_half_width,
    )
    if measurement_center_right < measurement_center_left:
        return None

    rut_x, rut_elevation, rut_depth, footprint_point_count = _maximum_footprint_gap(
        np.asarray(profile.x_inches, dtype=np.float64),
        np.asarray(profile.elevation_inches, dtype=np.float64),
        center_left=measurement_center_left,
        center_right=measurement_center_right,
        measurement_width_inches=measurement_width_inches,
        bar_slope=slope,
        bar_y_at_wheel_center=y_at_center,
        wheel_center=wheel_center,
    )
    normalizer = math.sqrt(1.0 + slope**2)
    if rut_depth < 64.0 * np.finfo(np.float64).eps:
        rut_depth = 0.0
    vertical_gap = rut_depth * normalizer
    measurement_x = float(rut_x - slope * vertical_gap / (1.0 + slope**2))
    measurement_elevation = float(rut_elevation + vertical_gap / (1.0 + slope**2))

    return WheelPathRut(
        side=side,
        rut_depth_inches=float(rut_depth),
        rut_x_inches=float(rut_x),
        rut_elevation_inches=rut_elevation,
        bar_elevation_inches=measurement_elevation,
        bar_slope=slope,
        left_contact_x_inches=float(support_x[left_contact]),
        left_contact_elevation_inches=float(support_y[left_contact]),
        right_contact_x_inches=float(support_x[right_contact]),
        right_contact_elevation_inches=float(support_y[right_contact]),
        support_point_count=int(support_x.size),
        measurement_point_count=footprint_point_count,
        measurement_x_inches=measurement_x,
        measurement_elevation_inches=measurement_elevation,
        measurement_width_inches=float(measurement_width_inches),
    )


def measure_profile_rutting(
    profile: ReducedProfile,
    lane_geometry: LaneGeometry | None = None,
    *,
    lane_left_inches: float | None = None,
    lane_right_inches: float | None = None,
    lane_center_offset_inches: float = 0.0,
    bar_length_inches: float = DEFAULT_RUT_BAR_LENGTH_INCHES,
    measurement_width_inches: float = DEFAULT_MEASUREMENT_WIDTH_INCHES,
    remove_lane_shoulders: bool = True,
) -> RutBarResult:
    """Return left, right, and derived overall rut depths for one profile.

    Supply either a :class:`LaneGeometry` or explicit lane edges.  If neither is
    supplied, the first and last reduced-profile x coordinates define the lane.
    """

    if lane_geometry is not None and (
        lane_left_inches is not None or lane_right_inches is not None or lane_center_offset_inches != 0.0
    ):
        raise ValueError("provide lane_geometry or explicit lane arguments, not both")
    if lane_geometry is None:
        left = float(profile.x_inches[0]) if lane_left_inches is None else lane_left_inches
        right = float(profile.x_inches[-1]) if lane_right_inches is None else lane_right_inches
        lane_geometry = LaneGeometry(
            left_edge_inches=left,
            right_edge_inches=right,
            center_offset_inches=lane_center_offset_inches,
            wheel_path_center_offset_inches=DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES,
            wheel_path_width_inches=DEFAULT_RUT_PATH_WIDTH_INCHES,
        )

    # PathView's public rutting path removes geometric shoulders before it
    # finds supports, while its supplied cross-slope hint is calculated from
    # the original reduced profile. Keep those two inputs distinct.
    slope_hint = _profile_slope_hint(profile, lane_geometry)
    measurement_profile = remove_shoulders(profile, lane_geometry) if remove_lane_shoulders else profile
    left_result = measure_rut_depth(
        measurement_profile,
        "left",
        lane_geometry,
        bar_length_inches=bar_length_inches,
        measurement_width_inches=measurement_width_inches,
        slope_hint=slope_hint,
    )
    right_result = measure_rut_depth(
        measurement_profile,
        "right",
        lane_geometry,
        bar_length_inches=bar_length_inches,
        measurement_width_inches=measurement_width_inches,
        slope_hint=slope_hint,
    )
    return RutBarResult(
        left=left_result,
        right=right_result,
        bar_length_inches=float(bar_length_inches),
        lane_geometry=lane_geometry,
    )


calculate_rutting = measure_profile_rutting
