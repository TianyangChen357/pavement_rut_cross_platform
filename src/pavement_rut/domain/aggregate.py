"""Aggregation helpers for per-profile rut-bar results."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .models import RutAggregate, RutBarResult

DEFAULT_SEVERITY_THRESHOLDS_INCHES = (0.25, 0.5, 1.0)


def severity_from_rut_depth(
    rut_depth_inches: float,
    thresholds_inches: tuple[float, float, float] = DEFAULT_SEVERITY_THRESHOLDS_INCHES,
) -> int:
    """Map an average depth to severity 0--3 using ascending thresholds."""

    if len(thresholds_inches) != 3:
        raise ValueError("exactly three severity thresholds are required")
    low, medium, high = map(float, thresholds_inches)
    if not (0.0 <= low < medium < high):
        raise ValueError("severity thresholds must be non-negative and strictly increasing")
    if not np.isfinite(rut_depth_inches):
        return -1
    if rut_depth_inches < 0.0:
        raise ValueError("rut depth cannot be negative")
    if rut_depth_inches < low:
        return 0
    if rut_depth_inches < medium:
        return 1
    if rut_depth_inches < high:
        return 2
    return 3


def _finite_average(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def aggregate_rutting(
    results: Iterable[RutBarResult | None],
    *,
    severity_thresholds_inches: tuple[float, float, float] = DEFAULT_SEVERITY_THRESHOLDS_INCHES,
    require_both_sides: bool = True,
) -> RutAggregate:
    """Aggregate left/right values, skipping missing or non-finite results.

    ``overall_average_inches`` is the mean of the left and right wheel-path
    averages.  By default, both sides must exist; otherwise overall is NaN and
    severity is -1.  This matches the existing exporter's file-level behavior.
    Set ``require_both_sides=False`` only when an available-side estimate is an
    explicitly accepted downstream policy.
    """

    materialized = list(results)
    left_values: list[float] = []
    right_values: list[float] = []
    profiles_with_any_result = 0
    for result in materialized:
        if result is None:
            continue
        any_result = False
        if result.left is not None and np.isfinite(result.left.rut_depth_inches):
            left_values.append(float(result.left.rut_depth_inches))
            any_result = True
        if result.right is not None and np.isfinite(result.right.rut_depth_inches):
            right_values.append(float(result.right.rut_depth_inches))
            any_result = True
        if any_result:
            profiles_with_any_result += 1

    left_average = _finite_average(left_values)
    right_average = _finite_average(right_values)
    if require_both_sides and not (np.isfinite(left_average) and np.isfinite(right_average)):
        overall_average = float("nan")
    else:
        overall_average = _finite_average([left_average, right_average])
    return RutAggregate(
        left_average_inches=left_average,
        right_average_inches=right_average,
        overall_average_inches=overall_average,
        severity=severity_from_rut_depth(overall_average, severity_thresholds_inches),
        profiles_total=len(materialized),
        profiles_with_any_result=profiles_with_any_result,
        left_count=len(left_values),
        right_count=len(right_values),
    )


aggregate_rut_results = aggregate_rutting
