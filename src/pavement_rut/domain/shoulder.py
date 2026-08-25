"""Clean-room shoulder removal for reduced transverse profiles.

The rule in this module was inferred only from public PathView APIs and
input/output experiments.  It operates on the already reduced profile, detects
sustained steep runs in edge-search bands centered on the lane, and intersects
the result with the lane's strict-open bounds.  No vendor binary is needed at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pavement_rut.acceleration import try_shoulder_trim_indices

from .models import LaneGeometry, ReducedProfile

LOW_SLOPE_THRESHOLD = 0.055
HIGH_SLOPE_THRESHOLD = 0.17
MINIMUM_STEEP_SEGMENTS = 5
EDGE_SEARCH_FRACTION = 0.15
EDGE_SEARCH_CENTER_OFFSET_FRACTION = 0.5 - EDGE_SEARCH_FRACTION


@dataclass(frozen=True, slots=True)
class _SlopeRun:
    """Inclusive indices of one contiguous run in the segment-slope array."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _true_runs(mask: np.ndarray) -> list[_SlopeRun]:
    if mask.size == 0:
        return []
    padded = np.concatenate((np.array([False]), mask, np.array([False])))
    transitions = np.diff(padded.astype(np.int8, copy=False))
    starts = np.flatnonzero(transitions == 1)
    # Subtract one because these are inclusive indices into ``mask``.
    ends = np.flatnonzero(transitions == -1) - 1
    return [_SlopeRun(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _shoulder_trim_indices_python(
    profile: ReducedProfile,
    lane_geometry: LaneGeometry,
    *,
    profile_width_inches: float | None = None,
) -> tuple[int, int]:
    """Return the retained half-open slice ``(start, stop)`` in ``profile``.

    A shoulder candidate is a contiguous run of at least five adjacent
    segments whose absolute slopes are all greater than 0.055 in/in and which
    contains at least one slope greater than or equal to 0.17 in/in.  Search
    limits are 35 percent of the nominal profile width on either side of the
    lane center; equivalently, they bound the outer 15 percent bands when lane
    and profile are centered.  If several candidates exist on one side, the
    innermost candidate is selected.  The geometric result is then intersected
    with the strict-open lane bounds, excluding points exactly on either edge.

    ``profile_width_inches`` is normally inferred from the original sample
    count retained by :func:`reduce_profile`.  It can be supplied explicitly
    when a cropped :class:`ReducedProfile` was constructed by another caller.

    The returned indices refer to the original ``profile`` arrays, not a
    temporary lane-clipped array.  ``start == stop`` represents an empty
    retained sequence.
    """

    x = np.asarray(profile.x_inches, dtype=np.float64)
    y = np.asarray(profile.elevation_inches, dtype=np.float64)

    point_count = int(x.size)
    if point_count < 2:
        lane_start = int(np.searchsorted(x, lane_geometry.left_edge_inches, side="right"))
        lane_stop = int(np.searchsorted(x, lane_geometry.right_edge_inches, side="left"))
        return (lane_start, lane_stop) if lane_start < lane_stop else (lane_start, lane_start)

    slopes = np.abs(np.diff(y) / np.diff(x))
    low = slopes > LOW_SLOPE_THRESHOLD
    high = slopes >= HIGH_SLOPE_THRESHOLD
    qualifying_runs = [
        run
        for run in _true_runs(low)
        if run.length >= MINIMUM_STEEP_SEGMENTS and bool(np.any(high[run.start : run.end + 1]))
    ]

    if profile_width_inches is None:
        if profile.original_point_count is None:
            profile_width_inches = float(x[-1] - x[0])
        else:
            sample_spacing = float(np.median(np.diff(x)))
            profile_width_inches = sample_spacing * profile.original_point_count
    if not np.isfinite(profile_width_inches) or profile_width_inches <= 0.0:
        raise ValueError("profile_width_inches must be finite and positive")

    center_offset = EDGE_SEARCH_CENTER_OFFSET_FRACTION * profile_width_inches
    left_search_x = lane_geometry.center_inches - center_offset
    right_search_x = lane_geometry.center_inches + center_offset
    # The public behavior includes the segment beginning at the first sample
    # on or to the right of the continuous left search limit.
    left_edge_limit = int(np.searchsorted(x, left_search_x, side="left"))
    right_edge_limit = int(np.searchsorted(x, right_search_x, side="right")) - 1
    left_candidates = [run for run in qualifying_runs if run.start <= left_edge_limit]
    right_candidates = [run for run in qualifying_runs if run.end >= right_edge_limit]

    shoulder_start = 0
    if left_candidates:
        left_run = max(left_candidates, key=lambda run: run.start)
        shoulder_start = left_run.end + 2

    shoulder_stop = point_count
    if right_candidates:
        right_run = min(right_candidates, key=lambda run: run.start)
        shoulder_stop = right_run.start

    lane_start = int(np.searchsorted(x, lane_geometry.left_edge_inches, side="right"))
    lane_stop = int(np.searchsorted(x, lane_geometry.right_edge_inches, side="left"))
    retained_start = max(shoulder_start, lane_start)
    retained_stop = min(shoulder_stop, lane_stop)
    if retained_start >= retained_stop:
        empty_at = min(retained_start, point_count)
        return empty_at, empty_at
    return retained_start, retained_stop


def shoulder_trim_indices(
    profile: ReducedProfile,
    lane_geometry: LaneGeometry,
    *,
    profile_width_inches: float | None = None,
) -> tuple[int, int]:
    """Return the retained half-open slice, using JIT when available."""

    x = np.asarray(profile.x_inches, dtype=np.float64)
    y = np.asarray(profile.elevation_inches, dtype=np.float64)
    if x.size < 2:
        return _shoulder_trim_indices_python(
            profile,
            lane_geometry,
            profile_width_inches=profile_width_inches,
        )
    if profile_width_inches is None:
        if profile.original_point_count is None:
            profile_width_inches = float(x[-1] - x[0])
        else:
            sample_spacing = float(np.median(np.diff(x)))
            profile_width_inches = sample_spacing * profile.original_point_count
    if not np.isfinite(profile_width_inches) or profile_width_inches <= 0.0:
        raise ValueError("profile_width_inches must be finite and positive")

    fast_result = try_shoulder_trim_indices(
        x,
        y,
        lane_geometry.left_edge_inches,
        lane_geometry.right_edge_inches,
        lane_geometry.center_inches,
        profile_width_inches,
    )
    if fast_result is not None:
        return fast_result
    return _shoulder_trim_indices_python(
        profile,
        lane_geometry,
        profile_width_inches=profile_width_inches,
    )


def remove_shoulders(
    profile: ReducedProfile,
    lane_geometry: LaneGeometry,
    *,
    profile_width_inches: float | None = None,
) -> ReducedProfile:
    """Return a profile cropped to the lane after geometric shoulder removal.

    A :class:`ValueError` is raised when the strict lane clip and shoulder
    detection leave no points, because :class:`ReducedProfile` intentionally
    cannot represent an empty profile.
    """

    start, stop = shoulder_trim_indices(
        profile,
        lane_geometry,
        profile_width_inches=profile_width_inches,
    )
    if start >= stop:
        raise ValueError("shoulder removal left no profile points inside the lane")

    note = "clean-room shoulder removal applied"
    notes = profile.compatibility_notes
    if note not in notes:
        notes = (*notes, note)
    return ReducedProfile(
        x_inches=profile.x_inches[start:stop],
        elevation_inches=profile.elevation_inches[start:stop],
        is_noisy=profile.is_noisy,
        original_point_count=profile.original_point_count,
        valid_fraction=profile.valid_fraction,
        profile_id=profile.profile_id,
        compatibility_notes=notes,
    )


__all__ = [
    "EDGE_SEARCH_FRACTION",
    "EDGE_SEARCH_CENTER_OFFSET_FRACTION",
    "HIGH_SLOPE_THRESHOLD",
    "LOW_SLOPE_THRESHOLD",
    "MINIMUM_STEEP_SEGMENTS",
    "remove_shoulders",
    "shoulder_trim_indices",
]
