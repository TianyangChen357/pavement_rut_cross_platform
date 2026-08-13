"""Configurable, clean-room reduction of calibrated transverse profiles.

This module intentionally makes each policy visible.  The
``pathview_observed`` factory captures the single-laser, empty-dark-band
behavior measured through the public PathView 7.2.04 API.  Double-laser seam
handling and non-empty dark-band behavior remain outside that compatibility
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .models import ReducedProfile, TransverseProfile

InvalidPolicy = Literal["interpolate", "drop", "raise"]


@dataclass(frozen=True)
class ReductionConfig:
    """Policies applied before a profile enters the measurement algorithms.

    ``roll_degrees`` applies a rigid clockwise rotation in the profile plane.
    Unless ``roll_reference_x_inches`` is supplied, the first reduced point is
    the pivot, matching the observed compatibility behavior.

    The generic policy estimates noise from a centered moving-average
    residual.  When ``noise_batch_points`` is set, it instead uses the median
    sample standard deviation of complete non-overlapping batches.  Noise is
    only classified when ``max_high_frequency_std_inches`` is not ``None``;
    the profile is retained and marked noisy rather than silently discarded.
    """

    crop_left_inches: float | None = None
    crop_right_inches: float | None = None
    invalid_policy: InvalidPolicy = "interpolate"
    max_interpolation_gap_inches: float | None = None
    minimum_valid_fraction: float = 0.5
    minimum_points: int = 3
    outlier_sigma_threshold: float | None = None
    smoothing_window_points: int = 1
    drop_final_smoothing_window: bool = False
    roll_degrees: float = 0.0
    roll_reference_x_inches: float | None = None
    noise_window_inches: float = 2.0
    noise_batch_points: int | None = None
    max_high_frequency_std_inches: float | None = None

    def __post_init__(self) -> None:
        if self.invalid_policy not in {"interpolate", "drop", "raise"}:
            raise ValueError(f"unsupported invalid_policy: {self.invalid_policy!r}")
        if (
            self.crop_left_inches is not None
            and self.crop_right_inches is not None
            and self.crop_right_inches <= self.crop_left_inches
        ):
            raise ValueError("crop_right_inches must be greater than crop_left_inches")
        if self.max_interpolation_gap_inches is not None and self.max_interpolation_gap_inches <= 0:
            raise ValueError("max_interpolation_gap_inches must be positive")
        if not 0.0 <= self.minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be between zero and one")
        if self.minimum_points < 2:
            raise ValueError("minimum_points must be at least two")
        if self.outlier_sigma_threshold is not None and self.outlier_sigma_threshold <= 0.0:
            raise ValueError("outlier_sigma_threshold must be positive")
        if self.smoothing_window_points <= 0 or self.smoothing_window_points % 2 == 0:
            raise ValueError("smoothing_window_points must be a positive odd integer")
        if self.noise_window_inches <= 0.0:
            raise ValueError("noise_window_inches must be positive")
        if self.noise_batch_points is not None and self.noise_batch_points < 2:
            raise ValueError("noise_batch_points must be at least two")
        if self.max_high_frequency_std_inches is not None and self.max_high_frequency_std_inches < 0.0:
            raise ValueError("max_high_frequency_std_inches cannot be negative")

    @classmethod
    def pathview_observed(
        cls,
        *,
        crop_left_inches: float | None = None,
        crop_right_inches: float | None = None,
        invalid_policy: InvalidPolicy = "interpolate",
        max_interpolation_gap_inches: float | None = None,
        minimum_valid_fraction: float = 0.5,
        minimum_points: int = 3,
        roll_degrees: float = 0.0,
        max_stddev_inches_high_noise: float | None = 999.0,
    ) -> ReductionConfig:
        """Build the observed PathView 7.2.04-compatible reduction policy.

        This factory fixes the independently measured 3.5-sigma filter,
        19-point coordinate mean, final-window omission, and 30-point batch
        noise statistic.  It represents the currently validated single-laser,
        empty-dark-band path; it must not be used to claim double-laser or
        non-empty dark-band compatibility.
        """

        return cls(
            crop_left_inches=crop_left_inches,
            crop_right_inches=crop_right_inches,
            invalid_policy=invalid_policy,
            max_interpolation_gap_inches=max_interpolation_gap_inches,
            minimum_valid_fraction=minimum_valid_fraction,
            minimum_points=minimum_points,
            outlier_sigma_threshold=3.5,
            smoothing_window_points=19,
            drop_final_smoothing_window=True,
            roll_degrees=roll_degrees,
            noise_batch_points=30,
            max_high_frequency_std_inches=max_stddev_inches_high_noise,
        )


def _interpolate_bounded_gaps(
    x: np.ndarray,
    elevation: np.ndarray,
    max_gap_inches: float | None,
) -> np.ndarray:
    """Interpolate internal gaps; leave unbounded/oversized gaps non-finite."""

    finite = np.isfinite(elevation)
    if finite.all():
        return elevation.copy()
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 2:
        return elevation.copy()

    result = elevation.copy()
    missing = ~finite
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
    for start, end in zip(starts, ends, strict=True):
        left_index = start - 1
        right_index = end + 1
        if left_index < 0 or right_index >= x.size:
            continue
        span = float(x[right_index] - x[left_index])
        if max_gap_inches is not None and span > max_gap_inches:
            continue
        result[start : end + 1] = np.interp(
            x[start : end + 1],
            x[[left_index, right_index]],
            elevation[[left_index, right_index]],
        )
    return result


def estimate_high_frequency_noise_std(
    x_inches: np.ndarray,
    elevation_inches: np.ndarray,
    window_inches: float,
) -> float:
    """Return RMS high-frequency residual in inches.

    The input may be irregularly sampled.  A median sample spacing is used to
    select an odd moving-average window, making the estimator deterministic and
    NumPy-only.  This is a documented local policy, not a vendor-equivalent
    noise metric.
    """

    if x_inches.size < 3:
        return 0.0
    spacing = float(np.median(np.diff(x_inches)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return float("nan")
    window_points = max(3, int(round(window_inches / spacing)))
    if window_points % 2 == 0:
        window_points += 1
    if window_points > x_inches.size:
        window_points = x_inches.size if x_inches.size % 2 == 1 else x_inches.size - 1
    if window_points < 3:
        return 0.0

    pad = window_points // 2
    padded = np.pad(elevation_inches, pad_width=pad, mode="reflect")
    kernel = np.full(window_points, 1.0 / window_points, dtype=np.float64)
    trend = np.convolve(padded, kernel, mode="valid")
    residual = elevation_inches - trend
    return float(np.sqrt(np.mean(np.square(residual))))


def estimate_batched_noise_std(
    elevation_inches: np.ndarray,
    batch_points: int,
) -> float:
    """Median sample standard deviation of complete non-overlapping batches."""

    complete_count = (elevation_inches.size // batch_points) * batch_points
    if complete_count == 0:
        return 0.0
    batches = elevation_inches[:complete_count].reshape(-1, batch_points)
    return float(np.median(np.std(batches, axis=1, ddof=1)))


def reduce_profile(
    profile: TransverseProfile,
    config: ReductionConfig | None = None,
) -> ReducedProfile:
    """Crop, repair invalid samples, correct roll, and classify noise.

    The returned coordinates always contain finite elevations and remain in
    increasing-x order.  Large gaps that are not eligible for interpolation are
    dropped; no extrapolation is performed at either profile edge.
    """

    config = config or ReductionConfig()
    x = np.asarray(profile.x_inches, dtype=np.float64)
    elevation = np.asarray(profile.elevation_inches, dtype=np.float64)

    crop_mask = np.ones(x.size, dtype=bool)
    if config.crop_left_inches is not None:
        crop_mask &= x >= config.crop_left_inches
    if config.crop_right_inches is not None:
        crop_mask &= x <= config.crop_right_inches
    x = x[crop_mask]
    elevation = elevation[crop_mask]
    if x.size < config.minimum_points:
        raise ValueError(f"only {x.size} profile points remain after cropping; minimum_points={config.minimum_points}")

    initially_finite = np.isfinite(elevation)
    valid_fraction = float(np.mean(initially_finite))
    if valid_fraction < config.minimum_valid_fraction:
        raise ValueError(
            f"valid elevation fraction {valid_fraction:.3f} is below "
            f"minimum_valid_fraction={config.minimum_valid_fraction:.3f}"
        )
    if config.invalid_policy == "raise" and not initially_finite.all():
        invalid_count = int(np.count_nonzero(~initially_finite))
        raise ValueError(f"profile contains {invalid_count} non-finite elevations")
    if config.invalid_policy == "interpolate":
        elevation = _interpolate_bounded_gaps(
            x,
            elevation,
            config.max_interpolation_gap_inches,
        )

    finite = np.isfinite(elevation)
    x = x[finite]
    elevation = elevation[finite]
    if x.size < config.minimum_points:
        raise ValueError(f"only {x.size} finite profile points remain; minimum_points={config.minimum_points}")

    noise_source_elevation = elevation.copy()
    notes: list[str] = []
    if config.outlier_sigma_threshold is not None and elevation.size >= 2:
        mean = float(np.mean(elevation))
        standard_deviation = float(np.std(elevation, ddof=1))
        if standard_deviation > 0.0:
            keep = np.abs(elevation - mean) <= config.outlier_sigma_threshold * standard_deviation
            x = x[keep]
            elevation = elevation[keep]
        notes.append(
            f"samples beyond {config.outlier_sigma_threshold:g} sample standard deviations "
            "from the global mean are removed before smoothing"
        )
        if x.size < config.minimum_points:
            raise ValueError(
                f"only {x.size} profile points remain after outlier removal; minimum_points={config.minimum_points}"
            )
    if config.smoothing_window_points > 1:
        window = config.smoothing_window_points
        if x.size < window:
            raise ValueError(f"only {x.size} finite profile points remain; smoothing_window_points={window}")
        half_window = window // 2
        kernel = np.full(window, 1.0 / window, dtype=np.float64)
        elevation = np.convolve(elevation, kernel, mode="valid")
        x = np.convolve(x, kernel, mode="valid")
        if config.drop_final_smoothing_window:
            x = x[:-1]
            elevation = elevation[:-1]
        notes.append(
            f"coordinates use a centered {window}-point valid box mean; "
            f"the first output is input center index {half_window}"
        )
        if config.drop_final_smoothing_window:
            notes.append("the final otherwise-valid smoothing window is omitted")
        if x.size < config.minimum_points:
            raise ValueError(
                f"only {x.size} profile points remain after smoothing; minimum_points={config.minimum_points}"
            )
    if config.roll_degrees != 0.0:
        reference_x = config.roll_reference_x_inches
        if reference_x is None:
            reference_x = float(x[0])
            reference_y = float(elevation[0])
        else:
            if reference_x < x[0] or reference_x > x[-1]:
                raise ValueError("roll_reference_x_inches must fall within the reduced profile")
            reference_y = float(np.interp(reference_x, x, elevation))
        angle = np.deg2rad(config.roll_degrees)
        delta_x = x - reference_x
        delta_y = elevation - reference_y
        x = reference_x + delta_x * np.cos(angle) + delta_y * np.sin(angle)
        elevation = reference_y - delta_x * np.sin(angle) + delta_y * np.cos(angle)
        notes.append(
            "roll correction is a rigid clockwise rotation about the first reduced point "
            "unless an explicit reference x is supplied"
        )
    if config.invalid_policy == "interpolate" and not initially_finite.all():
        notes.append("internal invalid values use linear interpolation; no edge extrapolation")
    elif config.invalid_policy == "drop" and not initially_finite.all():
        notes.append("invalid values are dropped without resampling")

    if config.noise_batch_points is None:
        noise_std = estimate_high_frequency_noise_std(
            x,
            elevation,
            config.noise_window_inches,
        )
    else:
        noise_std = estimate_batched_noise_std(
            noise_source_elevation,
            config.noise_batch_points,
        )
    is_noisy = config.max_high_frequency_std_inches is not None and noise_std > config.max_high_frequency_std_inches
    if config.max_high_frequency_std_inches is not None:
        if config.noise_batch_points is None:
            notes.append(
                "noise flag uses local moving-average residual RMS, not PathView-compatible 30-point batch statistics"
            )
        else:
            notes.append(
                f"noise flag uses the median sample standard deviation of complete "
                f"{config.noise_batch_points}-point batches"
            )

    return ReducedProfile(
        x_inches=x,
        elevation_inches=elevation,
        is_noisy=bool(is_noisy),
        original_point_count=profile.point_count,
        valid_fraction=valid_fraction,
        profile_id=profile.profile_id,
        compatibility_notes=tuple(notes),
    )
