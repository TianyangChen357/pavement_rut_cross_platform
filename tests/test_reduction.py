import numpy as np
import pytest

from pavement_rut.domain.models import TransverseProfile
from pavement_rut.domain.reduction import ReductionConfig, reduce_profile


def test_reduction_crops_and_interpolates_only_internal_gaps() -> None:
    raw = TransverseProfile(
        np.arange(7.0),
        [np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
    )
    reduced = reduce_profile(
        raw,
        ReductionConfig(
            crop_left_inches=1.0,
            crop_right_inches=6.0,
            minimum_valid_fraction=0.5,
        ),
    )

    np.testing.assert_allclose(reduced.x_inches, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(reduced.elevation_inches, [1, 2, 3, 4, 5])
    assert reduced.valid_fraction == pytest.approx(4 / 6)
    assert "no edge extrapolation" in reduced.compatibility_notes[0]


def test_large_invalid_gap_is_dropped_instead_of_interpolated() -> None:
    raw = TransverseProfile(np.arange(6.0), [0.0, 1.0, np.nan, np.nan, 4.0, 5.0])
    reduced = reduce_profile(
        raw,
        ReductionConfig(max_interpolation_gap_inches=2.0, minimum_valid_fraction=0.5),
    )

    np.testing.assert_allclose(reduced.x_inches, [0, 1, 4, 5])
    np.testing.assert_allclose(reduced.elevation_inches, [0, 1, 4, 5])


def test_roll_correction_removes_configured_linear_tilt() -> None:
    x = np.linspace(0.0, 10.0, 101)
    elevation = 2.5 + np.tan(np.deg2rad(2.0)) * (x - 5.0)
    reduced = reduce_profile(
        TransverseProfile(x, elevation),
        ReductionConfig(roll_degrees=2.0, roll_reference_x_inches=5.0),
    )

    np.testing.assert_allclose(reduced.elevation_inches, 2.5, atol=1e-12)
    np.testing.assert_allclose(
        reduced.x_inches,
        5.0 + (x - 5.0) / np.cos(np.deg2rad(2.0)),
        atol=1e-12,
    )
    assert "rigid clockwise rotation" in reduced.compatibility_notes[0]


def test_centered_box_smoothing_matches_valid_nineteen_point_mean() -> None:
    x = np.arange(30.0)
    elevation = np.square(x)
    reduced = reduce_profile(
        TransverseProfile(x, elevation),
        ReductionConfig(smoothing_window_points=19),
    )

    expected = np.convolve(elevation, np.full(19, 1.0 / 19.0), mode="valid")
    np.testing.assert_allclose(reduced.x_inches, x[9:-9])
    np.testing.assert_allclose(reduced.elevation_inches, expected, rtol=0.0, atol=1e-12)
    assert "center index 9" in reduced.compatibility_notes[0]


def test_default_reduction_does_not_smooth() -> None:
    x = np.arange(5.0)
    elevation = np.array([0.0, 2.0, -1.0, 3.0, 1.0])

    reduced = reduce_profile(TransverseProfile(x, elevation))

    np.testing.assert_array_equal(reduced.x_inches, x)
    np.testing.assert_array_equal(reduced.elevation_inches, elevation)


def test_compatibility_outlier_filter_smooths_both_axes_and_drops_final_window() -> None:
    x = np.arange(30.0)
    elevation = 0.1 * x
    elevation[10] = 100.0
    config = ReductionConfig(
        outlier_sigma_threshold=2.0,
        smoothing_window_points=5,
        drop_final_smoothing_window=True,
    )
    reduced = reduce_profile(TransverseProfile(x, elevation), config)

    keep = np.abs(elevation - np.mean(elevation)) <= 2.0 * np.std(elevation, ddof=1)
    kernel = np.full(5, 0.2)
    expected_x = np.convolve(x[keep], kernel, mode="valid")[:-1]
    expected_y = np.convolve(elevation[keep], kernel, mode="valid")[:-1]
    np.testing.assert_allclose(reduced.x_inches, expected_x, atol=1e-12)
    np.testing.assert_allclose(reduced.elevation_inches, expected_y, atol=1e-12)
    assert reduced.point_count == np.count_nonzero(keep) - 5


def test_batched_noise_policy_uses_complete_batches_and_sample_std() -> None:
    x = np.arange(95.0)
    alternating = np.where(np.arange(30) % 2 == 0, 1.0, -1.0)
    elevation = np.concatenate((0.1 * alternating, 0.2 * alternating, 0.3 * alternating, np.full(5, 99.0)))
    reduced = reduce_profile(
        TransverseProfile(x, elevation),
        ReductionConfig(
            noise_batch_points=30,
            max_high_frequency_std_inches=0.15,
        ),
    )

    expected = np.median(np.std(elevation[:90].reshape(3, 30), axis=1, ddof=1))
    assert expected > 0.15
    assert reduced.is_noisy
    assert "complete 30-point batches" in reduced.compatibility_notes[0]


@pytest.mark.parametrize("window", [0, 2, 4])
def test_smoothing_window_must_be_positive_and_odd(window: int) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        ReductionConfig(smoothing_window_points=window)


def test_outlier_sigma_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="outlier_sigma_threshold"):
        ReductionConfig(outlier_sigma_threshold=0.0)


def test_pathview_observed_factory_locks_compatibility_policy() -> None:
    config = ReductionConfig.pathview_observed()

    assert config.outlier_sigma_threshold == 3.5
    assert config.smoothing_window_points == 19
    assert config.drop_final_smoothing_window is True
    assert config.noise_batch_points == 30
    assert config.max_high_frequency_std_inches == 999.0
    assert config.roll_degrees == 0.0


def test_noise_policy_marks_but_retains_profile() -> None:
    x = np.linspace(0.0, 20.0, 201)
    elevation = 0.1 * np.where(np.arange(x.size) % 2 == 0, 1.0, -1.0)
    reduced = reduce_profile(
        TransverseProfile(x, elevation),
        ReductionConfig(
            noise_window_inches=1.0,
            max_high_frequency_std_inches=0.05,
        ),
    )

    assert reduced.is_noisy
    assert reduced.point_count == x.size
    assert "not PathView" in reduced.compatibility_notes[0]


def test_reduction_enforces_minimum_valid_fraction() -> None:
    raw = TransverseProfile([0, 1, 2, 3], [np.nan, np.nan, np.nan, 1.0])
    with pytest.raises(ValueError, match="valid elevation fraction"):
        reduce_profile(raw, ReductionConfig(minimum_valid_fraction=0.5))
