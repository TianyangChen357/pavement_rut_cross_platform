import numpy as np
import pytest

from pavement_rut.domain.cross_slope import fit_cross_slope
from pavement_rut.domain.models import LaneGeometry, ReducedProfile


def test_flat_profile_has_zero_cross_slope() -> None:
    x = np.linspace(0.0, 162.0, 1297)
    result = fit_cross_slope(ReducedProfile(x, np.zeros_like(x)), LaneGeometry(0.0, 162.0))

    assert result.rise_per_run == pytest.approx(0.0, abs=1e-15)
    assert result.percent == pytest.approx(0.0, abs=1e-15)
    assert result.angle_degrees == pytest.approx(0.0, abs=1e-15)
    assert result.r_squared == 1.0


def test_cross_slope_uses_pathview_half_lane_means() -> None:
    x = np.linspace(0.0, 162.0, 1297)
    elevation = 7.0 + 0.02 * x
    result = fit_cross_slope(ReducedProfile(x, elevation), LaneGeometry(0.0, 162.0))

    assert result.rise_per_run == pytest.approx(-0.02, abs=1e-14)
    assert result.percent == pytest.approx(-2.0, abs=1e-12)
    assert result.angle_degrees == pytest.approx(-1.145762838, abs=1e-9)
    assert result.intercept_inches == pytest.approx(7.0, abs=1e-12)
    assert result.r_squared == pytest.approx(1.0)


def test_cross_slope_respects_explicit_lane_bounds() -> None:
    x = np.linspace(0.0, 20.0, 201)
    elevation = np.where(x < 5.0, 10.0, 0.01 * x)
    result = fit_cross_slope(
        ReducedProfile(x, elevation),
        lane_left_inches=5.0,
        lane_right_inches=20.0,
    )

    assert result.rise_per_run == pytest.approx(-0.01, abs=1e-12)
    assert result.percent == pytest.approx(-1.0, abs=1e-12)


def test_lane_center_point_is_included_in_both_means() -> None:
    profile = ReducedProfile(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 9.0, 0.0]),
    )
    result = fit_cross_slope(
        profile,
        LaneGeometry(0.0, 4.0),
        remove_lane_shoulders=False,
    )

    # left mean = (0 + 9) / 2; right mean = (9 + 0) / 2
    assert result.rise_per_run == pytest.approx(0.0, abs=1e-15)
    assert result.point_count == 3


def test_reporting_sign_can_be_reversed() -> None:
    x = np.linspace(0.0, 20.0, 201)
    elevation = 0.01 * x
    result = fit_cross_slope(
        ReducedProfile(x, elevation),
        LaneGeometry(0.0, 20.0),
        positive_downward=False,
    )

    assert result.rise_per_run == pytest.approx(0.01, abs=1e-12)
    assert result.percent == pytest.approx(1.0, abs=1e-12)


def test_single_lane_center_point_returns_nan_like_pathview() -> None:
    result = fit_cross_slope(
        ReducedProfile(np.asarray([2.0]), np.asarray([9.0])),
        LaneGeometry(0.0, 4.0),
    )

    assert np.isnan(result.rise_per_run)
    assert np.isnan(result.percent)
    assert np.isnan(result.angle_degrees)
    assert result.point_count == 1
