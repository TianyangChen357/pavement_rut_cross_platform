import numpy as np
import pytest

from pavement_rut.domain.models import LaneGeometry, ReducedProfile
from pavement_rut.domain.rutbar import (
    DEFAULT_MEASUREMENT_WIDTH_INCHES,
    DEFAULT_RUT_BAR_LENGTH_INCHES,
    DEFAULT_RUT_PATH_HALF_WIDTH_INCHES,
    DEFAULT_RUT_PATH_WIDTH_INCHES,
    DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES,
    measure_profile_rutting,
)

LANE = LaneGeometry(0.0, 162.0)
X = np.arange(0.0, 162.0001, 0.125)


def test_public_geometry_constants_are_explicit() -> None:
    assert DEFAULT_RUT_BAR_LENGTH_INCHES == 72.0
    assert DEFAULT_RUT_PATH_WIDTH_INCHES == pytest.approx(44.29134)
    assert DEFAULT_RUT_PATH_HALF_WIDTH_INCHES == pytest.approx(22.14567)
    assert DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES == pytest.approx(34.448819)
    assert DEFAULT_MEASUREMENT_WIDTH_INCHES == 4.0


@pytest.mark.parametrize("slope", [0.0, 0.02, -0.035])
def test_flat_or_uniformly_sloped_plane_has_zero_rut(slope: float) -> None:
    elevation = slope * (X - LANE.center_inches)
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None
    assert result.right is not None
    assert result.left.rut_depth_inches == pytest.approx(0.0, abs=1e-12)
    assert result.right.rut_depth_inches == pytest.approx(0.0, abs=1e-12)
    assert result.overall_rut_depth_inches == pytest.approx(0.0, abs=1e-12)


def test_dual_gaussian_valleys_use_four_inch_average_footprint() -> None:
    left_center = LANE.wheel_path_center("left")
    right_center = LANE.wheel_path_center("right")
    elevation = -0.5 * np.exp(-0.5 * ((X - left_center) / 6.0) ** 2) - 0.75 * np.exp(
        -0.5 * ((X - right_center) / 6.0) ** 2
    )
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None
    assert result.right is not None
    # Analytic continuous-Gaussian footprint factors are 0.490893 and
    # 0.736340; the small residual is piecewise-linear sampling error.
    assert result.left.rut_depth_inches == pytest.approx(0.490893, abs=2e-5)
    assert result.right.rut_depth_inches == pytest.approx(0.736340, abs=3e-5)
    assert result.left.rut_x_inches == pytest.approx(left_center, abs=2e-5)
    assert result.right.rut_x_inches == pytest.approx(right_center, abs=2e-5)
    assert result.overall_rut_depth_inches == pytest.approx(0.613616, abs=3e-5)
    assert result.left.measurement_width_inches == 4.0


def test_parabola_uses_supported_chord_and_perpendicular_gap() -> None:
    elevation = 0.0002 * (X - LANE.center_inches) ** 2
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None
    assert result.right is not None
    assert result.left.rut_depth_inches == pytest.approx(0.2473403452, abs=1e-9)
    assert result.right.rut_depth_inches == pytest.approx(0.2473403452, abs=1e-9)
    assert result.left.left_contact_x_inches < LANE.wheel_path_center("left")
    assert result.left.right_contact_x_inches > LANE.wheel_path_center("left")


def test_depression_outside_wheel_path_is_not_reported_as_rut() -> None:
    elevation = -0.8 * np.exp(-0.5 * ((X - 5.0) / 1.0) ** 2)
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None
    assert result.right is not None
    assert result.left.rut_depth_inches == pytest.approx(0.0, abs=1e-12)
    assert result.right.rut_depth_inches == pytest.approx(0.0, abs=1e-12)


def test_depression_beyond_support_contact_is_not_measured_under_bar_overhang() -> None:
    elevation = np.zeros_like(X)
    elevation[(X >= 100.0) & (X <= 120.0)] = 1.0
    elevation -= 2.0 * np.exp(-0.5 * ((X - 135.0) / 0.5) ** 2)
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.right is not None
    assert result.right.left_contact_x_inches == pytest.approx(100.0)
    assert result.right.right_contact_x_inches == pytest.approx(120.0)
    assert result.right.rut_depth_inches == pytest.approx(0.0, abs=1e-12)


def test_complete_footprint_must_remain_inside_wheel_path_zone() -> None:
    zone_left, _ = LANE.wheel_path_bounds("left")
    elevation = -np.exp(-0.5 * ((X - zone_left) / 0.5) ** 2)
    # This unit isolates the wheel-zone footprint boundary. The synthetic
    # edge feature deliberately resembles a shoulder, so bypass the separately
    # tested PathView-compatible shoulder-removal stage here.
    result = measure_profile_rutting(
        ReducedProfile(X, elevation),
        LANE,
        remove_lane_shoulders=False,
    )

    assert result.left is not None
    assert result.left.rut_x_inches == pytest.approx(zone_left + 2.0, abs=1e-12)
    assert result.left.rut_x_inches - result.left.measurement_width_inches / 2.0 >= zone_left


def test_rut_point_elevation_is_piecewise_linear_footprint_average() -> None:
    elevation = 0.0002 * (X - LANE.center_inches) ** 2
    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None
    half_width = result.left.measurement_width_inches / 2.0
    dense_x = np.linspace(
        result.left.rut_x_inches - half_width,
        result.left.rut_x_inches + half_width,
        20_001,
    )
    dense_y = np.interp(dense_x, X, elevation)
    expected_average = np.trapezoid(dense_y, dense_x) / result.left.measurement_width_inches
    assert result.left.rut_elevation_inches == pytest.approx(expected_average, abs=1e-9)


def test_explicit_lane_edges_are_an_alternative_to_lane_geometry() -> None:
    result = measure_profile_rutting(
        ReducedProfile(X, np.zeros_like(X)),
        lane_left_inches=0.0,
        lane_right_inches=162.0,
    )

    assert result.left is not None and result.right is not None
