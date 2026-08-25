import numpy as np
import pytest

import pavement_rut.domain.rutbar as rutbar_module
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


def _reference_upper_concave_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull: list[int] = []
    for index in range(x.size):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            first_term = (x[second] - x[first]) * (y[index] - y[second])
            second_term = (y[second] - y[first]) * (x[index] - x[second])
            cross = first_term - second_term
            scale = max(1.0, abs(first_term), abs(second_term))
            if cross >= -32.0 * np.finfo(np.float64).eps * scale:
                hull.pop()
            else:
                break
        hull.append(index)
    return np.asarray(hull, dtype=np.int64)


def test_public_geometry_constants_are_explicit() -> None:
    assert DEFAULT_RUT_BAR_LENGTH_INCHES == 72.0
    assert DEFAULT_RUT_PATH_WIDTH_INCHES == pytest.approx(44.29134)
    assert DEFAULT_RUT_PATH_HALF_WIDTH_INCHES == pytest.approx(22.14567)
    assert DEFAULT_WHEEL_PATH_CENTER_OFFSET_INCHES == pytest.approx(34.448819)
    assert DEFAULT_MEASUREMENT_WIDTH_INCHES == 4.0


@pytest.mark.parametrize("seed", range(8))
def test_optimized_upper_concave_hull_is_exactly_equivalent(seed: int) -> None:
    generator = np.random.default_rng(seed)
    x = np.cumsum(generator.uniform(0.05, 0.2, size=1536), dtype=np.float64)
    y = generator.normal(size=1536).astype(np.float64)

    expected = _reference_upper_concave_hull_indices(x, y)
    actual = rutbar_module._upper_concave_hull_indices(x, y)

    assert np.array_equal(actual, expected)


def test_rutbar_reuses_hull_for_an_unchanged_support_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = rutbar_module._upper_concave_hull_indices

    def counted(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(x, y)

    monkeypatch.setattr(rutbar_module, "_upper_concave_hull_indices", counted)
    elevation = 0.0002 * (X - LANE.center_inches) ** 2

    result = measure_profile_rutting(ReducedProfile(X, elevation), LANE)

    assert result.left is not None and result.right is not None
    assert calls == 2


def test_rutbar_reuses_a_supplied_successful_shoulder_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = ReducedProfile(X, 0.0002 * (X - LANE.center_inches) ** 2)
    trimmed = rutbar_module.remove_shoulders(profile, LANE)
    expected = measure_profile_rutting(profile, LANE)

    def unexpected_remove(*args: object, **kwargs: object) -> None:
        raise AssertionError("shoulder removal should not run again")

    monkeypatch.setattr(rutbar_module, "remove_shoulders", unexpected_remove)
    actual = measure_profile_rutting(profile, LANE, shoulder_removed_profile=trimmed)

    assert actual == expected


@pytest.mark.parametrize("seed", range(12))
def test_accelerated_footprint_gap_is_exactly_equivalent(seed: int) -> None:
    generator = np.random.default_rng(seed)
    x = np.cumsum(generator.uniform(0.08, 0.14, size=1536), dtype=np.float64)
    x -= x[0]
    y = np.cumsum(generator.normal(0.0, 0.01, size=1536), dtype=np.float64)
    kwargs = {
        "center_left": float(x[350]),
        "center_right": float(x[1150]),
        "measurement_width_inches": 4.0,
        "bar_slope": float(generator.normal(0.0, 0.02)),
        "bar_y_at_wheel_center": float(np.max(y) + 1.0),
        "wheel_center": float(x[750]),
    }

    expected = rutbar_module._maximum_footprint_gap_python(x, y, **kwargs)
    actual = rutbar_module._maximum_footprint_gap(x, y, **kwargs)

    assert actual == expected


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
