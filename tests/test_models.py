import numpy as np
import pytest

from pavement_rut.domain.models import LaneGeometry, ReducedProfile, TransverseProfile


def test_profile_arrays_are_normalized_and_read_only() -> None:
    profile = TransverseProfile([0, 1, 2], [0, np.nan, 1])

    assert profile.x_inches.dtype == np.float64
    assert profile.point_count == 3
    assert not profile.x_inches.flags.writeable
    with pytest.raises(ValueError):
        profile.x_inches[0] = 10.0


def test_reduced_profile_rejects_nonfinite_elevations() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ReducedProfile([0, 1], [0, np.nan])


def test_profile_requires_strictly_increasing_x() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TransverseProfile([0, 1, 1], [0, 0, 0])


def test_lane_geometry_defines_explicit_left_and_right_wheel_zones() -> None:
    lane = LaneGeometry(0.0, 162.0)

    assert lane.center_inches == pytest.approx(81.0)
    assert lane.wheel_path_center("left") == pytest.approx(46.551181)
    assert lane.wheel_path_center("right") == pytest.approx(115.448819)
    assert lane.wheel_path_bounds("left") == pytest.approx((24.405511, 68.696851))
    assert lane.wheel_path_bounds("right") == pytest.approx((93.303149, 137.594489))
