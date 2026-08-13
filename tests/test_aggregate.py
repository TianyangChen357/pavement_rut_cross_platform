from dataclasses import replace

import numpy as np
import pytest

from pavement_rut.domain.aggregate import aggregate_rutting, severity_from_rut_depth
from pavement_rut.domain.models import LaneGeometry, ReducedProfile
from pavement_rut.domain.rutbar import measure_profile_rutting


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(0.0, 0), (0.2499, 0), (0.25, 1), (0.5, 2), (1.0, 3), (np.nan, -1)],
)
def test_severity_boundaries(depth: float, expected: int) -> None:
    assert severity_from_rut_depth(depth) == expected


def test_aggregate_averages_sides_before_overall() -> None:
    x = np.arange(0.0, 162.0001, 0.125)
    lane = LaneGeometry(0.0, 162.0)

    def result(left_depth: float, right_depth: float):
        y = -left_depth * np.exp(-0.5 * ((x - lane.wheel_path_center("left")) / 6.0) ** 2) - right_depth * np.exp(
            -0.5 * ((x - lane.wheel_path_center("right")) / 6.0) ** 2
        )
        return measure_profile_rutting(ReducedProfile(x, y), lane)

    aggregate = aggregate_rutting([result(0.2, 0.4), result(0.4, 0.8), None])

    assert aggregate.left_average_inches == pytest.approx(0.29452553, abs=1e-8)
    assert aggregate.right_average_inches == pytest.approx(0.58905109, abs=1e-8)
    assert aggregate.overall_average_inches == pytest.approx(0.44178831, abs=1e-8)
    assert aggregate.severity == 1
    assert aggregate.profiles_total == 3
    assert aggregate.profiles_with_any_result == 2
    assert aggregate.left_count == aggregate.right_count == 2


def test_aggregate_requires_both_sides_for_compatible_overall() -> None:
    x = np.arange(0.0, 162.0001, 0.125)
    lane = LaneGeometry(0.0, 162.0)
    complete = measure_profile_rutting(ReducedProfile(x, np.zeros_like(x)), lane)
    left_only = replace(complete, right=None)

    compatible = aggregate_rutting([left_only])
    available_side = aggregate_rutting([left_only], require_both_sides=False)

    assert np.isnan(compatible.overall_average_inches)
    assert compatible.severity == -1
    assert available_side.overall_average_inches == pytest.approx(0.0)
    assert available_side.severity == 0
