import math

from pavement_rut.severity import finite_mean, severity_from_average


def test_severity_boundaries() -> None:
    assert severity_from_average(math.nan) == -1
    assert severity_from_average(0.249999) == 0
    assert severity_from_average(0.25) == 1
    assert severity_from_average(0.5) == 2
    assert severity_from_average(1.0) == 3


def test_finite_mean_ignores_non_finite_values() -> None:
    assert finite_mean([1.0, math.nan, math.inf, 3.0]) == 2.0
