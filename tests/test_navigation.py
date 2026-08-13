import math
from pathlib import Path

from pavement_rut.navigation import NavigationLookup, interpolate_columns


def test_interpolate_columns() -> None:
    rows = [(10.0, 1.0, 3.0), (20.0, 3.0, 7.0)]
    assert interpolate_columns(rows, 15.0, (1, 2)) == (2.0, 5.0)
    assert all(math.isnan(value) for value in interpolate_columns(rows, 5.0, (1, 2)))


def test_navigation_missing_files_returns_nan(tmp_path: Path) -> None:
    set_dir = tmp_path / "112"
    set_dir.mkdir()
    point = NavigationLookup(set_dir)(100.0)
    assert math.isnan(point.latitude)
    assert math.isnan(point.longitude)
    assert math.isnan(point.heading)
