from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from pavement_rut.io.calibration import CalibrationError, load_calibration, loads_calibration


def _calibration_text(*, extras: str = "772 3.1", width: str = "40") -> str:
    offsets = "-1.5 0 2.5 10"
    metadata = f"3 1 4 {width} 0.03 0.03 4 0.125 2020 2 29 14 57"
    return f"3D_Camera_Calibration\n{offsets} {metadata} {extras}".rstrip()


def test_parses_calibration_and_applies_vectorized_formula() -> None:
    calibration = loads_calibration(_calibration_text(), columns=4)

    assert calibration.columns == 4
    assert calibration.pavement_width_inches == 40.0
    assert calibration.pixel_width_inches == 10.0
    assert calibration.recorded_height_resolution_inches == 0.125
    assert calibration.height_resolution_inches == 0.78125
    assert calibration.calibrated_at == datetime(2020, 2, 29, 14, 57)
    assert calibration.additional_parameters == (772.0, 3.1)
    np.testing.assert_array_equal(calibration.x_inches(), [0.0, 10.0, 20.0, 30.0])

    raw = np.array([[2, 2, 2, 2], [10, 20, 30, 40]], dtype=np.uint16)
    expected = (raw.astype(np.float64) - np.array([-1.5, 0.0, 2.5, 10.0])) * calibration.height_resolution_inches
    np.testing.assert_allclose(calibration.apply(raw), expected, rtol=0.0, atol=0.0)
    assert calibration.height_offsets.flags.writeable is False


def test_supports_legacy_record_without_optional_values() -> None:
    calibration = loads_calibration(_calibration_text(extras=""), columns=4)

    assert calibration.additional_parameters == ()


def test_loads_utf8_bom_file(tmp_path) -> None:
    path = tmp_path / "3D_Camera.cal"
    path.write_text("\ufeff" + _calibration_text(), encoding="utf-8")

    calibration = load_calibration(path, columns=4)

    assert calibration.camera_height_inches == 4.0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Wrong_Header 1 2 3", "unexpected calibration header"),
        ("3D_Camera_Calibration 1 2", "expected at least"),
        (_calibration_text().replace("2020 2 29", "2019 2 29"), "invalid calibration date/time"),
        (_calibration_text(width="-1"), "pavement_width_inches must be positive"),
    ],
)
def test_rejects_invalid_calibration(text: str, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        loads_calibration(text, columns=4)


def test_apply_requires_matching_numeric_last_axis() -> None:
    calibration = loads_calibration(_calibration_text(), columns=4)

    with pytest.raises(ValueError, match="must have 4 columns"):
        calibration.apply(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="must be numeric"):
        calibration.apply(np.array([["a", "b", "c", "d"]]))
