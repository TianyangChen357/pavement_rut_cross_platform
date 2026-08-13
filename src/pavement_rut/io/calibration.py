"""Parser and vectorized application of Pathway ``3D_Camera.cal`` files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

CALIBRATION_HEADER: Final = "3D_Camera_Calibration"
DEFAULT_COLUMNS: Final = 1536
_REQUIRED_TRAILING_VALUES: Final = 13
_RAW_HEIGHT_SUBPIXEL_SCALE: Final = 16.0


class CalibrationError(ValueError):
    """Raised when a camera calibration record is incomplete or invalid."""


def _finite_float(token: str, *, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise CalibrationError(f"invalid {field}: {token!r}") from exc
    if not np.isfinite(value):
        raise CalibrationError(f"{field} must be finite, got {token!r}")
    return value


def _integer(token: str, *, field: str) -> int:
    value = _finite_float(token, field=field)
    integer = int(value)
    if float(integer) != value:
        raise CalibrationError(f"{field} must be an integer, got {token!r}")
    return integer


@dataclass(frozen=True, slots=True, eq=False)
class CameraCalibration:
    """The calibration values required to turn raw samples into inches.

    ``height_offsets`` follows the public/PathView-compatible left-to-right
    column order.  The two unnamed calibration terms and optional trailing
    values are retained losslessly as numbers even though this processing path
    does not need to interpret them.
    """

    height_offsets: NDArray[np.float64]
    camera_horizontal_inches_to_line: float
    laser_horizontal_inches_to_line: float
    camera_height_inches: float
    pavement_width_inches: float
    calibration_terms: tuple[float, float]
    laser_height_inches: float
    recorded_height_resolution_inches: float
    calibrated_at: datetime
    additional_parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        offsets = np.asarray(self.height_offsets, dtype=np.float64)
        if offsets.ndim != 1 or offsets.size == 0:
            raise CalibrationError("height_offsets must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(offsets)):
            raise CalibrationError("height_offsets contains a non-finite value")
        offsets = np.array(offsets, dtype=np.float64, copy=True)
        offsets.setflags(write=False)
        object.__setattr__(self, "height_offsets", offsets)

        positive_fields = {
            "camera_height_inches": self.camera_height_inches,
            "pavement_width_inches": self.pavement_width_inches,
            "laser_height_inches": self.laser_height_inches,
            "recorded_height_resolution_inches": self.recorded_height_resolution_inches,
        }
        for name, value in positive_fields.items():
            if not np.isfinite(value) or value <= 0:
                raise CalibrationError(f"{name} must be positive and finite, got {value!r}")
        for name, value in {
            "camera_horizontal_inches_to_line": self.camera_horizontal_inches_to_line,
            "laser_horizontal_inches_to_line": self.laser_horizontal_inches_to_line,
            "calibration term 1": self.calibration_terms[0],
            "calibration term 2": self.calibration_terms[1],
            **{
                f"additional calibration parameter {index}": item
                for index, item in enumerate(self.additional_parameters, start=1)
            },
        }.items():
            if not np.isfinite(value):
                raise CalibrationError(f"{name} must be finite, got {value!r}")
        if self.height_resolution_inches <= 0:
            raise CalibrationError("camera/laser geometry produces a non-positive height resolution")

    @property
    def columns(self) -> int:
        """Number of transverse samples represented by the calibration."""

        return int(self.height_offsets.size)

    @property
    def pixel_width_inches(self) -> float:
        """Uniform spacing of the raw lateral grid in inches."""

        return self.pavement_width_inches / self.columns

    @property
    def height_resolution_inches(self) -> float:
        """Height represented by one raw integer step, in inches.

        The calibration record includes a rounded copy of this value, retained
        as :attr:`recorded_height_resolution_inches`.  The reference reader
        derives its full-precision value from the camera/laser triangulation
        geometry, the lateral pixel width, and the 1/16-pixel raw encoding.
        """

        denominator = _RAW_HEIGHT_SUBPIXEL_SCALE * (
            self.camera_horizontal_inches_to_line * self.laser_height_inches
            + self.laser_horizontal_inches_to_line * self.camera_height_inches
        )
        if denominator == 0:
            raise CalibrationError("camera/laser angle is invalid")
        camera_slant_range = hypot(
            self.camera_horizontal_inches_to_line,
            self.camera_height_inches,
        )
        return (self.pixel_width_inches * camera_slant_range * self.laser_height_inches) / denominator

    def x_inches(self) -> NDArray[np.float64]:
        """Return the raw lateral grid ``0, pixel_width, ...`` in inches.

        Cropping, invalid-point removal, roll correction, and other profile
        reduction steps belong to the domain layer and are intentionally not
        applied here.
        """

        return np.arange(self.columns, dtype=np.float64) * self.pixel_width_inches

    def apply(self, raw_heights: ArrayLike) -> NDArray[np.float64]:
        """Calibrate one profile or a stack of profiles to height inches.

        The last input dimension must equal :attr:`columns`.  Calibration is
        vectorized as ``(raw - offset[column]) * height_resolution``.
        """

        raw = np.asarray(raw_heights)
        if raw.ndim == 0 or raw.shape[-1] != self.columns:
            actual = "scalar" if raw.ndim == 0 else str(raw.shape[-1])
            raise ValueError(f"raw height data must have {self.columns} columns on its last axis; got {actual}")
        if not np.issubdtype(raw.dtype, np.number):
            raise TypeError(f"raw height data must be numeric, got dtype {raw.dtype}")
        return (raw.astype(np.float64, copy=False) - self.height_offsets) * self.height_resolution_inches


def loads_calibration(text: str, *, columns: int = DEFAULT_COLUMNS) -> CameraCalibration:
    """Parse one ``3D_Camera_Calibration`` text record."""

    if columns <= 0:
        raise ValueError("columns must be positive")
    tokens = text.lstrip("\ufeff").split()
    if not tokens:
        raise CalibrationError("calibration file is empty")
    if tokens[0] != CALIBRATION_HEADER:
        raise CalibrationError(f"unexpected calibration header {tokens[0]!r}; expected {CALIBRATION_HEADER!r}")

    values = tokens[1:]
    minimum_values = columns + _REQUIRED_TRAILING_VALUES
    if len(values) < minimum_values:
        raise CalibrationError(
            f"calibration record has {len(values)} numeric values; expected at least {minimum_values} "
            f"for {columns} columns"
        )

    offsets = np.fromiter(
        (_finite_float(token, field=f"height offset {index}") for index, token in enumerate(values[:columns])),
        dtype=np.float64,
        count=columns,
    )
    tail = values[columns:]

    camera_horizontal = _finite_float(tail[0], field="camera horizontal distance")
    laser_horizontal = _finite_float(tail[1], field="laser horizontal distance")
    camera_height = _finite_float(tail[2], field="camera height")
    pavement_width = _finite_float(tail[3], field="pavement width")
    calibration_terms = (
        _finite_float(tail[4], field="calibration term 1"),
        _finite_float(tail[5], field="calibration term 2"),
    )
    laser_height = _finite_float(tail[6], field="laser height")
    height_resolution = _finite_float(tail[7], field="height resolution")

    year = _integer(tail[8], field="calibration year")
    month = _integer(tail[9], field="calibration month")
    day = _integer(tail[10], field="calibration day")
    hour = _integer(tail[11], field="calibration hour")
    minute = _integer(tail[12], field="calibration minute")
    try:
        calibrated_at = datetime(year, month, day, hour, minute)
    except ValueError as exc:
        raise CalibrationError(f"invalid calibration date/time: {exc}") from exc

    additional_parameters = tuple(
        _finite_float(token, field=f"additional calibration parameter {index}")
        for index, token in enumerate(tail[_REQUIRED_TRAILING_VALUES:], start=1)
    )

    return CameraCalibration(
        height_offsets=offsets,
        camera_horizontal_inches_to_line=camera_horizontal,
        laser_horizontal_inches_to_line=laser_horizontal,
        camera_height_inches=camera_height,
        pavement_width_inches=pavement_width,
        calibration_terms=calibration_terms,
        laser_height_inches=laser_height,
        recorded_height_resolution_inches=height_resolution,
        calibrated_at=calibrated_at,
        additional_parameters=additional_parameters,
    )


def load_calibration(
    path: str | Path,
    *,
    columns: int = DEFAULT_COLUMNS,
) -> CameraCalibration:
    """Load a camera calibration file from disk."""

    calibration_path = Path(path)
    try:
        text = calibration_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CalibrationError(f"calibration file is not UTF-8 text: {calibration_path}") from exc
    return loads_calibration(text, columns=columns)


__all__ = [
    "CALIBRATION_HEADER",
    "DEFAULT_COLUMNS",
    "CalibrationError",
    "CameraCalibration",
    "load_calibration",
    "loads_calibration",
]
