#!/usr/bin/env python3
"""Compare the portable implementation with a real PathView oracle export.

The companion NPZ is treated as untrusted input: it is resolved beside the
manifest, opened with ``allow_pickle=False``, and fully validated before any
profile is processed.  The tool never needs PathView or .NET.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pavement_rut.domain import (
    LaneGeometry,
    ReductionConfig,
    TransverseProfile,
    fit_cross_slope,
    measure_profile_rutting,
    reduce_profile,
)
from pavement_rut.domain.models import WheelPathRut

SCHEMA_VERSION = "1.0.0"
REQUIRED_ARRAYS = frozenset(
    {
        "profile_indices",
        "raw_offsets",
        "raw_height_u16",
        "calibrated_height_inches",
        "intensity_u8",
        "reduced_offsets",
        "reduced_x_inches",
        "reduced_y_inches",
    }
)
POINT_NAMES = (
    "left_reference_point",
    "right_reference_point",
    "left_contact_point",
    "right_contact_point",
    "measurement_point",
    "rut_point",
)


class GoldenValidationError(ValueError):
    """Raised when a manifest or companion NPZ violates the oracle schema."""


@dataclass(frozen=True)
class Tolerances:
    """Default thresholds documented in ``docs/compatibility.md``."""

    reduced_inches: float = 1e-6
    cross_slope_percent: float = 1e-4
    cross_slope_angle_degrees: float = 1e-4
    rut_depth_inches: float = 1e-3
    geometry_inches: float = 1e-2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GoldenInputs:
    metadata_path: Path
    npz_path: Path
    manifest: Mapping[str, Any]
    arrays: Mapping[str, NDArray[Any]]
    pixel_width_inches: float
    lane_geometry: LaneGeometry
    reduction_config: ReductionConfig
    bar_length_inches: float


def _reject_constant(value: str) -> None:
    raise GoldenValidationError(f"non-standard JSON numeric constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GoldenValidationError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoldenValidationError(f"cannot read strict JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoldenValidationError("manifest root must be a JSON object")
    return payload


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GoldenValidationError(f"{location} must be an object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise GoldenValidationError(f"{location} must be an array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise GoldenValidationError(f"{location} must be a string")
    return value


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise GoldenValidationError(f"{location} must be a boolean")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoldenValidationError(f"{location} must be an integer")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoldenValidationError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise GoldenValidationError(f"{location} must be finite")
    return result


def _member(parent: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in parent:
        raise GoldenValidationError(f"missing {location}.{key}")
    return parent[key]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GoldenValidationError(f"cannot read companion NPZ {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_companion(metadata_path: Path, arrays_record: Mapping[str, Any]) -> Path:
    file_name = _string(_member(arrays_record, "file_name", "arrays"), "arrays.file_name")
    windows_path = PureWindowsPath(file_name)
    posix_path = PurePosixPath(file_name)
    if (
        not file_name
        or file_name in {".", ".."}
        or "/" in file_name
        or "\\" in file_name
        or windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or posix_path.name != file_name
        or not file_name.lower().endswith(".npz")
    ):
        raise GoldenValidationError("arrays.file_name must be a safe relative NPZ basename")

    parent = metadata_path.parent.resolve()
    candidate = parent / file_name
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise GoldenValidationError(f"companion NPZ not found beside manifest: {file_name}") from exc
    if resolved.parent != parent or not resolved.is_file():
        raise GoldenValidationError("companion NPZ must be a regular file beside the manifest")
    return resolved


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise GoldenValidationError("companion NPZ contains duplicate array names")
            missing = sorted(REQUIRED_ARRAYS - set(archive.files))
            if missing:
                raise GoldenValidationError(f"companion NPZ is missing arrays: {', '.join(missing)}")
            arrays: dict[str, NDArray[Any]] = {}
            for name in archive.files:
                try:
                    array = np.asarray(archive[name])
                except ValueError as exc:
                    raise GoldenValidationError(f"NPZ array {name!r} cannot be loaded with allow_pickle=False") from exc
                if array.dtype.hasobject:
                    raise GoldenValidationError(f"NPZ array {name!r} has a forbidden object dtype")
                arrays[name] = array
    except GoldenValidationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise GoldenValidationError(f"invalid companion NPZ {path}: {exc}") from exc
    return arrays


def _validate_1d(
    array: NDArray[Any],
    name: str,
    kind: str | None = None,
    dtype: np.dtype[Any] | None = None,
) -> None:
    if array.ndim != 1:
        raise GoldenValidationError(f"NPZ array {name!r} must be one-dimensional")
    if kind == "integer" and array.dtype.kind not in "iu":
        raise GoldenValidationError(f"NPZ array {name!r} must have an integer dtype")
    if kind == "float" and array.dtype.kind != "f":
        raise GoldenValidationError(f"NPZ array {name!r} must have a floating dtype")
    if dtype is not None and array.dtype != dtype:
        raise GoldenValidationError(f"NPZ array {name!r} must have dtype {dtype}, got {array.dtype}")


def _validate_offsets(
    offsets: NDArray[Any],
    name: str,
    profile_count: int,
    terminal_sizes: Mapping[str, int],
) -> None:
    _validate_1d(offsets, name, "integer")
    if offsets.size != profile_count + 1:
        raise GoldenValidationError(f"{name} must contain profile_count + 1 entries")
    if offsets.dtype.kind == "u" and np.any(offsets > np.iinfo(np.int64).max):
        raise GoldenValidationError(f"{name} contains values outside the signed 64-bit range")
    values = offsets.astype(np.int64, copy=False)
    if values[0] != 0 or np.any(values < 0) or np.any(np.diff(values) < 0):
        raise GoldenValidationError(f"{name} must start at zero and be non-decreasing")
    for array_name, size in terminal_sizes.items():
        if int(values[-1]) != size:
            raise GoldenValidationError(
                f"{name} terminal offset {int(values[-1])} does not match {array_name} size {size}"
            )


def _validate_npz_layout(arrays: Mapping[str, NDArray[Any]]) -> None:
    for name in REQUIRED_ARRAYS:
        _validate_1d(arrays[name], name)
    _validate_1d(arrays["profile_indices"], "profile_indices", dtype=np.dtype(np.int64))
    _validate_1d(arrays["raw_offsets"], "raw_offsets", dtype=np.dtype(np.int64))
    _validate_1d(arrays["raw_height_u16"], "raw_height_u16", dtype=np.dtype(np.uint16))
    _validate_1d(arrays["intensity_u8"], "intensity_u8", dtype=np.dtype(np.uint8))
    _validate_1d(
        arrays["calibrated_height_inches"],
        "calibrated_height_inches",
        dtype=np.dtype(np.float64),
    )
    _validate_1d(arrays["reduced_offsets"], "reduced_offsets", dtype=np.dtype(np.int64))
    _validate_1d(arrays["reduced_x_inches"], "reduced_x_inches", dtype=np.dtype(np.float64))
    _validate_1d(arrays["reduced_y_inches"], "reduced_y_inches", dtype=np.dtype(np.float64))

    profile_count = int(arrays["profile_indices"].size)
    if profile_count == 0:
        raise GoldenValidationError("companion NPZ must contain at least one profile")
    _validate_offsets(
        arrays["raw_offsets"],
        "raw_offsets",
        profile_count,
        {
            "raw_height_u16": int(arrays["raw_height_u16"].size),
            "calibrated_height_inches": int(arrays["calibrated_height_inches"].size),
            "intensity_u8": int(arrays["intensity_u8"].size),
        },
    )
    _validate_offsets(
        arrays["reduced_offsets"],
        "reduced_offsets",
        profile_count,
        {
            "reduced_x_inches": int(arrays["reduced_x_inches"].size),
            "reduced_y_inches": int(arrays["reduced_y_inches"].size),
        },
    )
    profile_indices = arrays["profile_indices"].astype(np.int64, copy=False)
    if np.any(profile_indices < 0) or np.unique(profile_indices).size != profile_count:
        raise GoldenValidationError("profile_indices must contain unique non-negative values")
    if not np.all(np.isfinite(arrays["reduced_x_inches"])) or not np.all(np.isfinite(arrays["reduced_y_inches"])):
        raise GoldenValidationError("expected reduced coordinates must contain only finite values")


def _validate_expected_geometry(value: Any, location: str) -> None:
    if value is None:
        return
    geometry = _mapping(value, location)
    _number(_member(geometry, "rutting_inches", location), f"{location}.rutting_inches")
    for point_name in POINT_NAMES:
        point = _mapping(_member(geometry, point_name, location), f"{location}.{point_name}")
        _number(_member(point, "x_inches", f"{location}.{point_name}"), f"{location}.{point_name}.x_inches")
        _number(_member(point, "y_inches", f"{location}.{point_name}"), f"{location}.{point_name}.y_inches")


def _validate_profile_records(manifest: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]) -> None:
    records = _list(_member(manifest, "profiles", "manifest"), "profiles")
    profile_count = int(arrays["profile_indices"].size)
    if len(records) != profile_count:
        raise GoldenValidationError("profiles length must equal NPZ profile_indices length")
    slots: set[int] = set()
    seen_indices: set[int] = set()
    raw_offsets = arrays["raw_offsets"].astype(np.int64, copy=False)
    reduced_offsets = arrays["reduced_offsets"].astype(np.int64, copy=False)
    profile_indices = arrays["profile_indices"].astype(np.int64, copy=False)
    for position, untyped_record in enumerate(records):
        record = _mapping(untyped_record, f"profiles[{position}]")
        profile_index = _integer(
            _member(record, "profile_index", f"profiles[{position}]"),
            f"profiles[{position}].profile_index",
        )
        slot = _integer(
            _member(record, "npz_profile_slot", f"profiles[{position}]"),
            f"profiles[{position}].npz_profile_slot",
        )
        if profile_index < 0 or profile_index in seen_indices:
            raise GoldenValidationError("profile records must have unique non-negative profile_index values")
        if slot < 0 or slot >= profile_count or slot in slots:
            raise GoldenValidationError("profile records must map one-to-one onto valid NPZ slots")
        if int(profile_indices[slot]) != profile_index:
            raise GoldenValidationError(f"profile {profile_index} does not match profile_indices slot {slot}")
        raw_count = _integer(
            _member(record, "raw_count", f"profiles[{position}]"),
            f"profiles[{position}].raw_count",
        )
        reduced_count = _integer(
            _member(record, "reduced_count", f"profiles[{position}]"),
            f"profiles[{position}].reduced_count",
        )
        if raw_count <= 0 or raw_count != int(raw_offsets[slot + 1] - raw_offsets[slot]):
            raise GoldenValidationError(f"profile {profile_index} raw_count disagrees with raw_offsets")
        if reduced_count <= 0 or reduced_count != int(reduced_offsets[slot + 1] - reduced_offsets[slot]):
            raise GoldenValidationError(f"profile {profile_index} reduced_count disagrees with reduced_offsets")
        reduced_start, reduced_stop = int(reduced_offsets[slot]), int(reduced_offsets[slot + 1])
        expected_x = arrays["reduced_x_inches"][reduced_start:reduced_stop]
        if np.any(np.diff(expected_x) <= 0.0):
            raise GoldenValidationError(f"profile {profile_index} expected reduced X must be strictly increasing")
        status = _string(
            _member(record, "status", f"profiles[{position}]"),
            f"profiles[{position}].status",
        )
        if status != "ok":
            raise GoldenValidationError(
                f"profile {profile_index} has oracle status {status!r}; only successful records are comparable"
            )
        _boolean(
            _member(record, "is_profile_noisy", f"profiles[{position}]"),
            f"profiles[{position}].is_profile_noisy",
        )
        cross_slope = _mapping(
            _member(record, "cross_slope", f"profiles[{position}]"),
            f"profiles[{position}].cross_slope",
        )
        _number(
            _member(cross_slope, "percent", f"profiles[{position}].cross_slope"),
            f"profiles[{position}].cross_slope.percent",
        )
        _number(
            _member(cross_slope, "angle_degrees", f"profiles[{position}].cross_slope"),
            f"profiles[{position}].cross_slope.angle_degrees",
        )
        rutting = _mapping(
            _member(record, "rutting", f"profiles[{position}]"),
            f"profiles[{position}].rutting",
        )
        for side in ("left", "right"):
            _validate_expected_geometry(
                _member(rutting, side, f"profiles[{position}].rutting"),
                f"profiles[{position}].rutting.{side}",
            )
        slots.add(slot)
        seen_indices.add(profile_index)


def load_golden(metadata_path: Path) -> GoldenInputs:
    """Load and validate one real-profile manifest and companion array file."""

    try:
        metadata_path = metadata_path.resolve(strict=True)
    except OSError as exc:
        raise GoldenValidationError(f"manifest not found: {metadata_path}") from exc
    if not metadata_path.is_file():
        raise GoldenValidationError(f"manifest is not a regular file: {metadata_path}")
    manifest = _load_strict_json(metadata_path)
    if _member(manifest, "schema", "manifest") != "pathview-oracle-golden":
        raise GoldenValidationError("unsupported manifest schema")
    if _member(manifest, "schema_version", "manifest") != SCHEMA_VERSION:
        raise GoldenValidationError(f"only schema_version {SCHEMA_VERSION} is supported")
    if _member(manifest, "kind", "manifest") != "real_3dc_profiles":
        raise GoldenValidationError("comparison requires kind='real_3dc_profiles'")

    arrays_record = _mapping(_member(manifest, "arrays", "manifest"), "arrays")
    npz_path = _resolve_companion(metadata_path, arrays_record)
    expected_hash = _string(_member(arrays_record, "sha256", "arrays"), "arrays.sha256").lower()
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
        raise GoldenValidationError("arrays.sha256 must be a lowercase hexadecimal SHA-256 digest")
    actual_hash = _sha256(npz_path)
    if actual_hash != expected_hash:
        raise GoldenValidationError(f"companion NPZ SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    arrays = _load_npz(npz_path)
    _validate_npz_layout(arrays)
    _validate_profile_records(manifest, arrays)

    calibration = _mapping(_member(manifest, "calibration", "manifest"), "calibration")
    pixel_width = _mapping(_member(calibration, "pixel_width", "calibration"), "calibration.pixel_width")
    pixel_width_inches = _number(
        _member(pixel_width, "inches", "calibration.pixel_width"),
        "calibration.pixel_width.inches",
    )
    if pixel_width_inches <= 0.0:
        raise GoldenValidationError("calibration.pixel_width.inches must be positive")

    parameters = _mapping(_member(manifest, "parameters", "manifest"), "parameters")
    if _boolean(_member(parameters, "is_double_laser", "parameters"), "parameters.is_double_laser"):
        raise GoldenValidationError("double-laser oracle exports are not supported by pathview_observed()")
    dark_bands = _list(_member(parameters, "dark_band_columns", "parameters"), "parameters.dark_band_columns")
    if dark_bands:
        raise GoldenValidationError("oracle exports with dark-band columns are not supported")
    roll_degrees = _number(_member(parameters, "roll_degrees", "parameters"), "parameters.roll_degrees")
    maximum_noise = _number(
        _member(parameters, "max_stddev_inches_high_noise", "parameters"),
        "parameters.max_stddev_inches_high_noise",
    )
    lane_left = _number(
        _member(parameters, "lane_left_inches", "parameters"),
        "parameters.lane_left_inches",
    )
    lane_right = _number(
        _member(parameters, "lane_right_inches", "parameters"),
        "parameters.lane_right_inches",
    )
    bar_record = _mapping(_member(parameters, "rut_bar_used", "parameters"), "parameters.rut_bar_used")
    bar_length = _number(
        _member(bar_record, "inches", "parameters.rut_bar_used"),
        "parameters.rut_bar_used.inches",
    )
    if bar_length <= 0.0:
        raise GoldenValidationError("parameters.rut_bar_used.inches must be positive")

    public_contract = _mapping(
        _member(manifest, "public_api_contract", "manifest"),
        "public_api_contract",
    )
    constants = _mapping(
        _member(public_contract, "rut_bar_public_constants", "public_api_contract"),
        "public_api_contract.rut_bar_public_constants",
    )
    wheel_width = _number(
        _member(constants, "rut_path_width_inches", "rut_bar_public_constants"),
        "public_api_contract.rut_bar_public_constants.rut_path_width_inches",
    )
    wheel_offset = _number(
        _member(constants, "wheel_path_center_inches_from_centerline", "rut_bar_public_constants"),
        "public_api_contract.rut_bar_public_constants.wheel_path_center_inches_from_centerline",
    )
    try:
        lane_geometry = LaneGeometry(
            left_edge_inches=lane_left,
            right_edge_inches=lane_right,
            wheel_path_center_offset_inches=wheel_offset,
            wheel_path_width_inches=wheel_width,
        )
        reduction_config = ReductionConfig.pathview_observed(
            roll_degrees=roll_degrees,
            max_stddev_inches_high_noise=maximum_noise,
        )
    except ValueError as exc:
        raise GoldenValidationError(f"invalid processing parameters: {exc}") from exc
    return GoldenInputs(
        metadata_path=metadata_path,
        npz_path=npz_path,
        manifest=manifest,
        arrays=arrays,
        pixel_width_inches=pixel_width_inches,
        lane_geometry=lane_geometry,
        reduction_config=reduction_config,
        bar_length_inches=bar_length,
    )


def _array_check(expected: NDArray[np.float64], actual: NDArray[np.float64], tolerance: float) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {
            "passed": False,
            "expected_count": int(expected.size),
            "actual_count": int(actual.size),
            "max_abs_error": None,
            "rmse": None,
            "tolerance": tolerance,
        }
    if expected.size == 0:
        max_error = rmse = 0.0
    else:
        difference = actual - expected
        max_error = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
    return {
        "passed": bool(max_error <= tolerance),
        "expected_count": int(expected.size),
        "actual_count": int(actual.size),
        "max_abs_error": max_error,
        "rmse": rmse,
        "tolerance": tolerance,
    }


def _scalar_check(expected: Any, actual: Any, tolerance: float) -> dict[str, Any]:
    if expected is None or actual is None:
        passed = expected is None and actual is None
        return {
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "abs_error": 0.0 if passed else None,
            "tolerance": tolerance,
        }
    expected_value = _number(expected, "expected comparison value")
    actual_value = float(actual)
    error = abs(actual_value - expected_value) if math.isfinite(actual_value) else None
    return {
        "passed": bool(error is not None and error <= tolerance),
        "expected": expected_value,
        "actual": actual_value if math.isfinite(actual_value) else None,
        "abs_error": error,
        "tolerance": tolerance,
    }


def _point(x: float, y: float) -> dict[str, float]:
    return {"x_inches": float(x), "y_inches": float(y)}


def _actual_geometry(
    wheel: WheelPathRut | None,
    lane: LaneGeometry,
    bar_length_inches: float,
) -> dict[str, Any] | None:
    if wheel is None:
        return None
    if wheel.measurement_x_inches is None or wheel.measurement_elevation_inches is None:
        raise RuntimeError("rut result did not provide its bar measurement point")
    wheel_center = lane.wheel_path_center(wheel.side)
    half_horizontal = (bar_length_inches / 2.0) / math.sqrt(1.0 + wheel.bar_slope**2)
    left_reference_x = wheel_center - half_horizontal
    right_reference_x = wheel_center + half_horizontal

    def bar_y(x: float) -> float:
        return wheel.measurement_elevation_inches + wheel.bar_slope * (x - wheel.measurement_x_inches)

    return {
        "rutting_inches": wheel.rut_depth_inches,
        "left_reference_point": _point(left_reference_x, bar_y(left_reference_x)),
        "right_reference_point": _point(right_reference_x, bar_y(right_reference_x)),
        "left_contact_point": _point(
            wheel.left_contact_x_inches,
            wheel.left_contact_elevation_inches,
        ),
        "right_contact_point": _point(
            wheel.right_contact_x_inches,
            wheel.right_contact_elevation_inches,
        ),
        "measurement_point": _point(
            wheel.measurement_x_inches,
            wheel.measurement_elevation_inches,
        ),
        "rut_point": _point(wheel.rut_x_inches, wheel.rut_elevation_inches),
    }


def _geometry_check(expected_value: Any, actual_value: Any, tolerance: float) -> dict[str, Any]:
    if expected_value is None or actual_value is None:
        return {
            "passed": expected_value is None and actual_value is None,
            "expected_present": expected_value is not None,
            "actual_present": actual_value is not None,
            "max_abs_error": 0.0 if expected_value is None and actual_value is None else None,
            "rmse": 0.0 if expected_value is None and actual_value is None else None,
            "tolerance": tolerance,
            "coordinates": {},
        }
    expected = _mapping(expected_value, "expected rutting geometry")
    actual = _mapping(actual_value, "actual rutting geometry")
    coordinates: dict[str, Any] = {}
    errors: list[float] = []
    passed = True
    for point_name in POINT_NAMES:
        expected_point = _mapping(
            _member(expected, point_name, "expected rutting geometry"),
            f"expected rutting geometry.{point_name}",
        )
        actual_point = _mapping(
            _member(actual, point_name, "actual rutting geometry"),
            f"actual rutting geometry.{point_name}",
        )
        point_checks: dict[str, Any] = {}
        for axis in ("x_inches", "y_inches"):
            check = _scalar_check(
                _member(expected_point, axis, f"expected rutting geometry.{point_name}"),
                _member(actual_point, axis, f"actual rutting geometry.{point_name}"),
                tolerance,
            )
            point_checks[axis] = check
            passed &= bool(check["passed"])
            if check["abs_error"] is not None:
                errors.append(float(check["abs_error"]))
        coordinates[point_name] = point_checks
    return {
        "passed": passed,
        "expected_present": True,
        "actual_present": True,
        "max_abs_error": max(errors, default=0.0),
        "rmse": float(math.sqrt(sum(error * error for error in errors) / len(errors))) if errors else 0.0,
        "tolerance": tolerance,
        "coordinates": coordinates,
    }


def compare_golden(inputs: GoldenInputs, tolerances: Tolerances | None = None) -> dict[str, Any]:
    """Run all stored profiles and return a strict-JSON-compatible report."""

    tolerances = tolerances or Tolerances()
    raw_offsets = inputs.arrays["raw_offsets"].astype(np.int64, copy=False)
    reduced_offsets = inputs.arrays["reduced_offsets"].astype(np.int64, copy=False)
    calibrated = inputs.arrays["calibrated_height_inches"]
    expected_x_all = inputs.arrays["reduced_x_inches"]
    expected_y_all = inputs.arrays["reduced_y_inches"]
    records = _list(inputs.manifest["profiles"], "profiles")
    reports: list[dict[str, Any]] = []

    for untyped_record in records:
        record = _mapping(untyped_record, "profile record")
        profile_index = int(record["profile_index"])
        slot = int(record["npz_profile_slot"])
        raw_start, raw_stop = int(raw_offsets[slot]), int(raw_offsets[slot + 1])
        reduced_start, reduced_stop = int(reduced_offsets[slot]), int(reduced_offsets[slot + 1])
        elevation = np.asarray(calibrated[raw_start:raw_stop], dtype=np.float64)
        x = np.arange(elevation.size, dtype=np.float64) * inputs.pixel_width_inches
        profile = TransverseProfile(x, elevation, profile_id=str(profile_index))
        reduced = reduce_profile(profile, inputs.reduction_config)
        cross_slope = fit_cross_slope(reduced, inputs.lane_geometry)
        rutting = measure_profile_rutting(
            reduced,
            inputs.lane_geometry,
            bar_length_inches=inputs.bar_length_inches,
        )
        expected_x = np.asarray(expected_x_all[reduced_start:reduced_stop], dtype=np.float64)
        expected_y = np.asarray(expected_y_all[reduced_start:reduced_stop], dtype=np.float64)
        expected_cross = _mapping(record["cross_slope"], f"profile {profile_index}.cross_slope")
        expected_rutting = _mapping(record["rutting"], f"profile {profile_index}.rutting")

        checks: dict[str, Any] = {
            "reduced_count": {
                "passed": reduced.point_count == int(record["reduced_count"]),
                "expected": int(record["reduced_count"]),
                "actual": reduced.point_count,
            },
            "reduced_x": _array_check(
                expected_x,
                np.asarray(reduced.x_inches),
                tolerances.reduced_inches,
            ),
            "reduced_y": _array_check(
                expected_y,
                np.asarray(reduced.elevation_inches),
                tolerances.reduced_inches,
            ),
            "is_profile_noisy": {
                "passed": reduced.is_noisy
                == _boolean(record["is_profile_noisy"], f"profile {profile_index}.is_profile_noisy"),
                "expected": bool(record["is_profile_noisy"]),
                "actual": reduced.is_noisy,
            },
            "cross_slope_percent": _scalar_check(
                expected_cross["percent"],
                cross_slope.percent,
                tolerances.cross_slope_percent,
            ),
            "cross_slope_angle_degrees": _scalar_check(
                expected_cross["angle_degrees"],
                cross_slope.angle_degrees,
                tolerances.cross_slope_angle_degrees,
            ),
        }
        actual_by_side = {
            "left": _actual_geometry(rutting.left, inputs.lane_geometry, inputs.bar_length_inches),
            "right": _actual_geometry(rutting.right, inputs.lane_geometry, inputs.bar_length_inches),
        }
        for side in ("left", "right"):
            expected_geometry = expected_rutting.get(side)
            actual_geometry = actual_by_side[side]
            expected_depth = (
                None
                if expected_geometry is None
                else _member(_mapping(expected_geometry, f"profile {profile_index}.{side}"), "rutting_inches", side)
            )
            actual_depth = None if actual_geometry is None else actual_geometry["rutting_inches"]
            checks[f"{side}_rut_depth"] = _scalar_check(
                expected_depth,
                actual_depth,
                tolerances.rut_depth_inches,
            )
            checks[f"{side}_geometry"] = _geometry_check(
                expected_geometry,
                actual_geometry,
                tolerances.geometry_inches,
            )

        failures = [name for name, check in checks.items() if not check["passed"]]
        reports.append(
            {
                "profile_index": profile_index,
                "passed": not failures,
                "failures": failures,
                "checks": checks,
            }
        )

    passed_count = sum(bool(report["passed"]) for report in reports)
    return {
        "schema": "pathview-golden-comparison",
        "schema_version": SCHEMA_VERSION,
        "passed": passed_count == len(reports),
        "metadata_file": inputs.metadata_path.name,
        "companion_file": inputs.npz_path.name,
        "tolerances": asdict(tolerances),
        "summary": {
            "profiles_total": len(reports),
            "profiles_passed": passed_count,
            "profiles_failed": len(reports) - passed_count,
        },
        "profiles": reports,
    }


def _format_error(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def print_human_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    overall = "PASS" if report["passed"] else "FAIL"
    print(
        f"PathView golden comparison: {overall} "
        f"({summary['profiles_passed']}/{summary['profiles_total']} profiles passed)"
    )
    print(f"Manifest: {report['metadata_file']}")
    for profile in report["profiles"]:
        status = "PASS" if profile["passed"] else "FAIL"
        print(f"[{status}] profile {profile['profile_index']}")
        checks = profile["checks"]
        print(
            "  reduced: "
            f"count {checks['reduced_count']['actual']}/{checks['reduced_count']['expected']}, "
            f"max |dx| {_format_error(checks['reduced_x']['max_abs_error'])} in, "
            f"max |dy| {_format_error(checks['reduced_y']['max_abs_error'])} in"
        )
        print(
            "  cross slope: "
            f"percent error {_format_error(checks['cross_slope_percent']['abs_error'])}, "
            f"angle error {_format_error(checks['cross_slope_angle_degrees']['abs_error'])} deg"
        )
        print(
            "  rut depth: "
            f"left error {_format_error(checks['left_rut_depth']['abs_error'])} in, "
            f"right error {_format_error(checks['right_rut_depth']['abs_error'])} in"
        )
        print(
            "  geometry: "
            f"left max {_format_error(checks['left_geometry']['max_abs_error'])} in, "
            f"right max {_format_error(checks['right_geometry']['max_abs_error'])} in"
        )
        if profile["failures"]:
            print(f"  failed checks: {', '.join(profile['failures'])}")


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_json", type=Path, help="Real-profile metadata JSON from export_pathview_golden.py")
    parser.add_argument("--json-output", type=Path, help="Also write the complete comparison report as strict JSON")
    parser.add_argument("--reduced-tolerance", type=_non_negative_float, default=1e-6, metavar="INCHES")
    parser.add_argument("--cross-slope-percent-tolerance", type=_non_negative_float, default=1e-4)
    parser.add_argument("--cross-slope-angle-tolerance", type=_non_negative_float, default=1e-4, metavar="DEGREES")
    parser.add_argument("--rut-depth-tolerance", type=_non_negative_float, default=1e-3, metavar="INCHES")
    parser.add_argument("--geometry-tolerance", type=_non_negative_float, default=1e-2, metavar="INCHES")
    return parser.parse_args(argv)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise GoldenValidationError(f"cannot write strict JSON report {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = load_golden(args.metadata_json)
        tolerances = Tolerances(
            reduced_inches=args.reduced_tolerance,
            cross_slope_percent=args.cross_slope_percent_tolerance,
            cross_slope_angle_degrees=args.cross_slope_angle_tolerance,
            rut_depth_inches=args.rut_depth_tolerance,
            geometry_inches=args.geometry_tolerance,
        )
        report = compare_golden(inputs, tolerances)
        if args.json_output is not None:
            output_path = args.json_output.resolve()
            if output_path in {inputs.metadata_path, inputs.npz_path}:
                raise GoldenValidationError("--json-output cannot overwrite the manifest or companion NPZ")
            _write_report(output_path, report)
    except (GoldenValidationError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print_human_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
