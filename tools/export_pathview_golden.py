#!/usr/bin/env python3
"""Export neutral PathView reference results for clean-room compatibility tests.

This Windows-only development tool calls the documented/public surface of a
locally installed, legitimately licensed PathView runtime through pythonnet. It
does not decompile assemblies, copy implementation code, or redistribute DLLs.

Two modes are available:

* ``real`` exports selected profiles from one user-supplied ``.3dc`` file to a
  JSON manifest plus a compressed NPZ array file.
* ``synthetic`` evaluates deterministic, non-survey profiles and writes a small
  JSON fixture that is safe to review and, subject to project policy, commit.

Real exports contain derived survey measurements. Keep them outside the Git
repository unless the data owner has explicitly approved publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ASSEMBLY_NAMES = (
    "Pathway.Core.dll",
    "Pathway.Data.dll",
    "Pathway.Processing.dll",
)
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class RuntimeTypes:
    Boolean: Any
    Double: Any
    Int32: Any
    Nullable: Any
    HashSet: Any
    List: Any
    Coordinate: Any
    ICoordinate: Any
    Length: Any
    ImageInfo: Any
    Surface3DCalibrationData: Any
    Surface3DImage: Any
    AashtoCrossSlope: Any
    DataReducedProfile: Any
    LanePositions: Any
    RutBarRutting: Any


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("provide one or more non-negative indices")
    return values


def parse_optional_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    return parse_csv_ints(value)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pathview-dir",
        type=Path,
        default=None,
        help="Local PathView installation. Defaults to the PATHVIEW_DIR environment variable.",
    )
    parser.add_argument("--roll-degrees", type=float, default=0.0)
    parser.add_argument("--is-double-laser", action="store_true")
    parser.add_argument("--max-stddev-inches-high-noise", type=float, default=999.0)
    parser.add_argument("--lane-left-inches", type=float, default=0.0)
    parser.add_argument(
        "--lane-right-inches",
        type=float,
        default=None,
        help="Defaults to the calibration pavement width (real) or 162 inches (synthetic).",
    )
    parser.add_argument("--default-edge-distance-inches", type=float, default=0.0)
    parser.add_argument(
        "--rut-bar-meters",
        type=float,
        default=None,
        help="Defaults to PathView RutBarRutting.DefaultRutBarLength.",
    )
    parser.add_argument("--calculate-center-rutting", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    real = subparsers.add_parser("real", help="Export selected profiles from one user-supplied .3dc file.")
    add_runtime_arguments(real)
    real.add_argument("--input-3dc", type=Path, required=True)
    real.add_argument(
        "--calibration-dir",
        type=Path,
        required=True,
        help="Dataset/set directory containing the matching 3D camera calibration files.",
    )
    real.add_argument("--output-json", type=Path, required=True)
    real.add_argument("--profile-indices", type=parse_csv_ints, default=[0])
    real.add_argument("--dark-band-columns", type=parse_optional_csv_ints, default=[])
    real.add_argument(
        "--include-source-paths",
        action="store_true",
        help="Include absolute input/install paths. Off by default to keep manifests portable.",
    )

    synthetic = subparsers.add_parser("synthetic", help="Create a deterministic fixture without reading survey data.")
    add_runtime_arguments(synthetic)
    synthetic.add_argument("--output-json", type=Path, required=True)
    synthetic.add_argument(
        "--sample-step-inches",
        type=float,
        default=0.5,
        help="Synthetic transverse sampling interval. Default: 0.5 inches.",
    )

    args = parser.parse_args(argv)
    if args.rut_bar_meters is not None and args.rut_bar_meters <= 0:
        parser.error("--rut-bar-meters must be greater than zero")
    if args.lane_right_inches is not None and args.lane_right_inches <= args.lane_left_inches:
        parser.error("--lane-right-inches must be greater than --lane-left-inches")
    if args.command == "synthetic" and args.sample_step_inches <= 0:
        parser.error("--sample-step-inches must be greater than zero")
    return args


def resolve_pathview_dir(value: Path | None) -> Path:
    candidates: list[Path] = []
    if value is not None:
        candidates.append(value)
    env_value = os.environ.get("PATHVIEW_DIR")
    if env_value:
        candidates.append(Path(env_value))

    if not candidates:
        raise FileNotFoundError("No PathView runtime location was supplied. Use --pathview-dir or set PATHVIEW_DIR.")

    for candidate in candidates:
        if (candidate / "PathView.runtimeconfig.json").is_file():
            missing = [name for name in ASSEMBLY_NAMES if not (candidate / name).is_file()]
            if not missing:
                return candidate.resolve()
    raise FileNotFoundError(
        "No complete PathView runtime was found. Checked: " + ", ".join(str(item) for item in candidates)
    )


def load_pathview_runtime(pathview_dir: Path) -> RuntimeTypes:
    try:
        from pythonnet import load
    except ImportError as exc:
        raise RuntimeError("pythonnet is required on the Windows oracle machine") from exc

    os.environ["PATH"] = os.pathsep.join((str(pathview_dir), str(pathview_dir / "Modules"), os.environ.get("PATH", "")))
    load("coreclr", runtime_config=str(pathview_dir / "PathView.runtimeconfig.json"))

    import clr  # type: ignore

    for assembly_name in ASSEMBLY_NAMES:
        clr.AddReference(str(pathview_dir / assembly_name))

    from Pathway.Core import Coordinate, ICoordinate  # type: ignore
    from Pathway.Core.Units import Length  # type: ignore
    from Pathway.Data.Image import ImageInfo  # type: ignore
    from Pathway.Data.Image.Surface3D import (  # type: ignore
        Surface3DCalibrationData,
        Surface3DImage,
    )
    from Pathway.Processing.LaneProfile import (  # type: ignore
        AashtoCrossSlope,
        DataReducedProfile,
        LanePositions,
        RutBarRutting,
    )
    from System import Boolean, Double, Int32, Nullable  # type: ignore
    from System.Collections.Generic import HashSet, List  # type: ignore

    return RuntimeTypes(
        Boolean=Boolean,
        Double=Double,
        Int32=Int32,
        Nullable=Nullable,
        HashSet=HashSet,
        List=List,
        Coordinate=Coordinate,
        ICoordinate=ICoordinate,
        Length=Length,
        ImageInfo=ImageInfo,
        Surface3DCalibrationData=Surface3DCalibrationData,
        Surface3DImage=Surface3DImage,
        AashtoCrossSlope=AashtoCrossSlope,
        DataReducedProfile=DataReducedProfile,
        LanePositions=LanePositions,
        RutBarRutting=RutBarRutting,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assembly_manifest(pathview_dir: Path) -> list[dict[str, Any]]:
    from System.Diagnostics import FileVersionInfo  # type: ignore
    from System.Reflection import AssemblyName  # type: ignore

    result: list[dict[str, Any]] = []
    for file_name in ASSEMBLY_NAMES:
        path = pathview_dir / file_name
        assembly = AssemblyName.GetAssemblyName(str(path))
        file_info = FileVersionInfo.GetVersionInfo(str(path))
        token = assembly.GetPublicKeyToken()
        result.append(
            {
                "file_name": file_name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "assembly_name": str(assembly.Name),
                "assembly_version": str(assembly.Version),
                "file_version": str(file_info.FileVersion) if file_info.FileVersion else None,
                "product_version": str(file_info.ProductVersion) if file_info.ProductVersion else None,
                "public_key_token": "".join(f"{int(item):02x}" for item in token) if token else None,
            }
        )
    runtime_config = pathview_dir / "PathView.runtimeconfig.json"
    result.append(
        {
            "file_name": runtime_config.name,
            "size_bytes": runtime_config.stat().st_size,
            "sha256": sha256_file(runtime_config),
            "assembly_name": None,
            "assembly_version": None,
            "file_version": None,
            "product_version": None,
            "public_key_token": None,
        }
    )
    return result


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def coordinate_record(value: Any) -> dict[str, float | None] | None:
    if value is None:
        return None
    return {"x_inches": finite_or_none(value.X), "y_inches": finite_or_none(value.Y)}


def geometry_record(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "rutting_inches": finite_or_none(value.RuttingInches),
        "left_reference_point": coordinate_record(value.LeftReferencePointInches),
        "right_reference_point": coordinate_record(value.RightReferencePointInches),
        "left_contact_point": coordinate_record(value.LeftContactPointInches),
        "right_contact_point": coordinate_record(value.RightContactPointInches),
        "measurement_point": coordinate_record(value.MeasurementPointInches),
        "rut_point": coordinate_record(value.RutPointInches),
    }


def length_record(value: Any) -> dict[str, Any]:
    return {
        "text": str(value),
        "inches": finite_or_none(value.Inches),
        "meters": finite_or_none(value.Meters),
    }


def lane_positions(types: RuntimeTypes, left: float, right: float, default_edge: float) -> Any:
    return types.LanePositions(
        types.Nullable[types.Double](float(left)),
        types.Nullable[types.Double](float(right)),
        float(default_edge),
    )


def rut_bar(types: RuntimeTypes, meters: float | None) -> Any:
    if meters is None:
        return types.RutBarRutting.DefaultRutBarLength
    return types.Length.FromMeters(float(meters))


def public_contract(types: RuntimeTypes) -> dict[str, Any]:
    return {
        "calls_used": [
            "Pathway.Data.Image.ImageInfo(string)",
            "Pathway.Data.Image.Surface3D.Surface3DCalibrationData(string)",
            "Pathway.Data.Image.Surface3D.Surface3DImage(ImageInfo, calibration).GetProfiles()",
            "ISurface3DProfile.GetRawHeightValue(int), HeightInches, Color",
            "Pathway.Processing.LaneProfile.DataReducedProfile.GetFrom3DC(...) (real mode)",
            "Pathway.Processing.LaneProfile.DataReducedProfile(IEnumerable<ICoordinate>, ...) (synthetic mode)",
            "Pathway.Processing.LaneProfile.AashtoCrossSlope(...) ",
            "Pathway.Processing.LaneProfile.RutBarRutting(...) ",
        ],
        "interfaces": {
            "DataReducedProfile.GetFrom3DC": {
                "inputs": [
                    "ISurface3DProfile profile",
                    "ISurface3DCalibration calibration",
                    "bool isDoubleLaser",
                    "double rollDegrees",
                    "HashSet<int> darkBandColumns",
                    "double maxStdDevInchesHighNoise",
                ],
                "outputs": ["IReadOnlyList<ICoordinate> Profile", "bool IsProfileNoisy"],
            },
            "AashtoCrossSlope": {
                "inputs": ["DataReducedProfile dataReduced", "LanePositions lanePositions"],
                "outputs": ["double Percent", "double AngleDegrees"],
            },
            "RutBarRutting": {
                "inputs": [
                    "DataReducedProfile dataReduced",
                    "LanePositions lanePositions",
                    "ICrossSlope crossSlope",
                    "Length rutBarLength",
                    "bool calculateCenterRutting",
                ],
                "outputs": [
                    "IRuttingGeometry Left",
                    "IRuttingGeometry Right",
                    "IRuttingGeometry Center",
                    "RuttingInches plus six reference/contact/measurement/rut coordinates per geometry",
                ],
            },
        },
        "rut_bar_public_constants": {
            "default_length": length_record(types.RutBarRutting.DefaultRutBarLength),
            "rut_path_width_inches": finite_or_none(types.RutBarRutting.RutPathWidthInches),
            "rut_path_half_width_inches": finite_or_none(types.RutBarRutting.RutPathHalfWidthInches),
            "wheel_path_center_inches_from_centerline": finite_or_none(
                types.RutBarRutting.WheelPathCenterInchesFromCenterline
            ),
        },
        "method": "Public API calls and runtime reflection metadata only; no decompilation or vendor code copying.",
    }


def common_manifest(args: argparse.Namespace, pathview_dir: Path, types: RuntimeTypes) -> dict[str, Any]:
    return {
        "schema": "pathview-oracle-golden",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": timestamp_utc(),
        "generator": "tools/export_pathview_golden.py",
        "provenance": {
            "purpose": "clean-room compatibility testing",
            "contains_vendor_binary": False,
            "contains_decompiled_code": False,
            "pathview_install_path": str(pathview_dir) if getattr(args, "include_source_paths", False) else None,
            "runtime_files": assembly_manifest(pathview_dir),
        },
        "public_api_contract": public_contract(types),
        "parameters": {
            "is_double_laser": bool(args.is_double_laser),
            "roll_degrees": float(args.roll_degrees),
            "max_stddev_inches_high_noise": float(args.max_stddev_inches_high_noise),
            "lane_left_inches": float(args.lane_left_inches),
            "lane_right_inches": finite_or_none(args.lane_right_inches),
            "default_edge_distance_inches": float(args.default_edge_distance_inches),
            "rut_bar_meters_override": finite_or_none(args.rut_bar_meters),
            "calculate_center_rutting": bool(args.calculate_center_rutting),
        },
    }


def analyze_reduced_profile(
    data_reduced: Any,
    lane: Any,
    bar: Any,
    types: RuntimeTypes,
    calculate_center: bool,
) -> tuple[dict[str, Any], list[float], list[float]]:
    reduced_x = [float(point.X) for point in data_reduced.Profile]
    reduced_y = [float(point.Y) for point in data_reduced.Profile]
    cross_slope = types.AashtoCrossSlope(data_reduced, lane)
    rutting = types.RutBarRutting(
        data_reduced,
        lane,
        cross_slope,
        bar,
        types.Boolean(bool(calculate_center)),
    )
    result = {
        "status": "ok",
        "error": None,
        "is_profile_noisy": bool(data_reduced.IsProfileNoisy),
        "cross_slope": {
            "percent": finite_or_none(cross_slope.Percent),
            "angle_degrees": finite_or_none(cross_slope.AngleDegrees),
        },
        "rutting": {
            "left": geometry_record(rutting.Left),
            "right": geometry_record(rutting.Right),
            "center": geometry_record(rutting.Center),
        },
        "reduced_count": len(reduced_x),
    }
    return result, reduced_x, reduced_y


def save_real_npz(path: Path, arrays: list[dict[str, Any]]) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to write the real-profile NPZ") from exc

    def pack(key: str, dtype: Any) -> tuple[Any, Any]:
        offsets = [0]
        values = []
        for item in arrays:
            array = np.asarray(item[key], dtype=dtype)
            values.append(array)
            offsets.append(offsets[-1] + int(array.size))
        flat = np.concatenate(values) if values else np.asarray([], dtype=dtype)
        return flat, np.asarray(offsets, dtype=np.int64)

    raw_u16, raw_offsets = pack("raw_height_u16", np.uint16)
    height_inches, height_offsets = pack("calibrated_height_inches", np.float64)
    intensity_u8, intensity_offsets = pack("intensity_u8", np.uint8)
    reduced_x, reduced_offsets = pack("reduced_x_inches", np.float64)
    reduced_y, reduced_y_offsets = pack("reduced_y_inches", np.float64)
    if not (
        raw_offsets.tolist() == height_offsets.tolist() == intensity_offsets.tolist()
        and reduced_offsets.tolist() == reduced_y_offsets.tolist()
    ):
        raise RuntimeError("array lengths are internally inconsistent")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        profile_indices=np.asarray([item["profile_index"] for item in arrays], dtype=np.int64),
        raw_offsets=raw_offsets,
        raw_height_u16=raw_u16,
        calibrated_height_inches=height_inches,
        intensity_u8=intensity_u8,
        reduced_offsets=reduced_offsets,
        reduced_x_inches=reduced_x,
        reduced_y_inches=reduced_y,
    )


def real_export(args: argparse.Namespace, pathview_dir: Path, types: RuntimeTypes) -> dict[str, Any]:
    input_3dc = args.input_3dc.resolve()
    calibration_dir = args.calibration_dir.resolve()
    if not input_3dc.is_file() or input_3dc.suffix.lower() != ".3dc":
        raise FileNotFoundError(f"input .3dc file not found: {input_3dc}")
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"calibration directory not found: {calibration_dir}")

    calibration_data = types.Surface3DCalibrationData(str(calibration_dir))
    if not bool(calibration_data.Exists):
        raise RuntimeError(f"PathView did not find usable calibration data in {calibration_dir}")
    calibration = calibration_data.Calibration
    lane_right = (
        float(args.lane_right_inches) if args.lane_right_inches is not None else float(calibration.PavementWidth.Inches)
    )
    if lane_right <= args.lane_left_inches:
        raise ValueError("resolved lane right edge must exceed the left edge")
    lane = lane_positions(types, args.lane_left_inches, lane_right, args.default_edge_distance_inches)
    bar = rut_bar(types, args.rut_bar_meters)
    dark_bands = types.HashSet[types.Int32]()
    for column in args.dark_band_columns:
        dark_bands.Add(types.Int32(column))

    selected = set(args.profile_indices)
    maximum = max(selected)
    records: list[dict[str, Any]] = []
    arrays: list[dict[str, Any]] = []
    surface = types.Surface3DImage(types.ImageInfo(str(input_3dc)), calibration)
    for profile_index, profile in enumerate(surface.GetProfiles()):
        if profile_index > maximum:
            break
        if profile_index not in selected:
            continue
        height_inches = [float(value) for value in profile.HeightInches]
        intensity_u8 = [int(value) for value in profile.Color]
        if len(height_inches) != len(intensity_u8):
            raise RuntimeError(f"profile {profile_index} has inconsistent raw array lengths")
        raw_height_u16 = [int(profile.GetRawHeightValue(index)) for index in range(len(height_inches))]

        record: dict[str, Any] = {
            "profile_index": profile_index,
            "raw_count": len(raw_height_u16),
            "npz_profile_slot": len(arrays),
        }
        reduced_x: list[float] = []
        reduced_y: list[float] = []
        try:
            data_reduced = types.DataReducedProfile.GetFrom3DC(
                profile,
                calibration,
                types.Boolean(bool(args.is_double_laser)),
                float(args.roll_degrees),
                dark_bands,
                float(args.max_stddev_inches_high_noise),
            )
            analysis, reduced_x, reduced_y = analyze_reduced_profile(
                data_reduced,
                lane,
                bar,
                types,
                args.calculate_center_rutting,
            )
            record.update(analysis)
        except Exception as exc:  # Preserve raw arrays even if a downstream vendor call rejects a profile.
            record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_profile_noisy": None,
                    "cross_slope": None,
                    "rutting": None,
                    "reduced_count": 0,
                }
            )
        records.append(record)
        arrays.append(
            {
                "profile_index": profile_index,
                "raw_height_u16": raw_height_u16,
                "calibrated_height_inches": height_inches,
                "intensity_u8": intensity_u8,
                "reduced_x_inches": reduced_x,
                "reduced_y_inches": reduced_y,
            }
        )

    found = {item["profile_index"] for item in records}
    missing = sorted(selected - found)
    if missing:
        raise IndexError(f"requested profile indices were not found: {missing}")

    output_json = args.output_json.resolve()
    output_npz = output_json.with_name(f"{output_json.stem}.arrays.npz")
    save_real_npz(output_npz, arrays)

    manifest = common_manifest(args, pathview_dir, types)
    manifest.update(
        {
            "kind": "real_3dc_profiles",
            "publication_warning": (
                "This manifest and its NPZ contain measurements derived from survey data. "
                "Do not commit or publish without data-owner approval."
            ),
            "source": {
                "file_name": input_3dc.name,
                "parent_name": input_3dc.parent.name,
                "size_bytes": input_3dc.stat().st_size,
                "sha256": sha256_file(input_3dc),
                "calibration_directory_name": calibration_dir.name,
                "input_path": str(input_3dc) if args.include_source_paths else None,
                "calibration_path": str(calibration_dir) if args.include_source_paths else None,
            },
            "calibration": {
                "camera_mode": str(calibration.CameraMode),
                "pixel_width": length_record(calibration.PixelWidth),
                "pavement_width": length_record(calibration.PavementWidth),
                "height_resolution": length_record(calibration.HeightResolution),
            },
            "parameters": {
                **manifest["parameters"],
                "lane_right_inches": lane_right,
                "dark_band_columns": list(args.dark_band_columns),
                "rut_bar_used": length_record(bar),
                "calculate_center_rutting": bool(args.calculate_center_rutting),
            },
            "arrays": {
                "format": "NumPy NPZ (compressed, no object arrays, allow_pickle=False compatible)",
                "file_name": output_npz.name,
                "sha256": None,
                "layout": {
                    "profile_indices": "NPZ slot to source profile index",
                    "raw_offsets": "slices raw_height_u16, calibrated_height_inches, intensity_u8",
                    "reduced_offsets": "slices reduced_x_inches and reduced_y_inches",
                },
            },
            "profiles": records,
        }
    )
    manifest["arrays"]["sha256"] = sha256_file(output_npz)
    write_json(output_json, manifest)
    return manifest


def synthetic_definitions() -> list[tuple[str, str, Callable[[float], float]]]:
    return [
        ("flat", "y = 0", lambda x: 0.0),
        ("planar_positive_one_percent", "y = 0.01 * x", lambda x: 0.01 * x),
        (
            "twin_gaussian_ruts",
            "y = -0.5*exp(-((x-45)/8)^2) - 0.8*exp(-((x-117)/8)^2)",
            lambda x: -0.5 * math.exp(-(((x - 45.0) / 8.0) ** 2)) - 0.8 * math.exp(-(((x - 117.0) / 8.0) ** 2)),
        ),
        ("center_v", "y = abs(x - 81) / 100", lambda x: abs(x - 81.0) / 100.0),
    ]


def synthetic_export(args: argparse.Namespace, pathview_dir: Path, types: RuntimeTypes) -> dict[str, Any]:
    lane_right = float(args.lane_right_inches) if args.lane_right_inches is not None else 162.0
    if lane_right <= args.lane_left_inches:
        raise ValueError("resolved lane right edge must exceed the left edge")
    lane = lane_positions(types, args.lane_left_inches, lane_right, args.default_edge_distance_inches)
    bar = rut_bar(types, args.rut_bar_meters)
    sample_count = int(round((lane_right - args.lane_left_inches) / args.sample_step_inches)) + 1
    input_x = [args.lane_left_inches + index * args.sample_step_inches for index in range(sample_count)]
    if not math.isclose(input_x[-1], lane_right, abs_tol=1e-9):
        raise ValueError("synthetic step must divide the lane width exactly")

    profiles: list[dict[str, Any]] = []
    for name, formula, function in synthetic_definitions():
        input_y = [float(function(x)) for x in input_x]
        coordinates = types.List[types.ICoordinate]()
        for x, y in zip(input_x, input_y, strict=True):
            coordinates.Add(types.Coordinate(float(x), float(y), None, None))
        data_reduced = types.DataReducedProfile(
            coordinates,
            types.Boolean(bool(args.is_double_laser)),
            float(args.roll_degrees),
            float(args.max_stddev_inches_high_noise),
        )
        analysis, reduced_x, reduced_y = analyze_reduced_profile(
            data_reduced,
            lane,
            bar,
            types,
            args.calculate_center_rutting,
        )
        profiles.append(
            {
                "name": name,
                "formula": formula,
                "input_y_inches": input_y,
                "reduced_x_inches": reduced_x,
                "reduced_y_inches": reduced_y,
                **analysis,
            }
        )

    manifest = common_manifest(args, pathview_dir, types)
    manifest.update(
        {
            "kind": "synthetic_profiles",
            "publication_warning": None,
            "parameters": {
                **manifest["parameters"],
                "lane_right_inches": lane_right,
                "rut_bar_used": length_record(bar),
            },
            "sampling": {
                "x_units": "inches",
                "y_units": "inches",
                "x_start": input_x[0],
                "x_stop": input_x[-1],
                "x_step": float(args.sample_step_inches),
                "input_x_inches": input_x,
            },
            "profiles": profiles,
        }
    )
    write_json(args.output_json.resolve(), manifest)
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pathview_dir = resolve_pathview_dir(args.pathview_dir)
        types = load_pathview_runtime(pathview_dir)
        if args.command == "real":
            manifest = real_export(args, pathview_dir, types)
            output_json = args.output_json.resolve()
            output_npz = output_json.with_name(manifest["arrays"]["file_name"])
            outputs = [str(output_json), str(output_npz)]
        else:
            manifest = synthetic_export(args, pathview_dir, types)
            outputs = [str(args.output_json.resolve())]
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"[DONE] wrote {manifest['kind']} golden data")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
