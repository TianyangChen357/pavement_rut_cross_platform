from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pavement_rut.domain import (
    LaneGeometry,
    ReductionConfig,
    TransverseProfile,
    fit_cross_slope,
    measure_profile_rutting,
    reduce_profile,
)
from pavement_rut.domain.models import WheelPathRut

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARE_TOOL = REPOSITORY_ROOT / "tools" / "compare_pathview_golden.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(x: float, y: float) -> dict[str, float]:
    return {"x_inches": float(x), "y_inches": float(y)}


def _geometry(wheel: WheelPathRut, lane: LaneGeometry, bar_length: float) -> dict[str, Any]:
    assert wheel.measurement_x_inches is not None
    assert wheel.measurement_elevation_inches is not None
    half_horizontal = (bar_length / 2.0) / math.sqrt(1.0 + wheel.bar_slope**2)
    wheel_center = lane.wheel_path_center(wheel.side)
    left_reference_x = wheel_center - half_horizontal
    right_reference_x = wheel_center + half_horizontal

    def bar_y(x: float) -> float:
        return wheel.measurement_elevation_inches + wheel.bar_slope * (x - wheel.measurement_x_inches)

    return {
        "rutting_inches": wheel.rut_depth_inches,
        "left_reference_point": _point(left_reference_x, bar_y(left_reference_x)),
        "right_reference_point": _point(right_reference_x, bar_y(right_reference_x)),
        "left_contact_point": _point(wheel.left_contact_x_inches, wheel.left_contact_elevation_inches),
        "right_contact_point": _point(wheel.right_contact_x_inches, wheel.right_contact_elevation_inches),
        "measurement_point": _point(wheel.measurement_x_inches, wheel.measurement_elevation_inches),
        "rut_point": _point(wheel.rut_x_inches, wheel.rut_elevation_inches),
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pixel_width = 0.10546875
    x = np.arange(1536, dtype=np.float64) * pixel_width
    calibrated = (
        2.0 + 0.001 * x - 0.2 * np.exp(-np.square((x - 46.0) / 7.0)) - 0.3 * np.exp(-np.square((x - 116.0) / 8.0))
    )
    lane = LaneGeometry(0.0, 162.0)
    reduced = reduce_profile(TransverseProfile(x, calibrated), ReductionConfig.pathview_observed())
    cross = fit_cross_slope(reduced, lane)
    rutting = measure_profile_rutting(reduced, lane, bar_length_inches=72.0)
    assert rutting.left is not None and rutting.right is not None

    npz_path = tmp_path / "fixture.arrays.npz"
    np.savez_compressed(
        npz_path,
        profile_indices=np.array([17], dtype=np.int64),
        raw_offsets=np.array([0, x.size], dtype=np.int64),
        raw_height_u16=np.zeros(x.size, dtype=np.uint16),
        calibrated_height_inches=calibrated,
        intensity_u8=np.zeros(x.size, dtype=np.uint8),
        reduced_offsets=np.array([0, reduced.point_count], dtype=np.int64),
        reduced_x_inches=reduced.x_inches,
        reduced_y_inches=reduced.elevation_inches,
    )
    manifest: dict[str, Any] = {
        "schema": "pathview-oracle-golden",
        "schema_version": "1.0.0",
        "kind": "real_3dc_profiles",
        "public_api_contract": {
            "rut_bar_public_constants": {
                "rut_path_width_inches": 44.29134,
                "wheel_path_center_inches_from_centerline": 34.448819,
            }
        },
        "parameters": {
            "is_double_laser": False,
            "roll_degrees": 0.0,
            "max_stddev_inches_high_noise": 999.0,
            "lane_left_inches": 0.0,
            "lane_right_inches": 162.0,
            "dark_band_columns": [],
            "rut_bar_used": {"inches": 72.0},
        },
        "calibration": {"pixel_width": {"inches": pixel_width}},
        "arrays": {"file_name": npz_path.name, "sha256": _sha256(npz_path)},
        "profiles": [
            {
                "profile_index": 17,
                "raw_count": x.size,
                "npz_profile_slot": 0,
                "status": "ok",
                "error": None,
                "is_profile_noisy": reduced.is_noisy,
                "cross_slope": {
                    "percent": cross.percent,
                    "angle_degrees": cross.angle_degrees,
                },
                "rutting": {
                    "left": _geometry(rutting.left, lane, 72.0),
                    "right": _geometry(rutting.right, lane, 72.0),
                    "center": None,
                },
                "reduced_count": reduced.point_count,
            }
        ],
    }
    metadata_path = tmp_path / "fixture.metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata_path, npz_path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMPARE_TOOL), *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_comparison_passes_and_writes_strict_json(tmp_path: Path) -> None:
    metadata_path, _ = _write_fixture(tmp_path)
    report_path = tmp_path / "report.json"

    result = _run(str(metadata_path), "--json-output", str(report_path))

    assert result.returncode == 0, result.stderr
    assert "PathView golden comparison: PASS" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["summary"] == {"profiles_total": 1, "profiles_passed": 1, "profiles_failed": 0}


def test_comparison_mismatch_returns_one(tmp_path: Path) -> None:
    metadata_path, _ = _write_fixture(tmp_path)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["profiles"][0]["cross_slope"]["percent"] += 0.1
    metadata_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    result = _run(str(metadata_path))

    assert result.returncode == 1
    assert "PathView golden comparison: FAIL" in result.stdout
    assert "cross_slope_percent" in result.stdout


def test_corrupt_offsets_return_two(tmp_path: Path) -> None:
    metadata_path, npz_path = _write_fixture(tmp_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["raw_offsets"] = np.array([0, arrays["raw_height_u16"].size + 1], dtype=np.int64)
    np.savez_compressed(npz_path, **arrays)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["arrays"]["sha256"] = _sha256(npz_path)
    metadata_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    result = _run(str(metadata_path))

    assert result.returncode == 2
    assert "raw_offsets terminal offset" in result.stderr


def test_object_array_is_rejected_without_pickle(tmp_path: Path) -> None:
    metadata_path, npz_path = _write_fixture(tmp_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["unexpected"] = np.array([{"unsafe": True}], dtype=object)
    np.savez_compressed(npz_path, **arrays)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["arrays"]["sha256"] = _sha256(npz_path)
    metadata_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    result = _run(str(metadata_path))

    assert result.returncode == 2
    assert "allow_pickle=False" in result.stderr


def test_unsafe_companion_path_is_rejected(tmp_path: Path) -> None:
    metadata_path, _ = _write_fixture(tmp_path)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["arrays"]["file_name"] = "../outside.npz"
    metadata_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    result = _run(str(metadata_path))

    assert result.returncode == 2
    assert "safe relative NPZ basename" in result.stderr


def test_non_standard_json_number_is_rejected(tmp_path: Path) -> None:
    metadata_path, _ = _write_fixture(tmp_path)
    text = metadata_path.read_text(encoding="utf-8").replace('"roll_degrees": 0.0', '"roll_degrees": NaN')
    metadata_path.write_text(text, encoding="utf-8")

    result = _run(str(metadata_path))

    assert result.returncode == 2
    assert "non-standard JSON numeric constant" in result.stderr
