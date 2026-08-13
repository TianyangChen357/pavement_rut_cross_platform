"""End-to-end cross-platform processing for one survey set."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pavement_rut.app.checkpoint import (
    CheckpointStore,
    RecordIdentity,
    build_checkpoint_manifest,
)
from pavement_rut.domain.aggregate import aggregate_rutting
from pavement_rut.domain.cross_slope import fit_cross_slope
from pavement_rut.domain.models import LaneGeometry, TransverseProfile
from pavement_rut.domain.reduction import ReductionConfig, reduce_profile
from pavement_rut.domain.rutbar import DEFAULT_RUT_BAR_LENGTH_INCHES, measure_profile_rutting
from pavement_rut.index import ImageRecord, get_or_build_index
from pavement_rut.io.calibration import load_calibration
from pavement_rut.io.three_dc import read_3dc
from pavement_rut.navigation import NavigationLookup
from pavement_rut.severity import finite_mean


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    lane_left_inches: float = 0.0
    lane_right_inches: float | None = None
    lane_center_offset_inches: float = 0.0
    rut_bar_length_inches: float = DEFAULT_RUT_BAR_LENGTH_INCHES
    roll_degrees: float = 0.0
    invalid_policy: str = "interpolate"
    minimum_valid_fraction: float = 0.5
    maximum_interpolation_gap_inches: float | None = None
    maximum_noise_std_inches: float | None = 999.0
    skip_noisy_profiles: bool = False

    def __post_init__(self) -> None:
        finite_values = {
            "lane_left_inches": self.lane_left_inches,
            "lane_center_offset_inches": self.lane_center_offset_inches,
            "rut_bar_length_inches": self.rut_bar_length_inches,
            "roll_degrees": self.roll_degrees,
            "minimum_valid_fraction": self.minimum_valid_fraction,
        }
        if self.lane_right_inches is not None:
            finite_values["lane_right_inches"] = self.lane_right_inches
        if self.maximum_interpolation_gap_inches is not None:
            finite_values["maximum_interpolation_gap_inches"] = self.maximum_interpolation_gap_inches
        if self.maximum_noise_std_inches is not None:
            finite_values["maximum_noise_std_inches"] = self.maximum_noise_std_inches
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.lane_right_inches is not None and self.lane_right_inches <= self.lane_left_inches:
            raise ValueError("lane_right_inches must be greater than lane_left_inches")
        if self.rut_bar_length_inches <= 0.0:
            raise ValueError("rut_bar_length_inches must be positive")
        if self.invalid_policy not in {"interpolate", "drop", "raise"}:
            raise ValueError(f"unsupported invalid_policy: {self.invalid_policy!r}")
        if not 0.0 <= self.minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be between zero and one")
        if self.maximum_interpolation_gap_inches is not None and self.maximum_interpolation_gap_inches <= 0.0:
            raise ValueError("maximum_interpolation_gap_inches must be positive")
        if self.maximum_noise_std_inches is not None and self.maximum_noise_std_inches < 0.0:
            raise ValueError("maximum_noise_std_inches cannot be negative")


@dataclass(frozen=True, slots=True)
class ExportSetConfig:
    set_dir: Path
    out_dir: Path
    calibration_path: Path | None = None
    index_path: Path | None = None
    rebuild_index: bool = False
    from_frame: float | None = None
    to_frame: float | None = None
    limit_files: int | None = None
    output_name: str | None = None
    jobs: int = 1
    progress_every: int = 25
    resume: bool = True
    checkpoint_dir: Path | None = None
    checkpoint_every: int = 1
    options: ProcessingOptions = field(default_factory=ProcessingOptions)


@dataclass(frozen=True, slots=True)
class FileRutResult:
    relative_path: str
    file_name: str
    start_frame: float
    end_frame: float
    left_average_inches: float
    right_average_inches: float
    overall_average_inches: float
    cross_slope_average_percent: float
    cross_slope_average_angle_degrees: float
    severity: int
    rows_total: int
    rows_ok: int
    rows_error: int
    rows_noisy: int
    left_count: int
    right_count: int
    cross_slope_count: int
    cross_slope_error_count: int
    error_samples: tuple[str, ...] = ()
    status: str = "ok"
    file_error: str | None = None


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _validated_output_base(value: str) -> str:
    """Return a filename-only output stem and reject path traversal."""

    candidate = value.strip()
    if not candidate or "/" in candidate or "\\" in candidate:
        raise ValueError("output_name must be a non-empty filename, not a path")
    for suffix in (".metadata.json", ".geojson", ".csv"):
        candidate = candidate.removesuffix(suffix)
    if not candidate or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", candidate) is None:
        raise ValueError(
            "output_name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens"
        )
    windows_stem = candidate.split(".", 1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
    reserved.update(f"COM{index}" for index in range(1, 10))
    reserved.update(f"LPT{index}" for index in range(1, 10))
    if windows_stem in reserved:
        raise ValueError(f"output_name is reserved on Windows: {candidate!r}")
    return candidate


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _checkpoint_result_payload(result: FileRutResult) -> dict[str, Any]:
    payload = _json_safe(asdict(result))
    if not isinstance(payload, dict):  # pragma: no cover - asdict is guaranteed here
        raise TypeError("FileRutResult did not serialize to an object")
    return payload


def _checkpoint_float(payload: dict[str, Any], name: str, *, nullable: bool) -> float:
    value = payload[name]
    if value is None and nullable:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Checkpoint result field {name!r} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Checkpoint result field {name!r} must be finite or null")
    return converted


def _checkpoint_int(payload: dict[str, Any], name: str) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Checkpoint result field {name!r} must be an integer")
    return value


def _file_result_from_checkpoint(payload: dict[str, Any]) -> FileRutResult:
    expected = {
        "relative_path",
        "file_name",
        "start_frame",
        "end_frame",
        "left_average_inches",
        "right_average_inches",
        "overall_average_inches",
        "cross_slope_average_percent",
        "cross_slope_average_angle_degrees",
        "severity",
        "rows_total",
        "rows_ok",
        "rows_error",
        "rows_noisy",
        "left_count",
        "right_count",
        "cross_slope_count",
        "cross_slope_error_count",
        "error_samples",
        "status",
        "file_error",
    }
    if set(payload) != expected:
        raise ValueError("Checkpoint FileRutResult fields do not match schema")
    if not isinstance(payload["relative_path"], str) or not isinstance(payload["file_name"], str):
        raise ValueError("Checkpoint result paths must be strings")
    if payload["status"] not in {"ok", "partial", "failed"}:
        raise ValueError("Checkpoint result status is invalid")
    file_error = payload["file_error"]
    if file_error is not None and not isinstance(file_error, str):
        raise ValueError("Checkpoint result file_error must be a string or null")
    errors = payload["error_samples"]
    if not isinstance(errors, list) or not all(isinstance(item, str) for item in errors):
        raise ValueError("Checkpoint result error_samples must be an array of strings")

    counts = {
        name: _checkpoint_int(payload, name)
        for name in (
            "rows_total",
            "rows_ok",
            "rows_error",
            "rows_noisy",
            "left_count",
            "right_count",
            "cross_slope_count",
            "cross_slope_error_count",
        )
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("Checkpoint result counts cannot be negative")
    severity = _checkpoint_int(payload, "severity")
    if severity not in {-1, 0, 1, 2, 3}:
        raise ValueError("Checkpoint result severity is invalid")
    return FileRutResult(
        relative_path=payload["relative_path"],
        file_name=payload["file_name"],
        start_frame=_checkpoint_float(payload, "start_frame", nullable=False),
        end_frame=_checkpoint_float(payload, "end_frame", nullable=False),
        left_average_inches=_checkpoint_float(payload, "left_average_inches", nullable=True),
        right_average_inches=_checkpoint_float(payload, "right_average_inches", nullable=True),
        overall_average_inches=_checkpoint_float(payload, "overall_average_inches", nullable=True),
        cross_slope_average_percent=_checkpoint_float(payload, "cross_slope_average_percent", nullable=True),
        cross_slope_average_angle_degrees=_checkpoint_float(
            payload, "cross_slope_average_angle_degrees", nullable=True
        ),
        severity=severity,
        **counts,
        error_samples=tuple(errors),
        status=payload["status"],
        file_error=file_error,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
    )


def _x_coordinates(calibration: Any, columns: int) -> np.ndarray:
    """Return the raw lateral coordinate grid while tolerating API evolution."""

    if hasattr(calibration, "x_inches"):
        values = np.asarray(calibration.x_inches(), dtype=np.float64)
    elif hasattr(calibration, "x_coordinates_inches"):
        values = np.asarray(calibration.x_coordinates_inches(), dtype=np.float64)
    else:
        values = np.arange(columns, dtype=np.float64) * float(calibration.pixel_width_inches)
    if values.shape != (columns,):
        raise ValueError(f"Calibration produced x shape {values.shape}; expected {(columns,)}")
    return values


def _process_file(
    set_dir: Path,
    calibration_path: Path,
    record: ImageRecord,
    options: ProcessingOptions,
) -> FileRutResult:
    """Worker-safe processing of one image file."""

    calibration = load_calibration(calibration_path)
    path = record.resolve(set_dir)
    image = read_3dc(path)
    calibrated = calibration.apply(image.raw_heights)
    x_inches = _x_coordinates(calibration, image.columns)
    lane_right = (
        float(options.lane_right_inches)
        if options.lane_right_inches is not None
        else float(calibration.pavement_width_inches)
    )
    lane = LaneGeometry(
        left_edge_inches=float(options.lane_left_inches),
        right_edge_inches=lane_right,
        center_offset_inches=float(options.lane_center_offset_inches),
    )
    reduction = ReductionConfig.pathview_observed(
        invalid_policy=options.invalid_policy,  # type: ignore[arg-type]
        max_interpolation_gap_inches=options.maximum_interpolation_gap_inches,
        minimum_valid_fraction=options.minimum_valid_fraction,
        roll_degrees=options.roll_degrees,
        max_stddev_inches_high_noise=options.maximum_noise_std_inches,
    )

    profile_results = []
    errors: list[str] = []
    rows_ok = 0
    rows_error = 0
    rows_noisy = 0
    cross_slope_percentages: list[float] = []
    cross_slope_angles_degrees: list[float] = []
    cross_slope_error_count = 0
    for row_index, elevations in enumerate(calibrated):
        try:
            raw_profile = TransverseProfile(
                x_inches=x_inches,
                elevation_inches=elevations,
                profile_id=f"{path.name}:{row_index}",
            )
            reduced = reduce_profile(raw_profile, reduction)
            if reduced.is_noisy:
                rows_noisy += 1
                if options.skip_noisy_profiles:
                    profile_results.append(None)
                    continue
            try:
                cross_slope = fit_cross_slope(reduced, lane)
                if math.isfinite(cross_slope.percent) and math.isfinite(cross_slope.angle_degrees):
                    cross_slope_percentages.append(float(cross_slope.percent))
                    cross_slope_angles_degrees.append(float(cross_slope.angle_degrees))
                else:
                    raise ArithmeticError("cross-slope result is non-finite")
            except (ArithmeticError, ValueError) as exc:
                cross_slope_error_count += 1
                if len(errors) < 10:
                    errors.append(f"row {row_index} cross slope: {type(exc).__name__}: {exc}")
            result = measure_profile_rutting(
                reduced,
                lane,
                bar_length_inches=options.rut_bar_length_inches,
            )
            profile_results.append(result)
            rows_ok += 1
        except (ArithmeticError, ValueError) as exc:
            # Expected data/geometry failures are isolated per profile.  Do not
            # swallow TypeError, AttributeError, or other programming defects.
            profile_results.append(None)
            rows_error += 1
            if len(errors) < 10:
                errors.append(f"row {row_index}: {type(exc).__name__}: {exc}")

    aggregate = aggregate_rutting(profile_results)
    cross_slope_average_percent = finite_mean(cross_slope_percentages)
    cross_slope_average_angle_degrees = finite_mean(cross_slope_angles_degrees)
    has_complete_measurement = (
        math.isfinite(aggregate.left_average_inches)
        and math.isfinite(aggregate.right_average_inches)
        and math.isfinite(aggregate.overall_average_inches)
    )
    if not has_complete_measurement:
        status = "failed"
    elif rows_error or cross_slope_error_count or (options.skip_noisy_profiles and rows_noisy):
        status = "partial"
    else:
        status = "ok"
    return FileRutResult(
        relative_path=record.relative_path,
        file_name=path.name,
        start_frame=record.start_frame,
        end_frame=record.end_frame,
        left_average_inches=aggregate.left_average_inches,
        right_average_inches=aggregate.right_average_inches,
        overall_average_inches=aggregate.overall_average_inches,
        cross_slope_average_percent=cross_slope_average_percent,
        cross_slope_average_angle_degrees=cross_slope_average_angle_degrees,
        severity=-1 if aggregate.severity is None else int(aggregate.severity),
        rows_total=image.profile_count,
        rows_ok=rows_ok,
        rows_error=rows_error,
        rows_noisy=rows_noisy,
        left_count=aggregate.left_count,
        right_count=aggregate.right_count,
        cross_slope_count=len(cross_slope_percentages),
        cross_slope_error_count=cross_slope_error_count,
        error_samples=tuple(errors),
        status=status,
    )


def _worker(payload: tuple[Path, Path, ImageRecord, ProcessingOptions]) -> FileRutResult:
    return _process_file(*payload)


def _failed_file_result(record: ImageRecord, exc: Exception) -> FileRutResult:
    error = f"{type(exc).__name__}: {exc}"
    return FileRutResult(
        relative_path=record.relative_path,
        file_name=Path(record.relative_path).name,
        start_frame=record.start_frame,
        end_frame=record.end_frame,
        left_average_inches=float("nan"),
        right_average_inches=float("nan"),
        overall_average_inches=float("nan"),
        cross_slope_average_percent=float("nan"),
        cross_slope_average_angle_degrees=float("nan"),
        severity=-1,
        rows_total=0,
        rows_ok=0,
        rows_error=0,
        rows_noisy=0,
        left_count=0,
        right_count=0,
        cross_slope_count=0,
        cross_slope_error_count=0,
        error_samples=(error,),
        status="failed",
        file_error=error,
    )


def _selected_records(records: list[ImageRecord], config: ExportSetConfig) -> list[ImageRecord]:
    selected = records
    if config.from_frame is not None:
        selected = [record for record in selected if record.start_frame >= config.from_frame]
    if config.to_frame is not None:
        selected = [record for record in selected if record.start_frame <= config.to_frame]
    if config.limit_files is not None:
        if config.limit_files <= 0:
            raise ValueError("limit_files must be positive")
        selected = selected[: config.limit_files]
    if not selected:
        raise ValueError("No .3dc files were selected")
    return selected


def _process_records(
    set_dir: Path,
    calibration_path: Path,
    records: list[ImageRecord],
    options: ProcessingOptions,
    *,
    jobs: int,
    progress_every: int,
    on_result: Callable[[int, ImageRecord, FileRutResult], None] | None = None,
) -> list[FileRutResult]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if not records:
        return []

    completed = 0
    ordered: list[FileRutResult | None] = [None] * len(records)

    def accept(position: int, record: ImageRecord, result: FileRutResult) -> None:
        nonlocal completed
        ordered[position] = result
        if on_result is not None:
            on_result(position, record, result)
        completed += 1
        if completed % progress_every == 0 or completed == len(records):
            print(
                f"[INFO] Processed {completed}/{len(records)} .3dc files",
                file=sys.stderr,
                flush=True,
            )

    recoverable = (OSError, ValueError, ArithmeticError)
    if jobs == 1:
        for position, record in enumerate(records):
            try:
                result = _worker((set_dir, calibration_path, record, options))
            except recoverable as exc:
                result = _failed_file_result(record, exc)
            accept(position, record, result)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        record_iterator = iter(enumerate(records))
        futures: dict[Any, tuple[int, ImageRecord]] = {}

        def submit_next() -> bool:
            try:
                position, record = next(record_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _worker,
                (set_dir, calibration_path, record, options),
            )
            futures[future] = (position, record)
            return True

        try:
            for _ in range(min(len(records), max(jobs, 2 * jobs))):
                submit_next()
            while futures:
                future = next(as_completed(tuple(futures)))
                position, record = futures.pop(future)
                try:
                    result = future.result()
                except recoverable as exc:
                    result = _failed_file_result(record, exc)
                accept(position, record, result)
                submit_next()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    if any(result is None for result in ordered):  # pragma: no cover - defensive invariant
        raise RuntimeError("Not every selected .3dc file produced a result")
    return [result for result in ordered if result is not None]


def export_set(config: ExportSetConfig) -> dict[str, Any]:
    """Process one set and write GeoJSON, CSV, metadata, and resumable checkpoints."""

    started = datetime.now(timezone.utc)
    validated_output_base = _validated_output_base(config.output_name) if config.output_name is not None else None
    if config.jobs <= 0:
        raise ValueError("jobs must be positive")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    set_dir = config.set_dir.expanduser().resolve()
    out_dir = config.out_dir.expanduser().resolve()
    if not set_dir.is_dir():
        raise FileNotFoundError(f"Set directory does not exist: {set_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = (
        config.calibration_path.expanduser().resolve()
        if config.calibration_path is not None
        else set_dir / "3D_Camera.cal"
    )
    calibration = load_calibration(calibration_path)
    lane_right = (
        float(config.options.lane_right_inches)
        if config.options.lane_right_inches is not None
        else float(calibration.pavement_width_inches)
    )
    if lane_right <= config.options.lane_left_inches:
        raise ValueError("lane_right_inches must be greater than lane_left_inches")

    checkpoint_root = (
        config.checkpoint_dir.expanduser().resolve() if config.checkpoint_dir is not None else out_dir / ".checkpoints"
    )
    checkpoint_manifest = build_checkpoint_manifest(
        set_label=set_dir.name,
        set_dir=set_dir,
        calibration_path=calibration_path,
        options=asdict(config.options),
    )
    index_path = (
        config.index_path.expanduser().resolve()
        if config.index_path is not None
        else out_dir / ".cache" / f"set_{_safe_stem(set_dir.name)}_3dc_index.json"
    )

    with CheckpointStore(
        root=checkpoint_root,
        set_label=set_dir.name,
        manifest=checkpoint_manifest,
        resume=config.resume,
        fsync_every=config.checkpoint_every,
    ) as checkpoint:
        indexed_records = get_or_build_index(set_dir, index_path, rebuild=config.rebuild_index)
        records = _selected_records(indexed_records, config)
        file_results: list[FileRutResult | None] = [None] * len(records)
        identities: list[RecordIdentity | None] = []
        identity_failures = 0
        for position, record in enumerate(records):
            try:
                identities.append(RecordIdentity.from_record(set_dir, record))
            except OSError as exc:
                identities.append(None)
                file_results[position] = _failed_file_result(record, exc)
                identity_failures += 1
        pending_records: list[ImageRecord] = []
        pending_identities: list[RecordIdentity] = []
        pending_original_positions: list[int] = []
        records_reused = 0

        for position, (record, identity) in enumerate(zip(records, identities, strict=True)):
            if identity is None:
                continue
            entry = checkpoint.entries.get(identity.key)
            if entry is None:
                pending_records.append(record)
                pending_identities.append(identity)
                pending_original_positions.append(position)
                continue
            result = _file_result_from_checkpoint(entry.result)
            if (
                result.relative_path != identity.relative_path
                or result.start_frame != identity.start_frame
                or result.end_frame != identity.end_frame
            ):
                raise ValueError(f"Checkpoint result does not match source identity: {identity.relative_path}")
            # Keep incomplete results for audit, but retry them on every invocation.
            if result.status != "ok":
                pending_records.append(record)
                pending_identities.append(identity)
                pending_original_positions.append(position)
                continue
            file_results[position] = result
            records_reused += 1

        print(
            f"[INFO] Selected {len(records)} of {len(indexed_records)} indexed files; "
            f"resuming {records_reused}, processing {len(pending_records)}, "
            f"preflight failures {identity_failures}",
            file=sys.stderr,
            flush=True,
        )

        def checkpoint_result(
            pending_position: int,
            record: ImageRecord,
            result: FileRutResult,
        ) -> None:
            identity = pending_identities[pending_position]
            if record.relative_path != identity.relative_path:
                raise RuntimeError("Checkpoint callback record order changed unexpectedly")
            try:
                current_identity = RecordIdentity.from_record(set_dir, record)
            except OSError as exc:
                raise RuntimeError(
                    f"Source file became unavailable while it was processed: {record.relative_path}"
                ) from exc
            if current_identity != identity:
                raise RuntimeError(
                    f"Source file changed while it was processed; result was not checkpointed: {record.relative_path}"
                )
            checkpoint.append(identity, _checkpoint_result_payload(result))
            file_results[pending_original_positions[pending_position]] = result

        _process_records(
            set_dir,
            calibration_path,
            pending_records,
            config.options,
            jobs=config.jobs,
            progress_every=config.progress_every,
            on_result=checkpoint_result,
        )
        if any(result is None for result in file_results):  # pragma: no cover
            raise RuntimeError("Not every selected .3dc file has a checkpointed result")
        completed_results = [result for result in file_results if result is not None]

        navigation = NavigationLookup(set_dir)
        properties: list[dict[str, Any]] = []
        features: list[dict[str, Any]] = []
        severity_counts: dict[str, int] = {}
        for result in completed_results:
            nav = navigation(result.start_frame)
            row = {
                "set": set_dir.name,
                "file_name": result.file_name,
                "relative_path": result.relative_path,
                "starting_frame_number": result.start_frame,
                "latitude": _json_number(nav.latitude),
                "longitude": _json_number(nav.longitude),
                "heading": _json_number(nav.heading),
                "averaged_left_rut": _json_number(result.left_average_inches),
                "averaged_right_rut": _json_number(result.right_average_inches),
                "averaged_rut": _json_number(result.overall_average_inches),
                "cross_slope_average_percent": _json_number(result.cross_slope_average_percent),
                "cross_slope_average_angle_degrees": _json_number(result.cross_slope_average_angle_degrees),
                "cross_slope_count": result.cross_slope_count,
                "cross_slope_error_count": result.cross_slope_error_count,
                "severity": result.severity,
            }
            properties.append(row)
            severity_counts[str(result.severity)] = severity_counts.get(str(result.severity), 0) + 1
            geometry = None
            if math.isfinite(nav.longitude) and math.isfinite(nav.latitude):
                geometry = {"type": "Point", "coordinates": [nav.longitude, nav.latitude]}
            features.append({"type": "Feature", "geometry": geometry, "properties": row})

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        base_name = validated_output_base or f"set_{_safe_stem(set_dir.name)}_average_rut_6ft_{timestamp}"
        geojson_path = out_dir / f"{base_name}.geojson"
        csv_path = out_dir / f"{base_name}.csv"
        metadata_path = out_dir / f"{base_name}.metadata.json"
        _write_json_atomic(
            geojson_path,
            {"type": "FeatureCollection", "name": base_name, "features": features},
        )
        csv_temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with csv_temporary.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(properties[0]))
                writer.writeheader()
                writer.writerows(properties)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(csv_temporary, csv_path)
            _fsync_directory(csv_path.parent)
        finally:
            csv_temporary.unlink(missing_ok=True)

        files_ok = sum(result.status == "ok" for result in completed_results)
        files_partial = sum(result.status == "partial" for result in completed_results)
        files_failed = sum(result.status == "failed" for result in completed_results)
        export_status = "ok" if files_partial == 0 and files_failed == 0 else "partial"
        finished = datetime.now(timezone.utc)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "implementation": "pavement-rut-cross-platform",
            "algorithm": "independent 6-ft virtual straightedge implementation",
            "measurement_units": {
                "rut_depth": "inch",
                "cross_slope_average_percent": "percent",
                "cross_slope_average_angle_degrees": "degree",
            },
            "status": export_status,
            "created_at": finished.isoformat(timespec="seconds"),
            "elapsed_seconds": (finished - started).total_seconds(),
            "host": {"platform": os.name, "process_id": os.getpid()},
            "set": set_dir.name,
            "set_dir": str(set_dir),
            "calibration": str(calibration_path),
            "index_json": str(index_path),
            "navigation_sources": navigation.metadata(),
            "options": asdict(config.options),
            "lane_positions_inches": {
                "left": config.options.lane_left_inches,
                "right": lane_right,
                "center": (config.options.lane_left_inches + lane_right) / 2.0
                + config.options.lane_center_offset_inches,
            },
            "records_indexed": len(indexed_records),
            "records_exported": len(completed_results),
            "files_ok": files_ok,
            "files_partial": files_partial,
            "files_failed": files_failed,
            "severity_counts": severity_counts,
            "diagnostics": [asdict(result) for result in completed_results],
            "checkpoint": {
                "resume_requested": config.resume,
                "fingerprint": checkpoint_manifest["fingerprint"],
                "root": str(checkpoint_root),
                "run_dir": str(checkpoint.run_dir),
                "manifest_json": str(checkpoint.manifest_path),
                "journal_jsonl": str(checkpoint.journal_path),
                "records_reused": records_reused,
                "records_processed": len(pending_records) + identity_failures,
                "records_not_checkpointed": identity_failures,
                "fsync_every": config.checkpoint_every,
            },
            "outputs": {
                "geojson": str(geojson_path),
                "csv": str(csv_path),
                "metadata_json": str(metadata_path),
            },
            "compatibility_notice": (
                "Independent implementation; validate against approved reference outputs. See docs/compatibility.md."
            ),
        }
        _write_json_atomic(metadata_path, metadata)
    return metadata
