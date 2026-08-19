"""Parallel orchestration across numeric PathView set directories."""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pavement_rut.app.export_set import ExportSetConfig, ProcessingOptions, export_set


@dataclass(frozen=True, slots=True)
class BatchConfig:
    data_root: Path
    out_dir: Path
    sets: tuple[str, ...] | None = None
    jobs: int = 1
    index_dir: Path | None = None
    rebuild_indexes: bool = False
    limit_files_per_set: int | None = None
    progress_every: int = 25
    resume: bool = True
    checkpoint_dir: Path | None = None
    checkpoint_every: int = 1
    preview_min_severity: int | None = 2
    options: ProcessingOptions = field(default_factory=ProcessingOptions)


@dataclass(frozen=True, slots=True)
class BatchSetResult:
    set: str
    status: str
    records_exported: int
    files_partial: int
    files_failed: int
    elapsed_seconds: float | None
    geojson: str | None
    csv: str | None
    metadata_json: str | None
    error: str | None
    traceback: str | None = None


def discover_sets(data_root: Path, requested: tuple[str, ...] | None = None) -> list[Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    available = {path.name: path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()}
    if requested is None:
        selected = list(available.values())
    else:
        seen: set[str] = set()
        duplicate_labels: set[str] = set()
        for label in requested:
            if label in seen:
                duplicate_labels.add(label)
            seen.add(label)
        duplicates = sorted(duplicate_labels)
        if duplicates:
            raise ValueError(f"Requested set directories contain duplicates: {', '.join(duplicates)}")
        missing = [label for label in requested if label not in available]
        if missing:
            raise FileNotFoundError(f"Requested set directories are missing: {', '.join(missing)}")
        selected = [available[label] for label in requested]
    if not selected:
        raise ValueError(f"No numeric set directories were found under {data_root}")
    return sorted(selected, key=lambda path: (int(path.name), path.name))


def _run_set(payload: tuple[Path, Path, Path | None, Path, BatchConfig]) -> BatchSetResult:
    set_dir, set_out_dir, index_path, checkpoint_root, config = payload
    try:
        metadata = export_set(
            ExportSetConfig(
                set_dir=set_dir,
                out_dir=set_out_dir,
                index_path=index_path,
                rebuild_index=config.rebuild_indexes,
                limit_files=config.limit_files_per_set,
                jobs=1,
                progress_every=config.progress_every,
                resume=config.resume,
                checkpoint_dir=checkpoint_root,
                checkpoint_every=config.checkpoint_every,
                preview_min_severity=config.preview_min_severity,
                options=config.options,
            )
        )
        outputs = metadata["outputs"]
        files_partial = int(metadata["files_partial"])
        files_failed = int(metadata["files_failed"])
        status = "ok" if metadata["status"] == "ok" else "failed"
        return BatchSetResult(
            set=set_dir.name,
            status=status,
            records_exported=int(metadata["records_exported"]),
            files_partial=files_partial,
            files_failed=files_failed,
            elapsed_seconds=float(metadata["elapsed_seconds"]),
            geojson=str(outputs["geojson"]),
            csv=str(outputs["csv"]),
            metadata_json=str(outputs["metadata_json"]),
            error=(
                None
                if status == "ok"
                else f"Set completed partially: {files_partial} partial, {files_failed} failed files"
            ),
        )
    except Exception as exc:
        return BatchSetResult(
            set=set_dir.name,
            status="failed",
            records_exported=0,
            files_partial=0,
            files_failed=0,
            elapsed_seconds=None,
            geojson=None,
            csv=None,
            metadata_json=None,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
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


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _unexpected_set_failure(set_label: str, exc: Exception) -> BatchSetResult:
    return BatchSetResult(
        set=set_label,
        status="failed",
        records_exported=0,
        files_partial=0,
        files_failed=0,
        elapsed_seconds=None,
        geojson=None,
        csv=None,
        metadata_json=None,
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
    )


def run_batch(config: BatchConfig) -> dict[str, Any]:
    """Run selected sets with one isolated process per concurrent set."""

    if config.jobs <= 0:
        raise ValueError("jobs must be positive")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if config.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if config.limit_files_per_set is not None and config.limit_files_per_set <= 0:
        raise ValueError("limit_files_per_set must be positive")
    started = datetime.now(timezone.utc)
    sets = discover_sets(config.data_root, config.sets)
    out_dir = config.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = (
        config.index_dir.expanduser().resolve() if config.index_dir is not None else out_dir / ".cache" / "indexes"
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = (
        config.checkpoint_dir.expanduser().resolve() if config.checkpoint_dir is not None else out_dir / ".checkpoints"
    )
    payloads = [
        (
            set_dir,
            out_dir / f"set_{set_dir.name}",
            index_dir / f"set_{set_dir.name}_3dc_index.json",
            checkpoint_root,
            config,
        )
        for set_dir in sets
    ]

    results: list[BatchSetResult] = []
    if config.jobs == 1:
        for index, payload in enumerate(payloads, start=1):
            result = _run_set(payload)
            results.append(result)
            print(
                f"[BATCH] {index}/{len(payloads)} set {result.set}: {result.status}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=config.jobs) as executor:
            futures = {executor.submit(_run_set, payload): payload[0].name for payload in payloads}
            for index, future in enumerate(as_completed(futures), start=1):
                set_label = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # A child may terminate before _run_set can serialize its
                    # normal failure result. Preserve the other set outcomes
                    # and still write a non-success batch summary.
                    result = _unexpected_set_failure(set_label, exc)
                results.append(result)
                print(
                    f"[BATCH] {index}/{len(payloads)} set {result.set}: {result.status}",
                    file=sys.stderr,
                    flush=True,
                )
    results.sort(key=lambda result: (int(result.set), result.set))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    csv_path = out_dir / f"batch_summary_{timestamp}.csv"
    json_path = out_dir / f"batch_summary_{timestamp}.json"
    fieldnames = [
        "set",
        "status",
        "records_exported",
        "files_partial",
        "files_failed",
        "elapsed_seconds",
        "geojson",
        "csv",
        "metadata_json",
        "error",
    ]
    csv_temporary = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with csv_temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({name: getattr(result, name) for name in fieldnames})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(csv_temporary, csv_path)
        _fsync_directory(csv_path.parent)
    finally:
        csv_temporary.unlink(missing_ok=True)

    finished = datetime.now(timezone.utc)
    summary = {
        "schema_version": 1,
        "implementation": "pavement-rut-cross-platform",
        "created_at": finished.isoformat(timespec="seconds"),
        "elapsed_seconds": (finished - started).total_seconds(),
        "data_root": str(config.data_root.expanduser().resolve()),
        "out_dir": str(out_dir),
        "jobs": config.jobs,
        "resume_requested": config.resume,
        "checkpoint_root": str(checkpoint_root),
        "sets_total": len(results),
        "sets_ok": sum(result.status == "ok" for result in results),
        "sets_failed": sum(result.status != "ok" for result in results),
        "records_exported": sum(result.records_exported for result in results),
        "files_partial": sum(result.files_partial for result in results),
        "files_failed": sum(result.files_failed for result in results),
        "results": [asdict(result) for result in results],
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    _write_json_atomic(json_path, summary)
    return summary
