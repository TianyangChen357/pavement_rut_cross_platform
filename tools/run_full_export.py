#!/usr/bin/env python3
"""Run numeric survey sets with file-level parallelism and CPU-time reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pavement_rut.app.batch import discover_sets
from pavement_rut.app.export_set import ExportSetConfig, export_set


@dataclass(frozen=True, slots=True)
class CpuTimeMetrics:
    workers: int
    wall_seconds: float
    aggregate_cpu_seconds: float
    cpu_time_derived_speedup: float
    efficiency: float
    efficiency_percent: float


def performance_metrics(*, workers: int, wall_seconds: float, aggregate_cpu_seconds: float) -> CpuTimeMetrics:
    """Derive utilization-style speedup and efficiency from aggregate CPU time.

    ``aggregate_cpu_seconds`` is the sum of user and system CPU time consumed by
    the parent and all completed worker processes.  Dividing it by wall time is
    an observed concurrency/utilization proxy, not a measured serial-runtime
    comparison.
    """

    if workers <= 0:
        raise ValueError("workers must be positive")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise ValueError("wall time must be positive and finite")
    if not math.isfinite(aggregate_cpu_seconds) or aggregate_cpu_seconds < 0.0:
        raise ValueError("aggregate CPU time must be non-negative and finite")
    speedup = aggregate_cpu_seconds / wall_seconds
    efficiency = speedup / workers
    return CpuTimeMetrics(
        workers=workers,
        wall_seconds=wall_seconds,
        aggregate_cpu_seconds=aggregate_cpu_seconds,
        cpu_time_derived_speedup=speedup,
        efficiency=efficiency,
        efficiency_percent=efficiency * 100.0,
    )


def _aggregate_cpu_seconds() -> float:
    """Return parent plus completed-child user/system CPU seconds."""

    usage = os.times()
    return float(usage.user + usage.system + usage.children_user + usage.children_system)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _metrics_dict(*, workers: int, wall_seconds: float, cpu_seconds: float) -> dict[str, Any]:
    return asdict(
        performance_metrics(
            workers=workers,
            wall_seconds=wall_seconds,
            aggregate_cpu_seconds=cpu_seconds,
        )
    )


def _report_text(summary: dict[str, Any]) -> str:
    lines = [
        "Pavement rut full-run performance report",
        f"Status: {summary['status']}",
        f"Data root: {summary['data_root']}",
        f"Output root: {summary['out_root']}",
        f"Workers per set: {summary['workers_per_set']}",
        f"Sets completed: {summary['sets_completed']}/{summary['sets_total']}",
        f"Files exported: {summary['records_exported']}",
        f"Full-run wall time: {_duration(summary.get('full_run_elapsed_seconds'))}",
        f"Aggregate CPU time: {_duration(summary.get('aggregate_cpu_seconds'))}",
        f"Total script wall time: {_duration(summary.get('total_script_elapsed_seconds'))}",
    ]
    performance = summary.get("performance")
    if performance is not None:
        lines.extend(
            [
                "",
                "CPU-time-derived performance (no serial rerun):",
                f"Speedup proxy (aggregate CPU / wall): {performance['cpu_time_derived_speedup']:.3f}x",
                f"Efficiency (speedup / workers): {performance['efficiency_percent']:.2f}%",
                "Note: this is an observed CPU concurrency/utilization metric, not T1/TN from a serial baseline.",
            ]
        )
    lines.extend(["", "Per-set results:"])
    for result in summary["results"]:
        lines.append(
            f"  set {result['set']}: {result['status']}, files={result['records_exported']}, "
            f"wall={_duration(result['elapsed_seconds'])}, cpu={_duration(result['aggregate_cpu_seconds'])}, "
            f"speedup={result['cpu_time_derived_speedup']:.3f}x, "
            f"efficiency={result['efficiency_percent']:.2f}%, "
            f"throughput={result['files_per_second']:.3f} files/s"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process numeric sets sequentially while using multiple .3dc workers per set, "
            "with total wall time, aggregate CPU time, speedup, and efficiency reporting."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=32, help="Concurrent .3dc workers within each set")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--rebuild-indexes", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    return parser


def _refresh_summary_performance(
    summary: dict[str, Any],
    *,
    workers: int,
    full_started: float,
    full_cpu_started: float,
    script_started: float,
) -> None:
    wall_seconds = time.perf_counter() - full_started
    cpu_seconds = max(0.0, _aggregate_cpu_seconds() - full_cpu_started)
    summary["full_run_elapsed_seconds"] = wall_seconds
    summary["aggregate_cpu_seconds"] = cpu_seconds
    summary["total_script_elapsed_seconds"] = time.perf_counter() - script_started
    summary["performance"] = _metrics_dict(
        workers=workers,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
    )


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    script_started = time.perf_counter()
    data_root = args.data_root.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    requested = None if args.sets is None else tuple(str(value) for value in args.sets)
    set_dirs = discover_sets(data_root, requested)
    index_root = out_root / ".indexes"
    checkpoint_root = out_root / ".checkpoints"
    summary_path = out_root / "full_run_summary.json"
    report_path = out_root / "full_run_report.txt"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "performance_definition": (
            "CPU-time-derived speedup = aggregate parent+worker user+system CPU seconds / wall seconds; "
            "efficiency = speedup / workers. No serial baseline is run."
        ),
        "started_at": _utc_now(),
        "finished_at": None,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "workers_per_set": args.jobs,
        "checkpoint_every": args.checkpoint_every,
        "preview_min_severity": None if args.no_previews else 2,
        "sets_total": len(set_dirs),
        "sets_completed": 0,
        "sets_ok": 0,
        "sets_partial_or_failed": 0,
        "records_exported": 0,
        "full_run_elapsed_seconds": None,
        "aggregate_cpu_seconds": None,
        "total_script_elapsed_seconds": None,
        "performance": None,
        "results": [],
        "outputs": {"summary_json": str(summary_path), "report_txt": str(report_path)},
    }

    full_started = time.perf_counter()
    full_cpu_started = _aggregate_cpu_seconds()
    for position, set_dir in enumerate(set_dirs, start=1):
        print(
            f"[FULL RUN] {position}/{len(set_dirs)} set {set_dir.name} with {args.jobs} workers",
            file=sys.stderr,
            flush=True,
        )
        set_started = time.perf_counter()
        set_cpu_started = _aggregate_cpu_seconds()
        try:
            metadata = export_set(
                ExportSetConfig(
                    set_dir=set_dir,
                    out_dir=out_root / f"set_{set_dir.name}",
                    index_path=index_root / f"set_{set_dir.name}_3dc_index.json",
                    rebuild_index=args.rebuild_indexes,
                    output_name=f"set_{set_dir.name}_rut_results",
                    jobs=args.jobs,
                    progress_every=args.progress_every,
                    resume=args.resume,
                    checkpoint_dir=checkpoint_root,
                    checkpoint_every=args.checkpoint_every,
                    preview_min_severity=None if args.no_previews else 2,
                )
            )
            elapsed = time.perf_counter() - set_started
            cpu_seconds = max(0.0, _aggregate_cpu_seconds() - set_cpu_started)
            records = int(metadata["records_exported"])
            status = str(metadata["status"])
            metrics = _metrics_dict(workers=args.jobs, wall_seconds=elapsed, cpu_seconds=cpu_seconds)
            result = {
                "set": set_dir.name,
                "status": status,
                "records_exported": records,
                "files_ok": int(metadata["files_ok"]),
                "files_partial": int(metadata["files_partial"]),
                "files_failed": int(metadata["files_failed"]),
                "previews_generated": int(metadata["previews"]["generated"]),
                "elapsed_seconds": elapsed,
                "aggregate_cpu_seconds": cpu_seconds,
                "files_per_second": records / elapsed if elapsed > 0.0 else 0.0,
                "outputs": metadata["outputs"],
                "error": None,
                **{
                    "cpu_time_derived_speedup": metrics["cpu_time_derived_speedup"],
                    "efficiency": metrics["efficiency"],
                    "efficiency_percent": metrics["efficiency_percent"],
                },
            }
        except Exception as exc:
            elapsed = time.perf_counter() - set_started
            cpu_seconds = max(0.0, _aggregate_cpu_seconds() - set_cpu_started)
            metrics = _metrics_dict(workers=args.jobs, wall_seconds=elapsed, cpu_seconds=cpu_seconds)
            result = {
                "set": set_dir.name,
                "status": "failed",
                "records_exported": 0,
                "files_ok": 0,
                "files_partial": 0,
                "files_failed": 0,
                "previews_generated": 0,
                "elapsed_seconds": elapsed,
                "aggregate_cpu_seconds": cpu_seconds,
                "files_per_second": 0.0,
                "outputs": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "cpu_time_derived_speedup": metrics["cpu_time_derived_speedup"],
                "efficiency": metrics["efficiency"],
                "efficiency_percent": metrics["efficiency_percent"],
            }
        summary["results"].append(result)
        summary["sets_completed"] = len(summary["results"])
        summary["sets_ok"] = sum(item["status"] == "ok" for item in summary["results"])
        summary["sets_partial_or_failed"] = sum(item["status"] != "ok" for item in summary["results"])
        summary["records_exported"] = sum(item["records_exported"] for item in summary["results"])
        _refresh_summary_performance(
            summary,
            workers=args.jobs,
            full_started=full_started,
            full_cpu_started=full_cpu_started,
            script_started=script_started,
        )
        _write_json_atomic(summary_path, summary)
        _write_text_atomic(report_path, _report_text(summary))

    _refresh_summary_performance(
        summary,
        workers=args.jobs,
        full_started=full_started,
        full_cpu_started=full_cpu_started,
        script_started=script_started,
    )
    summary["finished_at"] = _utc_now()
    summary["status"] = "ok" if summary["sets_partial_or_failed"] == 0 else "partial"
    _write_json_atomic(summary_path, summary)
    _write_text_atomic(report_path, _report_text(summary))
    return summary, 0 if summary["status"] == "ok" else 1


def main(argv: list[str] | None = None) -> None:
    try:
        summary, code = run(_parser().parse_args(argv))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary["outputs"], indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
