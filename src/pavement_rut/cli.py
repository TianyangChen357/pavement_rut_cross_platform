"""Command-line interface for cross-platform rut-depth processing."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from pavement_rut import __version__
from pavement_rut.app.batch import BatchConfig, run_batch
from pavement_rut.app.export_set import ExportSetConfig, ProcessingOptions, export_set
from pavement_rut.domain.cross_slope import fit_cross_slope
from pavement_rut.domain.models import LaneGeometry, TransverseProfile
from pavement_rut.domain.reduction import ReductionConfig, reduce_profile
from pavement_rut.domain.rutbar import measure_profile_rutting
from pavement_rut.io.calibration import load_calibration
from pavement_rut.io.three_dc import read_3dc


def _add_processing_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lane-left-inches", type=float, default=0.0)
    parser.add_argument("--lane-right-inches", type=float, default=None)
    parser.add_argument("--lane-center-offset-inches", type=float, default=0.0)
    parser.add_argument("--rut-bar-inches", type=float, default=72.0)
    parser.add_argument("--roll-degrees", type=float, default=0.0)
    parser.add_argument("--invalid-policy", choices=("interpolate", "drop", "raise"), default="interpolate")
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.5)
    parser.add_argument("--max-interpolation-gap-inches", type=float, default=None)
    parser.add_argument("--max-noise-std-inches", type=float, default=None)
    parser.add_argument("--skip-noisy-profiles", action="store_true")


def _add_checkpoint_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore reusable file results and start a fresh checkpoint journal",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint root (default: OUT_DIR/.checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Fsync the append-only journal after this many completed files",
    )


def _add_preview_options(parser: argparse.ArgumentParser) -> None:
    previews = parser.add_mutually_exclusive_group()
    previews.add_argument(
        "--preview-min-severity",
        type=int,
        choices=(2, 3),
        default=2,
        help="Write calibrated-height grayscale PNG previews at or above this severity (default: 2)",
    )
    previews.add_argument(
        "--no-previews",
        action="store_const",
        const=None,
        dest="preview_min_severity",
        help="Disable automatic severity-triggered PNG previews",
    )


def _processing_options(args: argparse.Namespace) -> ProcessingOptions:
    return ProcessingOptions(
        lane_left_inches=args.lane_left_inches,
        lane_right_inches=args.lane_right_inches,
        lane_center_offset_inches=args.lane_center_offset_inches,
        rut_bar_length_inches=args.rut_bar_inches,
        roll_degrees=args.roll_degrees,
        invalid_policy=args.invalid_policy,
        minimum_valid_fraction=args.minimum_valid_fraction,
        maximum_interpolation_gap_inches=args.max_interpolation_gap_inches,
        maximum_noise_std_inches=(999.0 if args.max_noise_std_inches is None else args.max_noise_std_inches),
        skip_noisy_profiles=args.skip_noisy_profiles,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pavement-rut",
        description="Cross-platform .3dc decoding and 6-ft pavement rut-depth processing.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-file", help="Decode and summarize one .3dc file")
    inspect_parser.add_argument("file", type=Path)
    inspect_parser.add_argument("--calibration", type=Path, required=True)
    inspect_parser.add_argument("--profile-index", type=int, default=0)
    inspect_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    _add_processing_options(inspect_parser)

    export_parser = subparsers.add_parser("export-set", help="Process one numeric survey set")
    source = export_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--set-dir", type=Path)
    source.add_argument("--data-root", type=Path)
    export_parser.add_argument("--set", dest="set_number")
    export_parser.add_argument("--calibration", type=Path, default=None)
    export_parser.add_argument("--out-dir", type=Path, required=True)
    export_parser.add_argument("--index-json", type=Path, default=None)
    export_parser.add_argument("--rebuild-index", action="store_true")
    export_parser.add_argument("--from-frame", type=float, default=None)
    export_parser.add_argument("--to-frame", type=float, default=None)
    export_parser.add_argument("--limit-files", type=int, default=None)
    export_parser.add_argument("--output-name", default=None)
    export_parser.add_argument("--jobs", type=int, default=1, help="Concurrent .3dc files")
    export_parser.add_argument("--progress-every", type=int, default=25)
    _add_checkpoint_options(export_parser)
    _add_preview_options(export_parser)
    _add_processing_options(export_parser)

    batch_parser = subparsers.add_parser("batch", help="Process numeric set directories in parallel")
    batch_parser.add_argument("--data-root", type=Path, required=True)
    batch_parser.add_argument("--out-dir", type=Path, required=True)
    batch_parser.add_argument("--sets", nargs="+", default=None)
    batch_parser.add_argument("--jobs", type=int, default=1, help="Concurrent set processes")
    batch_parser.add_argument("--index-dir", type=Path, default=None)
    batch_parser.add_argument("--rebuild-indexes", action="store_true")
    batch_parser.add_argument("--limit-files", type=int, default=None, help="Limit files in every set")
    batch_parser.add_argument("--progress-every", type=int, default=25)
    _add_checkpoint_options(batch_parser)
    _add_preview_options(batch_parser)
    _add_processing_options(batch_parser)
    return parser


def _inspect(args: argparse.Namespace) -> int:
    source_path = args.file.expanduser().resolve()
    calibration = load_calibration(args.calibration.expanduser().resolve())
    image = read_3dc(source_path)
    if not 0 <= args.profile_index < image.profile_count:
        raise IndexError(f"profile-index must be between 0 and {image.profile_count - 1}")
    raw_heights = image.raw_heights[args.profile_index]
    elevations = calibration.apply(raw_heights)
    x = np.arange(image.columns, dtype=np.float64) * calibration.pixel_width_inches
    lane_right = calibration.pavement_width_inches if args.lane_right_inches is None else args.lane_right_inches
    reduced = reduce_profile(
        TransverseProfile(x, elevations, profile_id=f"{source_path.name}:{args.profile_index}"),
        ReductionConfig.pathview_observed(
            invalid_policy=args.invalid_policy,
            max_interpolation_gap_inches=args.max_interpolation_gap_inches,
            minimum_valid_fraction=args.minimum_valid_fraction,
            roll_degrees=args.roll_degrees,
            max_stddev_inches_high_noise=(999.0 if args.max_noise_std_inches is None else args.max_noise_std_inches),
        ),
    )
    lane = LaneGeometry(
        left_edge_inches=args.lane_left_inches,
        right_edge_inches=lane_right,
        center_offset_inches=args.lane_center_offset_inches,
    )
    rutting = measure_profile_rutting(
        reduced,
        lane,
        bar_length_inches=args.rut_bar_inches,
    )
    cross_slope = fit_cross_slope(reduced, lane)

    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    result = {
        "file": str(source_path),
        "compressed_bytes": source_path.stat().st_size,
        "profiles": image.profile_count,
        "columns": image.columns,
        "profile_index": args.profile_index,
        "raw_height_min": int(np.min(raw_heights)),
        "raw_height_max": int(np.max(raw_heights)),
        "calibrated_height_min_inches": finite_or_none(float(np.nanmin(elevations))),
        "calibrated_height_max_inches": finite_or_none(float(np.nanmax(elevations))),
        "reduced_points": reduced.point_count,
        "is_noisy": reduced.is_noisy,
        "cross_slope_percent": finite_or_none(cross_slope.percent),
        "cross_slope_angle_degrees": finite_or_none(cross_slope.angle_degrees),
        "left_rut_inches": (None if rutting.left is None else finite_or_none(rutting.left.rut_depth_inches)),
        "right_rut_inches": (None if rutting.right is None else finite_or_none(rutting.right.rut_depth_inches)),
        "overall_rut_inches": finite_or_none(rutting.overall_rut_depth_inches),
    }
    print(json.dumps(result, indent=None if args.json else 2, allow_nan=False))
    return 0


def _resolve_set_dir(args: argparse.Namespace) -> Path:
    if args.set_dir is not None:
        if args.set_number is not None:
            raise ValueError("--set is only valid with --data-root")
        return args.set_dir
    if args.set_number is None:
        raise ValueError("--data-root requires --set")
    label = str(args.set_number)
    if re.fullmatch(r"[0-9]+", label) is None:
        raise ValueError("--set must be a numeric set directory name")
    data_root = args.data_root.expanduser().resolve()
    set_dir = (data_root / label).resolve()
    if set_dir.parent != data_root:  # Defensive invariant if path rules change later.
        raise ValueError("--set must resolve directly below --data-root")
    return set_dir


def _export(args: argparse.Namespace) -> int:
    metadata = export_set(
        ExportSetConfig(
            set_dir=_resolve_set_dir(args),
            out_dir=args.out_dir,
            calibration_path=args.calibration,
            index_path=args.index_json,
            rebuild_index=args.rebuild_index,
            from_frame=args.from_frame,
            to_frame=args.to_frame,
            limit_files=args.limit_files,
            output_name=args.output_name,
            jobs=args.jobs,
            progress_every=args.progress_every,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            preview_min_severity=args.preview_min_severity,
            options=_processing_options(args),
        )
    )
    print(json.dumps(metadata["outputs"], indent=2))
    return 0 if metadata["status"] == "ok" else 1


def _batch(args: argparse.Namespace) -> int:
    summary = run_batch(
        BatchConfig(
            data_root=args.data_root,
            out_dir=args.out_dir,
            sets=None if args.sets is None else tuple(str(value) for value in args.sets),
            jobs=args.jobs,
            index_dir=args.index_dir,
            rebuild_indexes=args.rebuild_indexes,
            limit_files_per_set=args.limit_files,
            progress_every=args.progress_every,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            preview_min_severity=args.preview_min_severity,
            options=_processing_options(args),
        )
    )
    print(json.dumps(summary["outputs"], indent=2))
    return 1 if summary["sets_failed"] else 0


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-file":
            code = _inspect(args)
        elif args.command == "export-set":
            code = _export(args)
        elif args.command == "batch":
            code = _batch(args)
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(args.command)
    except (OSError, ValueError, RuntimeError, IndexError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
