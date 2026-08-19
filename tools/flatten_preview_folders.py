#!/usr/bin/env python3
"""Flatten per-set preview subdirectories and repair exported path references."""

from __future__ import annotations

import argparse
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreviewMove:
    source: Path
    destination: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move PNGs from set_*/previews subdirectories into each previews root "
            "and update CSV, GeoJSON, and metadata references."
        )
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration; without this option only validate and report",
    )
    return parser


def _discover_moves(output_root: Path) -> dict[Path, list[PreviewMove]]:
    by_set: dict[Path, list[PreviewMove]] = {}
    destinations: dict[Path, Path] = {}
    for preview_dir in sorted(output_root.glob("set_*/previews")):
        moves = [
            PreviewMove(source=source, destination=preview_dir / source.name)
            for source in sorted(preview_dir.rglob("*.png"))
            if source.parent != preview_dir
        ]
        for move in moves:
            previous = destinations.setdefault(move.destination, move.source)
            if previous != move.source:
                raise RuntimeError(f"Duplicate destination {move.destination}: {previous} and {move.source}")
            if move.destination.exists():
                raise RuntimeError(f"Destination already exists: {move.destination}")
        if moves:
            by_set[preview_dir.parent] = moves
    return by_set


def _artifacts(set_dir: Path) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    for suffix in (".csv", ".geojson", ".metadata.json"):
        matches = sorted(set_dir.glob(f"*_rut_results{suffix}"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one *_rut_results{suffix} in {set_dir}, found {len(matches)}")
        artifacts.append(matches[0])
    return tuple(artifacts)


def _updated_artifact(
    artifact: Path,
    *,
    preview_dir: Path,
    moves: list[PreviewMove],
) -> str:
    text = artifact.read_text(encoding="utf-8")
    replacements = {str(move.source.resolve()): str(move.destination.resolve()) for move in moves}
    prefix = re.escape(str(preview_dir.resolve()))
    pattern = re.compile(prefix + r"/(?:[^/\"\\,\r\n]+/)+(?P<filename>[^/\"\\,\r\n]+\.png)")
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        old_path = match.group(0)
        replacement = replacements.get(old_path)
        if replacement is None:
            raise RuntimeError(f"Unexpected nested preview reference in {artifact}: {old_path}")
        seen.append(old_path)
        return replacement

    updated = pattern.sub(replace, text)
    expected = set(replacements)
    if set(seen) != expected or len(seen) != len(expected):
        missing = sorted(expected.difference(seen))
        duplicates = len(seen) - len(set(seen))
        raise RuntimeError(
            f"Preview references in {artifact} do not match files: "
            f"missing={len(missing)}, duplicate_references={duplicates}"
        )
    return updated


def _write_text_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_empty_subdirectories(preview_dir: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in preview_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


def run(output_root: Path, *, apply: bool) -> int:
    output_root = output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    moves_by_set = _discover_moves(output_root)
    prepared: dict[Path, tuple[tuple[Path, str], ...]] = {}
    for set_dir, moves in moves_by_set.items():
        preview_dir = set_dir / "previews"
        prepared[set_dir] = tuple(
            (artifact, _updated_artifact(artifact, preview_dir=preview_dir, moves=moves))
            for artifact in _artifacts(set_dir)
        )

    total = sum(len(moves) for moves in moves_by_set.values())
    print(f"Validated {total:,} nested PNGs across {len(moves_by_set)} sets; collisions=0")
    if not apply:
        print("Dry run only; rerun with --apply to migrate files and references")
        return 0

    linked: list[Path] = []
    try:
        for moves in moves_by_set.values():
            for move in moves:
                os.link(move.source, move.destination)
                linked.append(move.destination)
    except BaseException:
        for destination in reversed(linked):
            destination.unlink(missing_ok=True)
        raise

    for artifacts in prepared.values():
        for artifact, updated in artifacts:
            _write_text_atomic(artifact, updated)

    for moves in moves_by_set.values():
        for move in moves:
            move.source.unlink()

    removed_directories = sum(_remove_empty_subdirectories(set_dir / "previews") for set_dir in moves_by_set)
    print(
        f"Flattened {total:,} PNGs and updated {len(moves_by_set) * 3} artifacts; "
        f"removed {removed_directories:,} empty subdirectories"
    )
    return 0


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(run(args.output_root, apply=args.apply))


if __name__ == "__main__":
    main()
