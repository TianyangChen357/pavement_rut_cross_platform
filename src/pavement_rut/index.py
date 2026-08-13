"""Portable indexing of Pathway ``.3dc`` image files."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_TIMECODE_RE = re.compile(r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})(?P<frame>\d{2})(?P<camera>[A-Za-z])$")


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One native 3D image and its half-open navigation-frame interval."""

    relative_path: str
    start_frame: float
    end_frame: float

    def resolve(self, set_dir: Path) -> Path:
        return set_dir / Path(self.relative_path)


def frame_from_filename(path: str | Path, *, frames_per_second: int = 30) -> int:
    """Decode the trailing ``HHMMSSFF<camera>`` timecode used by Pathway files.

    For example, ``11201330004C.3dc`` contains timecode ``01:33:00:04`` and
    maps to frame ``167404`` at 30 frames per second.
    """

    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    stem = Path(path).stem
    match = _TIMECODE_RE.search(stem)
    if match is None:
        raise ValueError(f"File name does not end in HHMMSSFF<camera>: {Path(path).name}")

    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    frame = int(match.group("frame"))
    if minute >= 60 or second >= 60 or frame >= frames_per_second:
        raise ValueError(f"Invalid timecode in file name: {Path(path).name}")
    return ((hour * 60 + minute) * 60 + second) * frames_per_second + frame


def build_index(set_dir: Path, *, frames_per_second: int = 30) -> list[ImageRecord]:
    """Recursively index a set using filename timecodes only."""

    set_dir = set_dir.expanduser().resolve()
    if not set_dir.is_dir():
        raise FileNotFoundError(f"Set directory does not exist: {set_dir}")

    starts: list[tuple[Path, float]] = []
    failures: list[tuple[Path, str]] = []
    for path in set_dir.rglob("*.3dc"):
        try:
            starts.append((path, float(frame_from_filename(path, frames_per_second=frames_per_second))))
        except ValueError as exc:
            failures.append((path, str(exc)))
    if not starts:
        detail = f" First error: {failures[0][1]}" if failures else ""
        raise RuntimeError(f"No indexable .3dc files found under {set_dir}.{detail}")

    starts.sort(key=lambda item: (item[1], item[0].as_posix().casefold()))
    positive_deltas = [
        starts[i + 1][1] - starts[i][1] for i in range(len(starts) - 1) if starts[i + 1][1] > starts[i][1]
    ]
    fallback_delta = float(statistics.median(positive_deltas)) if positive_deltas else 1.0

    records: list[ImageRecord] = []
    for index, (path, start_frame) in enumerate(starts):
        if index + 1 < len(starts) and starts[index + 1][1] > start_frame:
            end_frame = starts[index + 1][1]
        else:
            end_frame = start_frame + fallback_delta
        records.append(
            ImageRecord(
                relative_path=path.relative_to(set_dir).as_posix(),
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    return records


def write_index(path: Path, set_dir: Path, records: Iterable[ImageRecord]) -> None:
    """Write a machine-portable relative-path index atomically."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "set": set_dir.name,
        "records": [asdict(record) for record in records],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_index(path: Path, set_dir: Path) -> list[ImageRecord]:
    """Read a portable index and reject mismatched sets or unsafe paths."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported index schema in {path}")
    if str(payload.get("set")) != set_dir.name:
        raise ValueError(f"Index set {payload.get('set')!r} does not match directory {set_dir.name!r}")

    records: list[ImageRecord] = []
    for raw in payload.get("records", []):
        relative = Path(str(raw["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative path in index: {relative}")
        records.append(
            ImageRecord(
                relative_path=relative.as_posix(),
                start_frame=float(raw["start_frame"]),
                end_frame=float(raw["end_frame"]),
            )
        )
    if not records:
        raise ValueError(f"Index contains no records: {path}")
    return records


def get_or_build_index(
    set_dir: Path,
    index_path: Path,
    *,
    rebuild: bool = False,
    frames_per_second: int = 30,
) -> list[ImageRecord]:
    """Load a cached index or rebuild it from the set."""

    if index_path.exists() and not rebuild:
        return read_index(index_path, set_dir)
    records = build_index(set_dir, frames_per_second=frames_per_second)
    write_index(index_path, set_dir, records)
    return records
