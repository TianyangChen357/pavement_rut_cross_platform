"""Navigation-file parsing and interpolation."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NavigationPoint:
    frame: float
    latitude: float
    longitude: float
    heading: float


def read_numeric_rows(path: Path, *, minimum_columns: int) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < minimum_columns:
                continue
            try:
                rows.append(tuple(float(part) for part in parts))
            except ValueError:
                continue
    rows.sort(key=lambda row: row[0])
    return rows


def interpolate_columns(rows: Sequence[Sequence[float]], frame: float, columns: Sequence[int]) -> tuple[float, ...]:
    return _interpolate_columns(rows, [row[0] for row in rows], frame, columns)


def _interpolate_columns(
    rows: Sequence[Sequence[float]],
    frames: Sequence[float],
    frame: float,
    columns: Sequence[int],
) -> tuple[float, ...]:
    if not rows:
        return tuple(math.nan for _ in columns)
    if frame < frames[0] or frame > frames[-1]:
        return tuple(math.nan for _ in columns)

    right = bisect.bisect_left(frames, frame)
    if right < len(rows) and math.isclose(frames[right], frame):
        return tuple(float(rows[right][column]) if column < len(rows[right]) else math.nan for column in columns)

    left = right - 1
    if left < 0 or right >= len(rows):
        return tuple(math.nan for _ in columns)
    x0, x1 = frames[left], frames[right]
    if math.isclose(x0, x1):
        return tuple(float(rows[left][column]) if column < len(rows[left]) else math.nan for column in columns)
    fraction = (frame - x0) / (x1 - x0)
    result: list[float] = []
    for column in columns:
        if column >= len(rows[left]) or column >= len(rows[right]):
            result.append(math.nan)
            continue
        y0, y1 = float(rows[left][column]), float(rows[right][column])
        result.append(y0 + fraction * (y1 - y0))
    return tuple(result)


class NavigationLookup:
    """Load navigation tables once and provide frame-based interpolation."""

    def __init__(self, set_dir: Path) -> None:
        self.set_dir = set_dir
        set_label = set_dir.name
        self.gps_path = set_dir / f"gpsdis.{set_label}"
        self.heading_path = set_dir / f"heading.{set_label}"
        self.gps_rows = read_numeric_rows(self.gps_path, minimum_columns=4)
        self.heading_rows = read_numeric_rows(self.heading_path, minimum_columns=2)
        self._gps_frames = tuple(row[0] for row in self.gps_rows)
        self._heading_frames = tuple(row[0] for row in self.heading_rows)

    def __call__(self, frame: float) -> NavigationPoint:
        latitude, longitude = _interpolate_columns(self.gps_rows, self._gps_frames, frame, (1, 2))
        heading = _interpolate_columns(self.heading_rows, self._heading_frames, frame, (1,))[0]
        return NavigationPoint(frame, latitude, longitude, heading)

    def metadata(self) -> dict[str, str | int | None]:
        return {
            "gpsdis": str(self.gps_path) if self.gps_rows else None,
            "heading": str(self.heading_path) if self.heading_rows else None,
            "gps_rows": len(self.gps_rows),
            "heading_rows": len(self.heading_rows),
        }
