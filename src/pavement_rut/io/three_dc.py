"""Pure-Python reader for the Pathway ``.3dc`` profile container."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pavement_rut.io.calibration import DEFAULT_COLUMNS, CameraCalibration
from pavement_rut.io.quicklz import (
    DEFAULT_MAX_OUTPUT_SIZE,
    QuickLZError,
    QuickLZHeader,
    decompress,
    parse_header,
)

PROFILE_HEADER_BYTES: Final = 8
PATHWAY_COLUMNS: Final = DEFAULT_COLUMNS
PATHWAY_PROFILES_PER_FILE: Final = 900
PATHWAY_RECORD_BYTES: Final = PROFILE_HEADER_BYTES + PATHWAY_COLUMNS + (2 * PATHWAY_COLUMNS)
PATHWAY_DECOMPRESSED_BYTES: Final = PATHWAY_PROFILES_PER_FILE * PATHWAY_RECORD_BYTES


class ThreeDCFormatError(ValueError):
    """Raised when decompressed bytes do not form valid fixed-width profiles."""


def _readonly_copy(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ThreeDCProfile:
    """One transverse profile in PathView-compatible left-to-right order."""

    index: int
    header: bytes
    intensity: NDArray[np.uint8]
    raw_heights: NDArray[np.uint16]

    def calibrated_heights(self, calibration: CameraCalibration) -> NDArray[np.float64]:
        """Apply a camera calibration and return height values in inches."""

        return calibration.apply(self.raw_heights)


@dataclass(frozen=True, slots=True, eq=False)
class ThreeDCImage:
    """Decoded arrays for every transverse profile in one ``.3dc`` block."""

    record_headers: NDArray[np.uint8]
    intensity: NDArray[np.uint8]
    raw_heights: NDArray[np.uint16]
    quicklz_header: QuickLZHeader
    reverse_columns: bool
    source: Path | None = None

    @property
    def profile_count(self) -> int:
        return int(self.raw_heights.shape[0])

    @property
    def columns(self) -> int:
        return int(self.raw_heights.shape[1])

    def profile(self, index: int) -> ThreeDCProfile:
        """Return a zero-copy, read-only view of one decoded profile."""

        normalized = index if index >= 0 else self.profile_count + index
        if normalized < 0 or normalized >= self.profile_count:
            raise IndexError(f"profile index {index} is out of range for {self.profile_count} profiles")
        return ThreeDCProfile(
            index=normalized,
            header=self.record_headers[normalized].tobytes(),
            intensity=self.intensity[normalized],
            raw_heights=self.raw_heights[normalized],
        )

    def iter_profiles(self) -> Iterator[ThreeDCProfile]:
        """Iterate over profiles without copying their intensity/height arrays."""

        for index in range(self.profile_count):
            yield self.profile(index)

    def __iter__(self) -> Iterator[ThreeDCProfile]:
        return self.iter_profiles()

    def calibrated_heights(self, calibration: CameraCalibration) -> NDArray[np.float64]:
        """Calibrate the full ``(profiles, columns)`` height array to inches."""

        return calibration.apply(self.raw_heights)


def loads_3dc(
    data: bytes | bytearray | memoryview,
    *,
    columns: int = PATHWAY_COLUMNS,
    reverse_columns: bool = True,
    expected_profiles: int | None = None,
    max_output_size: int | None = DEFAULT_MAX_OUTPUT_SIZE,
    source: str | Path | None = None,
) -> ThreeDCImage:
    """Decompress and decode a complete in-memory ``.3dc`` file.

    Each record contains an opaque 8-byte header, ``columns`` intensity bytes,
    and ``columns`` little-endian unsigned 16-bit height samples.  The on-disk
    samples run in the opposite direction from PathView's public profile API;
    the default ``reverse_columns=True`` returns the compatible left-to-right
    order for both intensity and height.
    """

    if columns <= 0:
        raise ValueError("columns must be positive")
    if expected_profiles is not None and expected_profiles <= 0:
        raise ValueError("expected_profiles must be positive or None")

    try:
        quicklz_header = parse_header(data)
        payload = decompress(data, max_output_size=max_output_size)
    except QuickLZError as exc:
        raise ThreeDCFormatError(f"invalid .3dc QuickLZ block: {exc}") from exc

    record_bytes = PROFILE_HEADER_BYTES + columns + (2 * columns)
    if len(payload) == 0 or len(payload) % record_bytes:
        raise ThreeDCFormatError(
            f"decompressed .3dc size {len(payload)} is not a positive multiple of "
            f"the {record_bytes}-byte profile record"
        )
    profile_count = len(payload) // record_bytes
    if expected_profiles is not None and profile_count != expected_profiles:
        raise ThreeDCFormatError(f".3dc contains {profile_count} profiles; expected {expected_profiles}")

    byte_records = np.frombuffer(payload, dtype=np.uint8).reshape(profile_count, record_bytes)
    record_headers = _readonly_copy(byte_records[:, :PROFILE_HEADER_BYTES])
    intensity = byte_records[:, PROFILE_HEADER_BYTES : PROFILE_HEADER_BYTES + columns]

    # A strided view avoids first copying the interleaved record headers and
    # intensity bytes.  The final C-order copy also normalizes little-endian
    # uint16 values on any host architecture.
    raw_heights = np.ndarray(
        shape=(profile_count, columns),
        dtype="<u2",
        buffer=payload,
        offset=PROFILE_HEADER_BYTES + columns,
        strides=(record_bytes, 2),
    )
    if reverse_columns:
        intensity = intensity[:, ::-1]
        raw_heights = raw_heights[:, ::-1]

    intensity = _readonly_copy(intensity)
    raw_heights = _readonly_copy(raw_heights).astype(np.uint16, copy=False)
    raw_heights.setflags(write=False)

    return ThreeDCImage(
        record_headers=record_headers,
        intensity=intensity,
        raw_heights=raw_heights,
        quicklz_header=quicklz_header,
        reverse_columns=reverse_columns,
        source=Path(source) if source is not None else None,
    )


def read_3dc(
    path: str | Path,
    *,
    columns: int = PATHWAY_COLUMNS,
    reverse_columns: bool = True,
    expected_profiles: int | None = None,
    max_output_size: int | None = DEFAULT_MAX_OUTPUT_SIZE,
) -> ThreeDCImage:
    """Read, decompress, and decode one ``.3dc`` file."""

    source = Path(path)
    return loads_3dc(
        source.read_bytes(),
        columns=columns,
        reverse_columns=reverse_columns,
        expected_profiles=expected_profiles,
        max_output_size=max_output_size,
        source=source,
    )


__all__ = [
    "PATHWAY_COLUMNS",
    "PATHWAY_DECOMPRESSED_BYTES",
    "PATHWAY_PROFILES_PER_FILE",
    "PATHWAY_RECORD_BYTES",
    "PROFILE_HEADER_BYTES",
    "ThreeDCFormatError",
    "ThreeDCImage",
    "ThreeDCProfile",
    "loads_3dc",
    "read_3dc",
]
