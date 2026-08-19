"""Dependency-free grayscale PNG previews for calibrated ``.3dc`` heights."""

from __future__ import annotations

import os
import struct
import uuid
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def calibrated_height_grayscale(
    calibrated_heights: ArrayLike,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> NDArray[np.uint8]:
    """Map a ``(profiles, columns)`` calibrated-height array to grayscale.

    Low elevations are dark and high elevations are light. Robust percentile
    limits keep isolated invalid/extreme samples from flattening useful surface
    contrast. Non-finite samples are black.
    """

    heights = np.asarray(calibrated_heights, dtype=np.float64)
    if heights.ndim != 2 or 0 in heights.shape:
        raise ValueError("calibrated heights must be a non-empty two-dimensional array")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("preview percentiles must satisfy 0 <= lower < upper <= 100")

    finite = np.isfinite(heights)
    if not np.any(finite):
        raise ValueError("calibrated heights contain no finite values")
    lower, upper = np.percentile(heights[finite], [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper):  # pragma: no cover - guarded by finite mask
        raise ValueError("preview height range is not finite")

    if upper <= lower:
        normalized = np.full(heights.shape, 0.5, dtype=np.float64)
    else:
        normalized = np.clip((heights - lower) / (upper - lower), 0.0, 1.0)
    normalized = np.where(finite, normalized, 0.0)

    grayscale = np.rint(normalized * 255.0).astype(np.uint8)
    grayscale[~finite] = 0
    return grayscale


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def encode_grayscale_png(grayscale: ArrayLike, *, compression_level: int = 6) -> bytes:
    """Encode a two-dimensional uint8 array as a grayscale PNG byte string."""

    pixels = np.asarray(grayscale)
    if pixels.ndim != 2 or 0 in pixels.shape:
        raise ValueError("grayscale preview must have shape (height, width)")
    if pixels.dtype != np.uint8:
        raise TypeError("grayscale preview must use uint8 samples")
    if not 0 <= compression_level <= 9:
        raise ValueError("PNG compression_level must be between 0 and 9")

    height, width = pixels.shape
    scanlines = np.empty((height, 1 + width), dtype=np.uint8)
    scanlines[:, 0] = 0  # PNG filter type: None
    scanlines[:, 1:] = np.ascontiguousarray(pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    compressed = zlib.compress(scanlines.tobytes(), level=compression_level)
    return _PNG_SIGNATURE + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def write_height_preview_png(path: Path, calibrated_heights: ArrayLike) -> None:
    """Atomically write a calibrated-height grayscale PNG."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = encode_grayscale_png(calibrated_height_grayscale(calibrated_heights))
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["calibrated_height_grayscale", "encode_grayscale_png", "write_height_preview_png"]
