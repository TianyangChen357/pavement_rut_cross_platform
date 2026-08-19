from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from pavement_rut.preview import (
    calibrated_height_grayscale,
    encode_grayscale_png,
    write_height_preview_png,
)


def _png_chunks(payload: bytes) -> dict[bytes, bytes]:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    chunks: dict[bytes, bytes] = {}
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        chunks[kind] = chunks.get(kind, b"") + data
        offset += 12 + length
    return chunks


def test_calibrated_height_grayscale_uses_robust_monotonic_contrast() -> None:
    heights = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])

    grayscale = calibrated_height_grayscale(heights, lower_percentile=0.0, upper_percentile=100.0)

    assert grayscale.dtype == np.uint8
    assert grayscale.shape == heights.shape
    assert np.all(np.diff(grayscale[0].astype(int)) > 0)
    assert grayscale[1, 1] == 255
    assert grayscale[1, 2] == 0


def test_encode_grayscale_png_has_expected_dimensions_and_samples() -> None:
    grayscale = np.asarray([[0, 127, 255], [10, 20, 30]], dtype=np.uint8)

    chunks = _png_chunks(encode_grayscale_png(grayscale))

    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert (width, height) == (3, 2)
    assert (bit_depth, color_type, compression, filter_method, interlace) == (8, 0, 0, 0, 0)
    assert zlib.decompress(chunks[b"IDAT"]) == b"\x00\x00\x7f\xff\x00\x0a\x14\x1e"


def test_write_height_preview_png_is_atomic_and_rejects_bad_shape(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "preview.png"
    write_height_preview_png(destination, np.arange(12, dtype=float).reshape(3, 4))

    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(destination.parent.glob(".*.tmp"))
    with pytest.raises(ValueError, match="two-dimensional"):
        write_height_preview_png(destination, np.arange(4, dtype=float))
