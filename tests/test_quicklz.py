from __future__ import annotations

import struct

import pytest

from pavement_rut.io.quicklz import QuickLZError, decompress, parse_header


def _uncompressed_block(payload: bytes, *, long_header: bool = True) -> bytes:
    if long_header:
        return bytes([0x46]) + struct.pack("<II", len(payload) + 9, len(payload)) + payload
    if len(payload) + 3 > 255:
        raise ValueError("short QuickLZ test block is limited to 255 bytes")
    return bytes([0x44, len(payload) + 3, len(payload)]) + payload


def test_parse_and_decode_uncompressed_long_header() -> None:
    payload = bytes(range(128))
    block = _uncompressed_block(payload)

    header = parse_header(block)

    assert header.header_size == 9
    assert header.level == 1
    assert header.streaming_code == 0
    assert header.compressed_size == len(block)
    assert header.decompressed_size == len(payload)
    assert header.is_compressed is False
    assert decompress(block) == payload


def test_decode_uncompressed_short_header() -> None:
    payload = b"synthetic quicklz payload"
    block = _uncompressed_block(payload, long_header=False)

    assert parse_header(block).header_size == 3
    assert decompress(block) == payload


def test_decode_level1_match_with_overlap() -> None:
    # Synthetic stream: three literals (abc), a nine-byte back-reference to
    # abc, then four terminal literals.  This exercises the level-1 hash path
    # and overlapping LZ copy without embedding any survey data.
    body = bytes.fromhex("08 00 00 80") + b"abc" + bytes.fromhex("77 45") + b"WXYZ"
    block = bytes([0x47]) + struct.pack("<II", len(body) + 9, 16) + body

    assert decompress(block) == b"abcabcabcabcWXYZ"


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (b"\x46\x03", "minimum 3-byte header"),
        (bytes([0x46]) + struct.pack("<II", 11, 1) + b"x", "compressed size does not match"),
        (bytes([0x4E]) + struct.pack("<II", 10, 1) + b"x", "compression level 3"),
        (bytes([0x56]) + struct.pack("<II", 10, 1) + b"x", "streaming"),
    ],
)
def test_rejects_invalid_or_unsupported_blocks(block: bytes, message: str) -> None:
    with pytest.raises(QuickLZError, match=message):
        decompress(block)


def test_output_allocation_guard() -> None:
    block = _uncompressed_block(b"0123456789")

    with pytest.raises(QuickLZError, match="exceeds the configured limit"):
        decompress(block, max_output_size=9)
