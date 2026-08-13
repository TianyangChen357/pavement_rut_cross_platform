"""Small, defensive QuickLZ block decoder used by Pathway ``.3dc`` files.

The files observed in the target data set contain one non-streaming QuickLZ
1.4.1 level-1 block.  This module deliberately implements decompression only;
new files should use a current, well-supported compression format instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_LONG_HEADER_SIZE: Final = 9
_SHORT_HEADER_SIZE: Final = 3
_HASH_VALUES: Final = 4096
_UNCONDITIONAL_MATCH_LENGTH: Final = 6
_UNCOMPRESSED_END: Final = 4
DEFAULT_MAX_OUTPUT_SIZE: Final = 512 * 1024 * 1024


class QuickLZError(ValueError):
    """Raised when a QuickLZ block is unsupported, truncated, or corrupt."""


@dataclass(frozen=True, slots=True)
class QuickLZHeader:
    """Metadata encoded at the beginning of one QuickLZ block."""

    flags: int
    header_size: int
    compressed_size: int
    decompressed_size: int
    level: int
    streaming_code: int
    is_compressed: bool


def _as_bytes_view(data: bytes | bytearray | memoryview) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError as exc:  # pragma: no cover - Python supplies the exact type error
        raise TypeError("QuickLZ input must be a bytes-like object") from exc
    if view.ndim != 1 or view.itemsize != 1 or not view.contiguous:
        view = memoryview(view.tobytes())
    return view.cast("B")


def _read_little(view: memoryview | bytearray, position: int, size: int, *, what: str) -> int:
    if position < 0 or position + size > len(view):
        raise QuickLZError(f"truncated QuickLZ block while reading {what} at byte {position}")
    value = 0
    for index in range(size):
        value |= int(view[position + index]) << (index * 8)
    return value


def parse_header(data: bytes | bytearray | memoryview) -> QuickLZHeader:
    """Parse a QuickLZ short or long block header without decompressing it."""

    source = _as_bytes_view(data)
    if len(source) < _SHORT_HEADER_SIZE:
        raise QuickLZError("QuickLZ block is shorter than its minimum 3-byte header")

    flags = int(source[0])
    header_size = _LONG_HEADER_SIZE if flags & 0x02 else _SHORT_HEADER_SIZE
    if len(source) < header_size:
        raise QuickLZError(f"QuickLZ block is shorter than its {header_size}-byte header")

    size_width = 4 if header_size == _LONG_HEADER_SIZE else 1
    compressed_size = _read_little(source, 1, size_width, what="compressed size")
    decompressed_offset = 5 if header_size == _LONG_HEADER_SIZE else 2
    decompressed_size = _read_little(source, decompressed_offset, size_width, what="decompressed size")

    if compressed_size < header_size:
        raise QuickLZError(f"invalid QuickLZ compressed size {compressed_size}; it is smaller than the header")

    return QuickLZHeader(
        flags=flags,
        header_size=header_size,
        compressed_size=compressed_size,
        decompressed_size=decompressed_size,
        level=(flags >> 2) & 0x03,
        streaming_code=(flags >> 4) & 0x03,
        is_compressed=bool(flags & 0x01),
    )


def _update_hash_table(
    destination: bytearray,
    table: list[int],
    last_hashed: int,
    through: int,
) -> int:
    """Index completed three-byte sequences through ``through`` (inclusive)."""

    while last_hashed < through:
        last_hashed += 1
        fetch = _read_little(destination, last_hashed, 3, what="decompressed hash sequence")
        hash_value = ((fetch >> 12) ^ fetch) & (_HASH_VALUES - 1)
        table[hash_value] = last_hashed
    return last_hashed


def _decompress_level1(source: memoryview, header: QuickLZHeader) -> bytes:
    source_end = header.compressed_size
    output_size = header.decompressed_size
    destination = bytearray(output_size)
    hash_table = [0] * _HASH_VALUES

    source_position = header.header_size
    destination_position = 0
    control_word = 1
    last_hashed = -1
    fetch = 0
    last_match_start = output_size - _UNCONDITIONAL_MATCH_LENGTH - _UNCOMPRESSED_END - 1

    while True:
        if control_word == 1:
            control_word = _read_little(source, source_position, 4, what="control word")
            source_position += 4
            if control_word == 0:
                raise QuickLZError("invalid zero QuickLZ control word")
            if destination_position <= last_match_start:
                fetch = _read_little(source, source_position, 3, what="level-1 token")

        if control_word & 1:
            control_word >>= 1
            hash_value = (fetch >> 4) & (_HASH_VALUES - 1)
            match_position = hash_table[hash_value]

            if fetch & 0x0F:
                match_length = (fetch & 0x0F) + 2
                source_position += 2
            else:
                match_length = _read_little(source, source_position + 2, 1, what="extended match length")
                source_position += 3

            # QuickLZ level 1 hashes three-byte sequences, so a valid match
            # starts at least three bytes behind the current output cursor.
            if match_position < 0 or match_position > destination_position - 3:
                raise QuickLZError(
                    f"invalid QuickLZ match offset {match_position} at output byte {destination_position}"
                )
            if match_length < 3 or destination_position + match_length > output_size:
                raise QuickLZError(f"invalid QuickLZ match length {match_length} at output byte {destination_position}")

            match_start = destination_position
            # Copy one byte at a time because valid LZ matches may overlap.
            for index in range(match_length):
                destination[destination_position + index] = destination[match_position + index]
            destination_position += match_length

            last_hashed = _update_hash_table(destination, hash_table, last_hashed, match_start)
            last_hashed = destination_position - 1
            fetch = _read_little(source, source_position, 3, what="level-1 token")
            continue

        if destination_position <= last_match_start:
            if source_position >= source_end:
                raise QuickLZError("truncated QuickLZ literal")
            destination[destination_position] = int(source[source_position])
            destination_position += 1
            source_position += 1
            control_word >>= 1

            last_hashed = _update_hash_table(destination, hash_table, last_hashed, destination_position - 3)
            next_byte = _read_little(source, source_position + 2, 1, what="literal look-ahead")
            fetch = ((fetch >> 8) & 0xFFFF) | (next_byte << 16)
            continue

        # The final bytes are always literals.  Control words are still
        # interleaved after every 31 payload decisions and must be skipped.
        while destination_position < output_size:
            if control_word == 1:
                if source_position + 4 > source_end:
                    raise QuickLZError("truncated final QuickLZ control word")
                source_position += 4
                control_word = 0x80000000
            if source_position >= source_end:
                raise QuickLZError("truncated final QuickLZ literal")
            destination[destination_position] = int(source[source_position])
            destination_position += 1
            source_position += 1
            control_word >>= 1
        return bytes(destination)


def decompress(
    data: bytes | bytearray | memoryview,
    *,
    max_output_size: int | None = DEFAULT_MAX_OUTPUT_SIZE,
) -> bytes:
    """Decompress one non-streaming QuickLZ level-1 block.

    ``max_output_size`` is a guard against a forged header allocating an
    unexpectedly large buffer.  Pass ``None`` only for a trusted input whose
    expected size is checked by the caller.
    """

    source = _as_bytes_view(data)
    header = parse_header(source)

    if header.compressed_size != len(source):
        raise QuickLZError(
            "QuickLZ header compressed size does not match the supplied block: "
            f"header={header.compressed_size}, actual={len(source)}"
        )
    if header.decompressed_size <= 0:
        raise QuickLZError(f"invalid QuickLZ decompressed size {header.decompressed_size}")
    if max_output_size is not None:
        if max_output_size <= 0:
            raise ValueError("max_output_size must be positive or None")
        if header.decompressed_size > max_output_size:
            raise QuickLZError(
                f"QuickLZ output size {header.decompressed_size} exceeds the configured limit {max_output_size}"
            )
    if header.level != 1:
        raise QuickLZError(f"unsupported QuickLZ compression level {header.level}; only level 1 is supported")
    if header.streaming_code != 0:
        raise QuickLZError("streaming QuickLZ blocks are not supported")

    if not header.is_compressed:
        expected_size = header.header_size + header.decompressed_size
        if header.compressed_size != expected_size:
            raise QuickLZError(
                f"invalid uncompressed QuickLZ block size: header={header.compressed_size}, expected={expected_size}"
            )
        return bytes(source[header.header_size : expected_size])

    return _decompress_level1(source, header)


__all__ = [
    "DEFAULT_MAX_OUTPUT_SIZE",
    "QuickLZError",
    "QuickLZHeader",
    "decompress",
    "parse_header",
]
