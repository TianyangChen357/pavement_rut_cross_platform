from __future__ import annotations

import struct

import numpy as np
import pytest

from pavement_rut.io.calibration import loads_calibration
from pavement_rut.io.three_dc import ThreeDCFormatError, loads_3dc, read_3dc


def _quicklz_uncompressed(payload: bytes) -> bytes:
    return bytes([0x46]) + struct.pack("<II", len(payload) + 9, len(payload)) + payload


def _record(header: bytes, intensity: list[int], heights: list[int]) -> bytes:
    assert len(header) == 8
    assert len(intensity) == len(heights)
    return header + bytes(intensity) + struct.pack(f"<{len(heights)}H", *heights)


def _synthetic_3dc() -> bytes:
    # Samples are deliberately stored in disk order; the reader's default
    # reverses them to the order exposed by the reference profile API.
    payload = b"".join(
        [
            _record(b"HEADER01", [1, 2, 3, 4], [100, 200, 300, 400]),
            _record(b"HEADER02", [10, 20, 30, 40], [11, 22, 33, 44]),
        ]
    )
    return _quicklz_uncompressed(payload)


def test_decodes_records_and_reverses_column_direction() -> None:
    image = loads_3dc(_synthetic_3dc(), columns=4, expected_profiles=2)

    assert image.profile_count == 2
    assert image.columns == 4
    assert image.record_headers[0].tobytes() == b"HEADER01"
    np.testing.assert_array_equal(image.intensity, [[4, 3, 2, 1], [40, 30, 20, 10]])
    np.testing.assert_array_equal(image.raw_heights, [[400, 300, 200, 100], [44, 33, 22, 11]])
    assert image.intensity.dtype == np.uint8
    assert image.raw_heights.dtype == np.uint16
    assert image.intensity.flags.writeable is False
    assert image.raw_heights.flags.writeable is False

    profile = image.profile(-1)
    assert profile.index == 1
    assert profile.header == b"HEADER02"
    np.testing.assert_array_equal(profile.raw_heights, [44, 33, 22, 11])
    assert [item.index for item in image] == [0, 1]


def test_can_retain_physical_disk_order() -> None:
    image = loads_3dc(_synthetic_3dc(), columns=4, reverse_columns=False)

    np.testing.assert_array_equal(image.intensity[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(image.raw_heights[0], [100, 200, 300, 400])


def test_calibrates_profile_and_image() -> None:
    calibration = loads_calibration(
        "3D_Camera_Calibration\n0 1 2 3 30 8 82 40 0.03 0.03 80 0.5 2020 1 2 3 4",
        columns=4,
    )
    image = loads_3dc(_synthetic_3dc(), columns=4)

    expected = (image.raw_heights.astype(float) - np.array([0.0, 1.0, 2.0, 3.0])) * calibration.height_resolution_inches
    np.testing.assert_array_equal(image.calibrated_heights(calibration), expected)
    np.testing.assert_array_equal(image.profile(0).calibrated_heights(calibration), expected[0])


def test_read_3dc_records_source_path(tmp_path) -> None:
    path = tmp_path / "synthetic.3dc"
    path.write_bytes(_synthetic_3dc())

    image = read_3dc(path, columns=4)

    assert image.source == path


def test_rejects_invalid_record_size_and_profile_count() -> None:
    with pytest.raises(ThreeDCFormatError, match="not a positive multiple"):
        loads_3dc(_quicklz_uncompressed(b"wrong length"), columns=4)

    with pytest.raises(ThreeDCFormatError, match="contains 2 profiles; expected 3"):
        loads_3dc(_synthetic_3dc(), columns=4, expected_profiles=3)


def test_profile_index_bounds() -> None:
    image = loads_3dc(_synthetic_3dc(), columns=4)

    with pytest.raises(IndexError, match="out of range"):
        image.profile(2)
