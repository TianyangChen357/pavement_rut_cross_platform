from pathlib import Path

import pytest

from pavement_rut.index import build_index, frame_from_filename, read_index, write_index


def test_pathway_timecode_examples() -> None:
    assert frame_from_filename("11201330004C.3dc") == 167404
    assert frame_from_filename("11202062127C.3dc") == 227457


@pytest.mark.parametrize("name", ["bad.3dc", "11201600000C.3dc", "11201010030C.3dc"])
def test_invalid_timecode(name: str) -> None:
    with pytest.raises(ValueError):
        frame_from_filename(name)


def test_portable_index_round_trip(tmp_path: Path) -> None:
    set_dir = tmp_path / "112"
    nested = set_dir / "93"
    nested.mkdir(parents=True)
    (nested / "11201330004C.3dc").touch()
    (nested / "11201330017C.3dc").touch()

    records = build_index(set_dir)
    assert [record.start_frame for record in records] == [167404.0, 167417.0]
    assert records[0].relative_path == "93/11201330004C.3dc"
    assert records[-1].end_frame == 167430.0

    path = tmp_path / "index.json"
    write_index(path, set_dir, records)
    assert read_index(path, set_dir) == records
