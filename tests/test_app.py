from __future__ import annotations

import csv
import importlib
import json
import os
import struct
from pathlib import Path

import pytest

from pavement_rut import cli
from pavement_rut.app.batch import BatchConfig, discover_sets, run_batch
from pavement_rut.app.checkpoint import PROCESSING_SCHEMA_VERSION, RecordIdentity
from pavement_rut.app.export_set import ExportSetConfig, ProcessingOptions, export_set
from pavement_rut.domain.models import CrossSlopeResult

_COLUMNS = 1536


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


def _quicklz_uncompressed(payload: bytes) -> bytes:
    return bytes([0x46]) + struct.pack("<II", len(payload) + 9, len(payload)) + payload


def _write_synthetic_set(
    root: Path,
    set_number: str,
    *,
    file_count: int = 1,
    profiles_per_file: int = 1,
) -> Path:
    """Create the smallest real files accepted by the production I/O path."""

    set_dir = root / set_number
    image_dir = set_dir / "93"
    image_dir.mkdir(parents=True)

    calibration_tail = "30 8 82 162 0.03 0.03 80 0.5 2020 1 2 3 4"
    offsets = " ".join("0" for _ in range(_COLUMNS))
    (set_dir / "3D_Camera.cal").write_text(
        f"3D_Camera_Calibration\n{offsets} {calibration_tail}\n",
        encoding="utf-8",
    )

    intensity = bytes(index % 256 for index in range(_COLUMNS))
    raw_heights = struct.pack(f"<{_COLUMNS}H", *([1000] * _COLUMNS))
    record = b"HEADER01" + intensity + raw_heights
    for index in range(file_count):
        frame = 4 + index
        name = f"{set_number}013300{frame:02d}C.3dc"
        (image_dir / name).write_bytes(_quicklz_uncompressed(record * profiles_per_file))
    return set_dir


def test_export_set_writes_strict_json_csv_and_geojson(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    (set_dir / "gpsdis.112").write_text(
        "167400 35 -78 0\n167410 36 -79 0\n",
        encoding="utf-8",
    )
    (set_dir / "heading.112").write_text(
        "167400 10\n167410 20\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=out_dir,
            output_name="synthetic",
            progress_every=1,
        )
    )

    assert metadata["records_indexed"] == 1
    assert metadata["records_exported"] == 1
    assert metadata["diagnostics"][0]["rows_total"] == 1
    assert metadata["diagnostics"][0]["rows_ok"] == 1
    assert metadata["diagnostics"][0]["rows_error"] == 0
    assert metadata["diagnostics"][0]["cross_slope_average_percent"] == pytest.approx(0.0)
    assert metadata["diagnostics"][0]["cross_slope_average_angle_degrees"] == pytest.approx(0.0)
    assert metadata["diagnostics"][0]["cross_slope_count"] == 1
    assert metadata["diagnostics"][0]["cross_slope_error_count"] == 0
    assert metadata["options"]["maximum_noise_std_inches"] == 999.0

    geojson_path = Path(metadata["outputs"]["geojson"])
    csv_path = Path(metadata["outputs"]["csv"])
    metadata_path = Path(metadata["outputs"]["metadata_json"])
    assert geojson_path.name == "synthetic.geojson"
    assert csv_path.name == "synthetic.csv"
    assert metadata_path.name == "synthetic.metadata.json"

    # JSON must be portable to strict parsers; NaN/Infinity are not valid JSON.
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    persisted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    assert persisted_metadata["diagnostics"][0]["error_samples"] == []
    assert persisted_metadata["measurement_units"] == {
        "rut_depth": "inch",
        "cross_slope_average_percent": "percent",
        "cross_slope_average_angle_degrees": "degree",
    }
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert feature["geometry"]["coordinates"] == pytest.approx([-78.4, 35.4])
    assert feature["properties"]["averaged_rut"] == pytest.approx(0.0)
    assert feature["properties"]["cross_slope_average_percent"] == pytest.approx(0.0)
    assert feature["properties"]["cross_slope_average_angle_degrees"] == pytest.approx(0.0)
    assert feature["properties"]["cross_slope_count"] == 1
    assert feature["properties"]["cross_slope_error_count"] == 0
    assert feature["properties"]["severity"] == 0

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["relative_path"] == "93/11201330004C.3dc"
    assert float(rows[0]["averaged_rut"]) == pytest.approx(0.0)
    assert float(rows[0]["cross_slope_average_percent"]) == pytest.approx(0.0)
    assert float(rows[0]["cross_slope_average_angle_degrees"]) == pytest.approx(0.0)
    assert rows[0]["cross_slope_count"] == "1"
    assert rows[0]["cross_slope_error_count"] == "0"
    assert rows[0]["severity"] == "0"


def test_cross_slope_failure_keeps_rut_and_serializes_null(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    export_module = importlib.import_module("pavement_rut.app.export_set")

    def unavailable_cross_slope(*args: object, **kwargs: object) -> object:
        raise ValueError("simulated cross-slope geometry failure")

    monkeypatch.setattr(export_module, "fit_cross_slope", unavailable_cross_slope)
    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=tmp_path / "out",
            output_name="missing-cross-slope",
        )
    )

    result = metadata["diagnostics"][0]
    assert metadata["status"] == "partial"
    assert metadata["files_partial"] == 1
    assert result["status"] == "partial"
    assert result["overall_average_inches"] == pytest.approx(0.0)
    assert result["cross_slope_count"] == 0
    assert result["cross_slope_error_count"] == 1

    persisted = json.loads(
        Path(metadata["outputs"]["metadata_json"]).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    persisted_result = persisted["diagnostics"][0]
    assert persisted_result["cross_slope_average_percent"] is None
    assert persisted_result["cross_slope_average_angle_degrees"] is None

    geojson = json.loads(
        Path(metadata["outputs"]["geojson"]).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    properties = geojson["features"][0]["properties"]
    assert properties["averaged_rut"] == pytest.approx(0.0)
    assert properties["cross_slope_average_percent"] is None
    assert properties["cross_slope_average_angle_degrees"] is None
    assert properties["cross_slope_count"] == 0
    assert properties["cross_slope_error_count"] == 1

    with Path(metadata["outputs"]["csv"]).open(newline="", encoding="utf-8") as stream:
        csv_result = next(csv.DictReader(stream))
    assert float(csv_result["averaged_rut"]) == pytest.approx(0.0)
    assert csv_result["cross_slope_average_percent"] == ""
    assert csv_result["cross_slope_average_angle_degrees"] == ""
    assert csv_result["cross_slope_count"] == "0"
    assert csv_result["cross_slope_error_count"] == "1"


def test_cross_slope_averages_only_finite_profile_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(
        tmp_path / "data",
        "112",
        profiles_per_file=3,
    )
    export_module = importlib.import_module("pavement_rut.app.export_set")
    results = iter(
        (
            CrossSlopeResult(0.01, 1.0, 0.5, 0.0, 1.0, 100),
            CrossSlopeResult(float("nan"), float("nan"), float("nan"), 0.0, 1.0, 100),
            CrossSlopeResult(0.03, 3.0, 1.5, 0.0, 1.0, 100),
        )
    )

    def next_cross_slope(*args: object, **kwargs: object) -> CrossSlopeResult:
        return next(results)

    monkeypatch.setattr(export_module, "fit_cross_slope", next_cross_slope)
    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=tmp_path / "out",
            output_name="finite-cross-slope",
        )
    )

    result = metadata["diagnostics"][0]
    assert result["status"] == "partial"
    assert result["rows_ok"] == 3
    assert result["cross_slope_average_percent"] == pytest.approx(2.0)
    assert result["cross_slope_average_angle_degrees"] == pytest.approx(1.0)
    assert result["cross_slope_count"] == 2
    assert result["cross_slope_error_count"] == 1


def test_export_set_parallel_processes_each_file_once(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112", file_count=2)

    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=tmp_path / "out",
            output_name="parallel",
            jobs=2,
            progress_every=1,
        )
    )

    assert metadata["records_exported"] == 2
    assert [row["file_name"] for row in metadata["diagnostics"]] == [
        "11201330004C.3dc",
        "11201330005C.3dc",
    ]
    assert sum(row["rows_total"] for row in metadata["diagnostics"]) == 2


def test_batch_parallel_isolates_failure_and_writes_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_synthetic_set(data_root, "2")
    broken = _write_synthetic_set(data_root, "10")
    (broken / "3D_Camera.cal").unlink()

    summary = run_batch(
        BatchConfig(
            data_root=data_root,
            out_dir=tmp_path / "out",
            jobs=2,
            progress_every=1,
        )
    )

    assert summary["sets_total"] == 2
    assert summary["sets_ok"] == 1
    assert summary["sets_failed"] == 1
    assert [row["set"] for row in summary["results"]] == ["2", "10"]
    assert [row["status"] for row in summary["results"]] == ["ok", "failed"]
    assert "FileNotFoundError" in summary["results"][1]["error"]

    payload = json.loads(
        Path(summary["outputs"]["json"]).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    assert payload["sets_failed"] == 1


def test_discover_sets_sorts_numeric_labels_numerically(tmp_path: Path) -> None:
    for name in ("10", "2", "not-a-set"):
        (tmp_path / name).mkdir()

    assert [path.name for path in discover_sets(tmp_path)] == ["2", "10"]


def test_discover_sets_rejects_empty_data_root(tmp_path: Path) -> None:
    (tmp_path / "not-a-set").mkdir()

    with pytest.raises(ValueError, match="No numeric set directories"):
        discover_sets(tmp_path)


@pytest.mark.parametrize(
    "output_name",
    ["../escape", r"..\escape", "..", ".hidden", "CON", "LPT1.report"],
)
def test_export_set_rejects_unsafe_output_name(
    tmp_path: Path,
    output_name: str,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="output_name"):
        export_set(
            ExportSetConfig(
                set_dir=set_dir,
                out_dir=out_dir,
                output_name=output_name,
                progress_every=1,
            )
        )

    assert not (tmp_path / "escape.geojson").exists()
    assert not (tmp_path / "escape.csv").exists()
    assert not (tmp_path / "escape.metadata.json").exists()


def test_resume_reuses_completed_results_without_calling_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112", file_count=2)
    config = ExportSetConfig(
        set_dir=set_dir,
        out_dir=tmp_path / "out",
        output_name="resumable",
        progress_every=1,
    )
    first = export_set(config)
    journal_path = Path(first["checkpoint"]["journal_jsonl"])
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 2

    export_module = importlib.import_module("pavement_rut.app.export_set")

    def unexpected_worker(payload: object) -> object:
        raise AssertionError(f"cached result unexpectedly invoked worker: {payload!r}")

    monkeypatch.setattr(export_module, "_worker", unexpected_worker)
    second = export_set(config)

    assert second["checkpoint"]["records_reused"] == 2
    assert second["checkpoint"]["records_processed"] == 0
    assert second["diagnostics"] == first["diagnostics"]


def test_changed_source_identity_is_reprocessed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="identity")
    first = export_set(config)
    source = next(set_dir.rglob("*.3dc"))
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    export_module = importlib.import_module("pavement_rut.app.export_set")
    original_worker = export_module._worker
    calls = 0

    def counting_worker(payload: object) -> object:
        nonlocal calls
        calls += 1
        return original_worker(payload)

    monkeypatch.setattr(export_module, "_worker", counting_worker)
    second = export_set(config)

    assert first["checkpoint"]["fingerprint"] == second["checkpoint"]["fingerprint"]
    assert second["checkpoint"]["records_reused"] == 0
    assert second["checkpoint"]["records_processed"] == 1
    assert calls == 1


def test_calibration_hash_and_options_change_checkpoint_fingerprint(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    out_dir = tmp_path / "out"
    first = export_set(ExportSetConfig(set_dir=set_dir, out_dir=out_dir, output_name="first"))

    calibration_path = set_dir / "3D_Camera.cal"
    calibration_path.write_text(
        calibration_path.read_text(encoding="utf-8").rstrip() + " 123\n",
        encoding="utf-8",
    )
    second = export_set(ExportSetConfig(set_dir=set_dir, out_dir=out_dir, output_name="second"))
    third = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=out_dir,
            output_name="third",
            options=ProcessingOptions(roll_degrees=0.25),
        )
    )

    assert first["checkpoint"]["fingerprint"] != second["checkpoint"]["fingerprint"]
    assert second["checkpoint"]["fingerprint"] != third["checkpoint"]["fingerprint"]
    manifest = json.loads(Path(third["checkpoint"]["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["processing_schema_version"] == PROCESSING_SCHEMA_VERSION
    assert manifest["options"]["roll_degrees"] == 0.25
    assert manifest["source_set_dir"] == os.path.normcase(str(set_dir.resolve()))
    assert "first,middle,last" in manifest["source_probe"]


def test_checkpoint_repairs_only_a_partial_final_line_and_fails_closed_on_middle_corruption(
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112", file_count=2)
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="repair")
    first = export_set(config)
    journal_path = Path(first["checkpoint"]["journal_jsonl"])
    original = journal_path.read_bytes()
    journal_path.write_bytes(original + b'{"partial":')

    second = export_set(config)
    assert second["checkpoint"]["records_reused"] == 2
    assert journal_path.read_bytes() == original

    lines = original.splitlines(keepends=True)
    journal_path.write_bytes(lines[0] + b"{broken}\n" + b"".join(lines[1:]))
    with pytest.raises(ValueError, match="Corrupt checkpoint JSON.*line 2"):
        export_set(config)
    assert not (journal_path.parent.parent / "RUNNING.lock").exists()


def test_stale_checkpoint_lock_is_not_removed_automatically(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="locked")
    first = export_set(config)
    lock_path = Path(first["checkpoint"]["run_dir"]).parent / "RUNNING.lock"
    lock_path.write_text('{"host":"stale-host","process_id":123}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="remove this stale lock manually"):
        export_set(config)
    assert lock_path.exists()


def test_corrupt_file_is_checkpointed_and_export_finishes_partial(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112", file_count=2)
    corrupt = set_dir / "93" / "11201330005C.3dc"
    corrupt.write_bytes(b"not a QuickLZ block")

    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=tmp_path / "out",
            output_name="partial",
            jobs=2,
            progress_every=1,
        )
    )

    assert metadata["status"] == "partial"
    assert metadata["files_ok"] == 1
    assert metadata["files_failed"] == 1
    assert metadata["diagnostics"][0]["status"] == "ok"
    failed = metadata["diagnostics"][1]
    assert failed["status"] == "failed"
    assert "ThreeDCFormatError" in failed["file_error"]
    assert failed["overall_average_inches"] != failed["overall_average_inches"]

    journal_text = Path(metadata["checkpoint"]["journal_jsonl"]).read_text(encoding="utf-8")
    assert "NaN" not in journal_text
    journal_rows = [json.loads(line, parse_constant=_reject_nonfinite) for line in journal_text.splitlines()]
    assert len(journal_rows) == 2
    failed_checkpoint = next(row for row in journal_rows if row["result"]["status"] == "failed")
    assert failed_checkpoint["result"]["overall_average_inches"] is None


def test_missing_file_in_cached_index_becomes_non_checkpointed_failure(tmp_path: Path) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="missing")
    export_set(config)
    next(set_dir.rglob("*.3dc")).unlink()

    metadata = export_set(config)

    assert metadata["status"] == "partial"
    assert metadata["files_failed"] == 1
    assert metadata["checkpoint"]["records_not_checkpointed"] == 1
    assert metadata["diagnostics"][0]["status"] == "failed"
    assert "FileNotFoundError" in metadata["diagnostics"][0]["file_error"]


def test_batch_rejects_duplicate_requested_sets(tmp_path: Path) -> None:
    _write_synthetic_set(tmp_path, "112")

    with pytest.raises(ValueError, match="contain duplicates: 112"):
        discover_sets(tmp_path, ("112", "112"))


def test_batch_marks_partial_set_failed_but_preserves_output_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    set_dir = _write_synthetic_set(data_root, "112", file_count=2)
    (set_dir / "93" / "11201330005C.3dc").write_bytes(b"corrupt")

    summary = run_batch(BatchConfig(data_root=data_root, out_dir=tmp_path / "out", progress_every=1))

    assert summary["sets_failed"] == 1
    assert summary["files_failed"] == 1
    assert summary["results"][0]["status"] == "failed"
    assert Path(summary["results"][0]["metadata_json"]).exists()


def test_metadata_preserving_source_change_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="probe")
    export_set(config)

    source = next(set_dir.rglob("*.3dc"))
    original_stat = source.stat()
    payload = bytearray(source.read_bytes())
    payload[32] ^= 1  # Intensity data: keep the QuickLZ container and heights valid.
    source.write_bytes(payload)
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    export_module = importlib.import_module("pavement_rut.app.export_set")
    original_worker = export_module._worker
    calls = 0

    def counting_worker(payload: object) -> object:
        nonlocal calls
        calls += 1
        return original_worker(payload)

    monkeypatch.setattr(export_module, "_worker", counting_worker)
    metadata = export_set(config)

    assert metadata["checkpoint"]["records_reused"] == 0
    assert metadata["checkpoint"]["records_processed"] == 1
    assert calls == 1


def test_same_set_label_in_different_data_roots_has_a_distinct_checkpoint_run(
    tmp_path: Path,
) -> None:
    first_set = _write_synthetic_set(tmp_path / "data_a", "112")
    second_set = _write_synthetic_set(tmp_path / "data_b", "112")
    checkpoint_root = tmp_path / "shared_checkpoints"

    first = export_set(
        ExportSetConfig(
            set_dir=first_set,
            out_dir=tmp_path / "out_a",
            checkpoint_dir=checkpoint_root,
            output_name="first",
        )
    )
    second = export_set(
        ExportSetConfig(
            set_dir=second_set,
            out_dir=tmp_path / "out_b",
            checkpoint_dir=checkpoint_root,
            output_name="second",
        )
    )

    assert first["checkpoint"]["fingerprint"] != second["checkpoint"]["fingerprint"]
    assert first["checkpoint"]["run_dir"] != second["checkpoint"]["run_dir"]
    assert second["checkpoint"]["records_reused"] == 0


def test_source_change_during_processing_aborts_without_checkpointing_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    out_dir = tmp_path / "out"
    source = next(set_dir.rglob("*.3dc"))
    export_module = importlib.import_module("pavement_rut.app.export_set")
    original_worker = export_module._worker

    def mutating_worker(payload: object) -> object:
        result = original_worker(payload)
        original_stat = source.stat()
        contents = bytearray(source.read_bytes())
        contents[32] ^= 1
        source.write_bytes(contents)
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return result

    monkeypatch.setattr(export_module, "_worker", mutating_worker)
    with pytest.raises(RuntimeError, match="changed while it was processed"):
        export_set(
            ExportSetConfig(
                set_dir=set_dir,
                out_dir=out_dir,
                output_name="unstable",
            )
        )

    journals = list((out_dir / ".checkpoints").rglob("results.jsonl"))
    assert len(journals) == 1
    assert journals[0].read_bytes() == b""
    assert not list((out_dir / ".checkpoints").rglob("RUNNING.lock"))


def test_incomplete_checkpoint_result_is_retried_even_without_file_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    config = ExportSetConfig(set_dir=set_dir, out_dir=tmp_path / "out", output_name="retry")
    first = export_set(config)
    journal_path = Path(first["checkpoint"]["journal_jsonl"])
    row = json.loads(journal_path.read_text(encoding="utf-8"))
    row["result"]["status"] = "failed"
    row["result"]["file_error"] = None
    journal_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )

    export_module = importlib.import_module("pavement_rut.app.export_set")
    original_worker = export_module._worker
    calls = 0

    def counting_worker(payload: object) -> object:
        nonlocal calls
        calls += 1
        return original_worker(payload)

    monkeypatch.setattr(export_module, "_worker", counting_worker)
    second = export_set(config)

    assert second["checkpoint"]["records_reused"] == 0
    assert second["checkpoint"]["records_processed"] == 1
    assert calls == 1


def test_programming_error_is_not_silently_converted_to_partial_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    out_dir = tmp_path / "out"
    export_module = importlib.import_module("pavement_rut.app.export_set")

    def broken_algorithm(*args: object, **kwargs: object) -> object:
        raise TypeError("simulated programming defect")

    monkeypatch.setattr(export_module, "measure_profile_rutting", broken_algorithm)
    with pytest.raises(TypeError, match="programming defect"):
        export_set(
            ExportSetConfig(
                set_dir=set_dir,
                out_dir=out_dir,
                output_name="must-not-exist",
            )
        )

    assert not (out_dir / "must-not-exist.metadata.json").exists()
    assert not list((out_dir / ".checkpoints").rglob("RUNNING.lock"))


def test_all_profiles_skipped_as_noisy_is_a_failed_file_not_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    export_module = importlib.import_module("pavement_rut.app.export_set")

    class NoisyProfile:
        is_noisy = True

    monkeypatch.setattr(export_module, "reduce_profile", lambda profile, config: NoisyProfile())
    monkeypatch.setattr(
        export_module,
        "measure_profile_rutting",
        lambda *args, **kwargs: pytest.fail("a skipped noisy profile must not be measured"),
    )
    metadata = export_set(
        ExportSetConfig(
            set_dir=set_dir,
            out_dir=tmp_path / "out",
            output_name="all-noisy",
            options=ProcessingOptions(skip_noisy_profiles=True),
        )
    )

    result = metadata["diagnostics"][0]
    assert metadata["status"] == "partial"
    assert metadata["files_failed"] == 1
    assert result["status"] == "failed"
    assert result["rows_noisy"] == 1
    assert result["rows_ok"] == 0
    assert result["severity"] == -1


@pytest.mark.parametrize("relative_path", ["../escape.3dc", r"..\escape.3dc", "/abs.3dc", r"C:\abs.3dc"])
def test_checkpoint_identity_rejects_posix_and_windows_path_traversal(relative_path: str) -> None:
    with pytest.raises(ValueError, match="Unsafe checkpoint relative path"):
        RecordIdentity(
            relative_path=relative_path,
            start_frame=1.0,
            end_frame=2.0,
            size_bytes=1,
            mtime_ns=1,
            ctime_ns=1,
            source_probe_sha256="0" * 64,
        )


def test_atomic_json_failure_preserves_old_destination_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    export_module = importlib.import_module("pavement_rut.app.export_set")
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}', encoding="utf-8")

    def failed_replace(source: object, target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(export_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        export_module._write_json_atomic(destination, {"new": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.iterdir()) == [destination]


def test_batch_parent_isolates_unserialized_child_failure_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_synthetic_set(data_root, "112")
    batch_module = importlib.import_module("pavement_rut.app.batch")

    class FailedFuture:
        def result(self) -> object:
            raise RuntimeError("child exited before returning a result")

    class FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, payload: object) -> FailedFuture:
            return FailedFuture()

    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(batch_module, "as_completed", lambda futures: list(futures))
    summary = run_batch(BatchConfig(data_root=data_root, out_dir=tmp_path / "out", jobs=2))

    assert summary["sets_failed"] == 1
    assert "child exited" in summary["results"][0]["error"]
    persisted = json.loads(
        Path(summary["outputs"]["json"]).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    assert persisted["sets_failed"] == 1


def test_cli_export_stdout_is_one_strict_json_document(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    set_dir = _write_synthetic_set(tmp_path / "data", "112")
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "export-set",
                "--set-dir",
                str(set_dir),
                "--out-dir",
                str(tmp_path / "out"),
                "--output-name",
                "cli",
            ]
        )

    assert caught.value.code == 0
    captured = capsys.readouterr()
    outputs = json.loads(captured.out, parse_constant=_reject_nonfinite)
    assert Path(outputs["metadata_json"]).exists()
    assert "[INFO]" not in captured.out
    assert "[INFO]" in captured.err
