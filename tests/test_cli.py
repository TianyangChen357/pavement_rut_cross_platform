from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pavement_rut import cli


def _reject_nonfinite(value: str) -> None:
    raise ValueError(value)


def test_export_set_data_root_requires_set(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["export-set", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out")])

    assert caught.value.code == 2
    assert "--data-root requires --set" in capsys.readouterr().err


@pytest.mark.parametrize("set_number", ["../112", r"..\112", "set112", "1/2"])
def test_export_set_data_root_rejects_non_numeric_set(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    set_number: str,
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "export-set",
                "--data-root",
                str(tmp_path),
                "--set",
                set_number,
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )

    assert caught.value.code == 2
    assert "--set must be a numeric set directory name" in capsys.readouterr().err


def test_batch_returns_one_when_any_set_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_batch",
        lambda config: {
            "sets_failed": 1,
            "outputs": {"csv": str(tmp_path / "summary.csv"), "json": str(tmp_path / "summary.json")},
        },
    )

    with pytest.raises(SystemExit) as caught:
        cli.main(["batch", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out")])

    assert caught.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["json"].endswith("summary.json")


def test_export_partial_returns_one_and_checkpoint_flags_reach_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: list[object] = []

    def fake_export(config: object) -> dict[str, object]:
        captured.append(config)
        return {
            "status": "partial",
            "outputs": {"geojson": "a.geojson", "csv": "a.csv", "metadata_json": "a.metadata.json"},
        }

    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(cli, "export_set", fake_export)
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "export-set",
                "--set-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--no-resume",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--checkpoint-every",
                "7",
            ]
        )

    assert caught.value.code == 1
    assert len(captured) == 1
    config = captured[0]
    assert config.resume is False
    assert config.checkpoint_dir == checkpoint_dir
    assert config.checkpoint_every == 7
    assert json.loads(capsys.readouterr().out)["metadata_json"] == "a.metadata.json"


def test_invalid_processing_option_exits_two_before_export(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "export_set", lambda config: pytest.fail("export_set must not run"))

    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "export-set",
                "--set-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--rut-bar-inches",
                "-1",
            ]
        )

    assert caught.value.code == 2
    assert "rut_bar_length_inches must be positive" in capsys.readouterr().err


def test_inspect_json_uses_null_for_nonfinite_rut_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.3dc"
    calibration_path = tmp_path / "3D_Camera.cal"
    image_path.write_bytes(b"fake")
    calibration_path.write_text("fake", encoding="utf-8")

    calibration = SimpleNamespace(
        pavement_width_inches=4.0,
        pixel_width_inches=1.0,
        apply=lambda raw: np.asarray(raw, dtype=np.float64),
    )
    image = SimpleNamespace(
        raw_heights=np.array([[1, 2, 3, 4]], dtype=np.uint16),
        profile_count=1,
        columns=4,
    )
    reduced = SimpleNamespace(point_count=4, is_noisy=False)
    wheel = SimpleNamespace(rut_depth_inches=float("nan"))
    rutting = SimpleNamespace(left=wheel, right=wheel, overall_rut_depth_inches=float("nan"))
    cross_slope = SimpleNamespace(percent=float("nan"), angle_degrees=float("nan"))
    monkeypatch.setattr(cli, "load_calibration", lambda path: calibration)
    monkeypatch.setattr(cli, "read_3dc", lambda path: image)
    monkeypatch.setattr(cli, "reduce_profile", lambda profile, config: reduced)
    monkeypatch.setattr(cli, "measure_profile_rutting", lambda profile, lane, bar_length_inches: rutting)
    monkeypatch.setattr(cli, "fit_cross_slope", lambda profile, lane: cross_slope)

    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "inspect-file",
                str(image_path),
                "--calibration",
                str(calibration_path),
                "--json",
            ]
        )

    assert caught.value.code == 0
    raw_output = capsys.readouterr().out
    payload = json.loads(raw_output, parse_constant=_reject_nonfinite)
    assert payload["left_rut_inches"] is None
    assert payload["right_rut_inches"] is None
    assert payload["overall_rut_inches"] is None
    assert payload["cross_slope_percent"] is None
    assert payload["cross_slope_angle_degrees"] is None


def test_inspect_custom_lane_does_not_crop_reduction_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.3dc"
    calibration_path = tmp_path / "3D_Camera.cal"
    image_path.write_bytes(b"fake")
    calibration_path.write_text("fake", encoding="utf-8")

    columns = 1536
    calibration = SimpleNamespace(
        pavement_width_inches=162.0,
        pixel_width_inches=162.0 / columns,
        apply=lambda raw: np.asarray(raw, dtype=np.float64),
    )
    image = SimpleNamespace(
        raw_heights=np.full((1, columns), 1000, dtype=np.uint16),
        profile_count=1,
        columns=columns,
    )
    wheel = SimpleNamespace(rut_depth_inches=0.0)
    rutting = SimpleNamespace(left=wheel, right=wheel, overall_rut_depth_inches=0.0)
    cross_slope = SimpleNamespace(percent=0.0, angle_degrees=0.0)
    monkeypatch.setattr(cli, "load_calibration", lambda path: calibration)
    monkeypatch.setattr(cli, "read_3dc", lambda path: image)
    monkeypatch.setattr(cli, "measure_profile_rutting", lambda profile, lane, bar_length_inches: rutting)
    monkeypatch.setattr(cli, "fit_cross_slope", lambda profile, lane: cross_slope)

    def inspect(*extra: str) -> dict[str, object]:
        with pytest.raises(SystemExit) as caught:
            cli.main(
                [
                    "inspect-file",
                    str(image_path),
                    "--calibration",
                    str(calibration_path),
                    "--json",
                    *extra,
                ]
            )
        assert caught.value.code == 0
        return json.loads(capsys.readouterr().out)

    default = inspect()
    custom = inspect("--lane-left-inches", "40", "--lane-right-inches", "120")

    assert custom["reduced_points"] == default["reduced_points"]
