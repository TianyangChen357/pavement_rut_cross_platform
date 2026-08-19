from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_full_export.py"
SPEC = importlib.util.spec_from_file_location("run_full_export_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
performance_metrics = TOOL.performance_metrics


def test_performance_metrics_use_standard_speedup_and_efficiency() -> None:
    metrics = performance_metrics(
        workers=8,
        wall_seconds=16.0,
        aggregate_cpu_seconds=80.0,
    )

    assert metrics.cpu_time_derived_speedup == pytest.approx(5.0)
    assert metrics.efficiency == pytest.approx(0.625)
    assert metrics.efficiency_percent == pytest.approx(62.5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("workers", 0, "workers"),
        ("wall_seconds", 0.0, "wall"),
        ("aggregate_cpu_seconds", float("nan"), "CPU"),
    ],
)
def test_performance_metrics_reject_invalid_inputs(field: str, value: float, message: str) -> None:
    values = {"workers": 1, "wall_seconds": 1.0, "aggregate_cpu_seconds": 1.0}
    values[field] = value

    with pytest.raises(ValueError, match=message):
        performance_metrics(**values)
