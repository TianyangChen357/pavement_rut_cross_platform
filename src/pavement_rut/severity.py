"""Rutting summary helpers shared by CLI and exporters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.mean(array)) if array.size else math.nan


def severity_from_average(value: float) -> int:
    if not math.isfinite(value):
        return -1
    if value < 0.25:
        return 0
    if value < 0.5:
        return 1
    if value < 1.0:
        return 2
    return 3
