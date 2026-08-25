"""Optional Numba acceleration with a transparent pure-Python fallback."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

_DISABLE_VALUES = {"1", "true", "yes", "on"}
_disabled_by_environment = os.environ.get("PAVEMENT_RUT_DISABLE_JIT", "").strip().lower() in _DISABLE_VALUES

try:
    if _disabled_by_environment:
        raise ImportError("JIT disabled by PAVEMENT_RUT_DISABLE_JIT")
    from pavement_rut import _numba_kernels
except Exception as exc:  # pragma: no cover - depends on optional local runtime
    _numba_kernels = None
    _unavailable_reason = str(exc)
else:
    _unavailable_reason = None

_runtime_failure: Exception | None = None


@dataclass(frozen=True, slots=True)
class AccelerationStatus:
    """Current optional-acceleration state."""

    available: bool
    active: bool
    reason: str | None


def acceleration_status() -> AccelerationStatus:
    """Report whether Numba kernels are installed and usable."""

    if _numba_kernels is None:
        return AccelerationStatus(False, False, _unavailable_reason)
    if _runtime_failure is not None:
        return AccelerationStatus(True, False, f"JIT initialization failed: {_runtime_failure}")
    return AccelerationStatus(True, True, None)


def try_upper_concave_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Return an accelerated hull, or ``None`` when fallback is required."""

    global _runtime_failure
    if _numba_kernels is None or _runtime_failure is not None:
        return None
    try:
        return _numba_kernels.upper_concave_hull_indices(x, y)
    except Exception as exc:  # pragma: no cover - depends on local compiler/runtime
        _runtime_failure = exc
        return None


def try_shoulder_trim_indices(
    x: np.ndarray,
    y: np.ndarray,
    lane_left: float,
    lane_right: float,
    lane_center: float,
    profile_width: float,
) -> tuple[int, int] | None:
    """Return accelerated shoulder bounds, or ``None`` for fallback."""

    global _runtime_failure
    if _numba_kernels is None or _runtime_failure is not None:
        return None
    try:
        return _numba_kernels.shoulder_trim_indices(
            x,
            y,
            lane_left,
            lane_right,
            lane_center,
            profile_width,
        )
    except Exception as exc:  # pragma: no cover - depends on local compiler/runtime
        _runtime_failure = exc
        return None


def try_maximum_footprint_gap(
    x: np.ndarray,
    y: np.ndarray,
    center_left: float,
    center_right: float,
    measurement_width: float,
    bar_slope: float,
    bar_y_at_wheel_center: float,
    wheel_center: float,
) -> tuple[float, float, float, int] | None:
    """Return an accelerated footprint result, or ``None`` for fallback."""

    global _runtime_failure
    if _numba_kernels is None or _runtime_failure is not None:
        return None
    try:
        return _numba_kernels.maximum_footprint_gap(
            x,
            y,
            center_left,
            center_right,
            measurement_width,
            bar_slope,
            bar_y_at_wheel_center,
            wheel_center,
        )
    except Exception as exc:  # pragma: no cover - depends on local compiler/runtime
        _runtime_failure = exc
        return None


def try_decompress_quicklz_level1(
    source: np.ndarray,
    header_size: int,
    output_size: int,
) -> tuple[np.ndarray, int, int, int] | None:
    """Run the accelerated decoder, or return ``None`` to request fallback."""

    global _runtime_failure
    if _numba_kernels is None or _runtime_failure is not None:
        return None
    try:
        return _numba_kernels.decompress_quicklz_level1(source, header_size, output_size)
    except Exception as exc:  # pragma: no cover - depends on local compiler/runtime
        _runtime_failure = exc
        return None


def warm_jit_kernels() -> AccelerationStatus:
    """Compile/load all kernels before worker processes begin doing work."""

    if not acceleration_status().active:
        return acceleration_status()

    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if try_upper_concave_hull_indices(x, y) is None:
        return acceleration_status()

    if try_shoulder_trim_indices(x, y, -1.0, 3.0, 1.0, 4.0) is None:
        return acceleration_status()

    footprint_result = try_maximum_footprint_gap(x, y, 0.5, 1.5, 0.5, 0.0, 0.0, 1.0)
    if footprint_result is None:
        return acceleration_status()

    body = bytes.fromhex("08 00 00 80") + b"abc" + bytes.fromhex("77 45") + b"WXYZ"
    block = bytes([0x47]) + struct.pack("<II", len(body) + 9, 16) + body
    source = np.frombuffer(block, dtype=np.uint8)
    result = try_decompress_quicklz_level1(source, 9, 16)
    if result is None:
        return acceleration_status()
    output, status, _, _ = result
    if status != 0 or output.tobytes() != b"abcabcabcabcWXYZ":
        global _runtime_failure
        _runtime_failure = RuntimeError("JIT QuickLZ self-test produced an unexpected result")
    return acceleration_status()


__all__ = [
    "AccelerationStatus",
    "acceleration_status",
    "try_decompress_quicklz_level1",
    "try_maximum_footprint_gap",
    "try_shoulder_trim_indices",
    "try_upper_concave_hull_indices",
    "warm_jit_kernels",
]
