"""Numba kernels for optional CPU acceleration.

This module is imported only when Numba is installed.  Keep constants and
helpers beside the cached kernels so Numba's cache invalidation sees every
implementation detail that can affect generated machine code.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_HULL_ROUNDING_TOLERANCE = 32.0 * np.finfo(np.float64).eps
_HASH_VALUES = 4096
_UNCONDITIONAL_MATCH_LENGTH = 6
_UNCOMPRESSED_END = 4
_LOW_SLOPE_THRESHOLD = 0.055
_HIGH_SLOPE_THRESHOLD = 0.17
_MINIMUM_STEEP_SEGMENTS = 5
_FOOTPRINT_ROOT_TOLERANCE = 64.0 * np.finfo(np.float64).eps

QLZ_OK = 0
QLZ_ZERO_CONTROL_WORD = 1
QLZ_TRUNCATED_CONTROL_WORD = 2
QLZ_TRUNCATED_TOKEN = 3
QLZ_TRUNCATED_EXTENDED_LENGTH = 4
QLZ_INVALID_MATCH_OFFSET = 5
QLZ_INVALID_MATCH_LENGTH = 6
QLZ_TRUNCATED_LITERAL = 7
QLZ_TRUNCATED_LOOKAHEAD = 8
QLZ_TRUNCATED_FINAL_CONTROL_WORD = 9
QLZ_TRUNCATED_FINAL_LITERAL = 10
QLZ_INVALID_HASH_SEQUENCE = 11


@njit(cache=True, fastmath=False, nogil=True)
def upper_concave_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the exact hull produced by the reference scalar loop."""

    hull = np.empty(x.size, dtype=np.int64)
    hull_size = 0
    for index in range(x.size):
        x_index = float(x[index])
        y_index = float(y[index])
        while hull_size >= 2:
            first = hull[hull_size - 2]
            second = hull[hull_size - 1]
            first_x = float(x[first])
            second_x = float(x[second])
            first_y = float(y[first])
            second_y = float(y[second])
            first_term = (second_x - first_x) * (y_index - second_y)
            second_term = (second_y - first_y) * (x_index - second_x)
            cross = first_term - second_term
            scale = max(1.0, abs(first_term), abs(second_term))
            if cross >= -_HULL_ROUNDING_TOLERANCE * scale:
                hull_size -= 1
            else:
                break
        hull[hull_size] = index
        hull_size += 1
    return hull[:hull_size]


@njit(cache=True, inline="always", fastmath=False)
def _searchsorted_left(values: np.ndarray, target: float) -> int:
    low = 0
    high = values.size
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True, inline="always", fastmath=False)
def _searchsorted_right(values: np.ndarray, target: float) -> int:
    low = 0
    high = values.size
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True, fastmath=False, nogil=True)
def shoulder_trim_indices(
    x: np.ndarray,
    y: np.ndarray,
    lane_left: float,
    lane_right: float,
    lane_center: float,
    profile_width: float,
) -> tuple[int, int]:
    """Return the exact retained half-open slice without temporary arrays."""

    point_count = x.size
    lane_start = _searchsorted_right(x, lane_left)
    lane_stop = _searchsorted_left(x, lane_right)
    if point_count < 2:
        if lane_start < lane_stop:
            return lane_start, lane_stop
        return lane_start, lane_start

    center_offset = 0.35 * profile_width
    left_edge_limit = _searchsorted_left(x, lane_center - center_offset)
    right_edge_limit = _searchsorted_right(x, lane_center + center_offset) - 1

    left_best_start = -1
    left_best_end = -1
    right_best_start = point_count
    run_start = -1
    run_has_high = False

    for segment in range(point_count - 1):
        slope = abs((y[segment + 1] - y[segment]) / (x[segment + 1] - x[segment]))
        is_low = slope > _LOW_SLOPE_THRESHOLD
        if is_low:
            if run_start < 0:
                run_start = segment
                run_has_high = False
            if slope >= _HIGH_SLOPE_THRESHOLD:
                run_has_high = True

        is_last_segment = segment == point_count - 2
        if run_start >= 0 and (not is_low or is_last_segment):
            run_end = segment if is_low and is_last_segment else segment - 1
            run_length = run_end - run_start + 1
            if run_length >= _MINIMUM_STEEP_SEGMENTS and run_has_high:
                if run_start <= left_edge_limit and run_start > left_best_start:
                    left_best_start = run_start
                    left_best_end = run_end
                if run_end >= right_edge_limit and run_start < right_best_start:
                    right_best_start = run_start
            run_start = -1
            run_has_high = False

    shoulder_start = left_best_end + 2 if left_best_start >= 0 else 0
    shoulder_stop = right_best_start if right_best_start < point_count else point_count
    retained_start = max(shoulder_start, lane_start)
    retained_stop = min(shoulder_stop, lane_stop)
    if retained_start >= retained_stop:
        empty_at = min(retained_start, point_count)
        return empty_at, empty_at
    return retained_start, retained_stop


@njit(cache=True, inline="always", fastmath=False)
def _piecewise_value(x: np.ndarray, y: np.ndarray, query: float) -> float:
    position = _searchsorted_right(x, query) - 1
    if position >= x.size - 1:
        return float(y[-1])
    offset = query - x[position]
    slope = (y[position + 1] - y[position]) / (x[position + 1] - x[position])
    return float(y[position] + slope * offset)


@njit(cache=True, inline="always", fastmath=False)
def _piecewise_integral(
    x: np.ndarray,
    y: np.ndarray,
    prefix_area: np.ndarray,
    query: float,
) -> float:
    position = _searchsorted_right(x, query) - 1
    if position >= x.size - 1:
        position = x.size - 2
    offset = query - x[position]
    segment_slope = (y[position + 1] - y[position]) / (x[position + 1] - x[position])
    return float(prefix_area[position] + y[position] * offset + 0.5 * segment_slope * (offset * offset))


@njit(cache=True, fastmath=False, nogil=True)
def maximum_footprint_gap(
    x: np.ndarray,
    y: np.ndarray,
    center_left: float,
    center_right: float,
    measurement_width: float,
    bar_slope: float,
    bar_y_at_wheel_center: float,
    wheel_center: float,
) -> tuple[float, float, float, int]:
    """Fused exact-order footprint candidate search for one wheel path."""

    half_width = measurement_width / 2.0
    raw_breaks = np.empty(2 * x.size + 2, dtype=np.float64)
    raw_count = 0
    raw_breaks[raw_count] = center_left
    raw_count += 1
    for index in range(x.size):
        value = x[index] - half_width
        if value > center_left and value < center_right:
            raw_breaks[raw_count] = value
            raw_count += 1
    for index in range(x.size):
        value = x[index] + half_width
        if value > center_left and value < center_right:
            raw_breaks[raw_count] = value
            raw_count += 1
    raw_breaks[raw_count] = center_right
    raw_count += 1

    sorted_breaks = np.sort(raw_breaks[:raw_count])
    breaks = np.empty(raw_count, dtype=np.float64)
    break_count = 0
    for index in range(raw_count):
        value = sorted_breaks[index]
        if break_count == 0 or value != breaks[break_count - 1]:
            breaks[break_count] = value
            break_count += 1

    derivatives = np.empty(break_count, dtype=np.float64)
    for index in range(break_count):
        center = breaks[index]
        left_y = _piecewise_value(x, y, center - half_width)
        right_y = _piecewise_value(x, y, center + half_width)
        derivatives[index] = bar_slope - (right_y - left_y) / measurement_width

    prefix_area = np.empty(x.size, dtype=np.float64)
    prefix_area[0] = 0.0
    for index in range(x.size - 1):
        segment_area = 0.5 * (y[index] + y[index + 1]) * (x[index + 1] - x[index])
        prefix_area[index + 1] = prefix_area[index] + segment_area

    normalizer = np.sqrt(1.0 + bar_slope**2)
    best_x = breaks[0]
    best_average_y = 0.0
    best_gap = -1.0

    for interval in range(break_count):
        candidate = breaks[interval]
        right_integral = _piecewise_integral(x, y, prefix_area, candidate + half_width)
        left_integral = _piecewise_integral(x, y, prefix_area, candidate - half_width)
        average_y = (right_integral - left_integral) / measurement_width
        bar_y = bar_y_at_wheel_center + bar_slope * (candidate - wheel_center)
        gap = max(0.0, (bar_y - average_y) / normalizer)
        if gap > best_gap:
            best_x = candidate
            best_average_y = average_y
            best_gap = gap

        if interval + 1 >= break_count:
            continue
        interval_width = breaks[interval + 1] - breaks[interval]
        derivative_change = derivatives[interval + 1] - derivatives[interval]
        has_root = (
            interval_width > 0.0
            and derivatives[interval] * derivatives[interval + 1] <= 0.0
            and abs(derivative_change) > _FOOTPRINT_ROOT_TOLERANCE
        )
        if not has_root:
            continue
        root = breaks[interval] - derivatives[interval] * (interval_width / derivative_change)
        if root <= breaks[interval] or root >= breaks[interval + 1]:
            continue
        right_integral = _piecewise_integral(x, y, prefix_area, root + half_width)
        left_integral = _piecewise_integral(x, y, prefix_area, root - half_width)
        average_y = (right_integral - left_integral) / measurement_width
        bar_y = bar_y_at_wheel_center + bar_slope * (root - wheel_center)
        gap = max(0.0, (bar_y - average_y) / normalizer)
        if gap > best_gap:
            best_x = root
            best_average_y = average_y
            best_gap = gap

    footprint_count = 0
    footprint_left = best_x - half_width
    footprint_right = best_x + half_width
    for value in x:
        if value >= footprint_left and value <= footprint_right:
            footprint_count += 1
    return best_x, best_average_y, best_gap, footprint_count


@njit(cache=True, inline="always", fastmath=False)
def _read_little_unchecked(source: np.ndarray, position: int, size: int) -> int:
    value = 0
    for index in range(size):
        value |= int(source[position + index]) << (index * 8)
    return value


@njit(cache=True, fastmath=False, nogil=True)
def decompress_quicklz_level1(
    source: np.ndarray,
    header_size: int,
    output_size: int,
) -> tuple[np.ndarray, int, int, int]:
    """Decode a validated level-1 block and return output plus error details.

    The tuple is ``(destination, status, position, value)``.  Bounds and token
    validity remain explicit even though the hot loop runs without Python.
    """

    source_end = source.size
    destination = np.empty(output_size, dtype=np.uint8)
    hash_table = np.zeros(_HASH_VALUES, dtype=np.int64)
    source_position = header_size
    destination_position = 0
    control_word = 1
    last_hashed = -1
    fetch = 0
    last_match_start = output_size - _UNCONDITIONAL_MATCH_LENGTH - _UNCOMPRESSED_END - 1

    while True:
        if control_word == 1:
            if source_position < 0 or source_position + 4 > source_end:
                return destination, QLZ_TRUNCATED_CONTROL_WORD, source_position, 0
            control_word = _read_little_unchecked(source, source_position, 4)
            source_position += 4
            if control_word == 0:
                return destination, QLZ_ZERO_CONTROL_WORD, source_position - 4, 0
            if destination_position <= last_match_start:
                if source_position + 3 > source_end:
                    return destination, QLZ_TRUNCATED_TOKEN, source_position, 0
                fetch = _read_little_unchecked(source, source_position, 3)

        if control_word & 1:
            control_word >>= 1
            hash_value = (fetch >> 4) & (_HASH_VALUES - 1)
            match_position = hash_table[hash_value]

            if fetch & 0x0F:
                match_length = (fetch & 0x0F) + 2
                if source_position + 2 > source_end:
                    return destination, QLZ_TRUNCATED_TOKEN, source_position, 0
                source_position += 2
            else:
                if source_position + 3 > source_end:
                    return destination, QLZ_TRUNCATED_EXTENDED_LENGTH, source_position + 2, 0
                match_length = int(source[source_position + 2])
                source_position += 3

            if match_position < 0 or match_position > destination_position - 3:
                return destination, QLZ_INVALID_MATCH_OFFSET, destination_position, match_position
            if match_length < 3 or destination_position + match_length > output_size:
                return destination, QLZ_INVALID_MATCH_LENGTH, destination_position, match_length

            match_start = destination_position
            for index in range(match_length):
                destination[destination_position + index] = destination[match_position + index]
            destination_position += match_length

            while last_hashed < match_start:
                last_hashed += 1
                if last_hashed + 3 > output_size:
                    return destination, QLZ_INVALID_HASH_SEQUENCE, last_hashed, 0
                hash_fetch = _read_little_unchecked(destination, last_hashed, 3)
                hash_value = ((hash_fetch >> 12) ^ hash_fetch) & (_HASH_VALUES - 1)
                hash_table[hash_value] = last_hashed
            last_hashed = destination_position - 1
            if source_position + 3 > source_end:
                return destination, QLZ_TRUNCATED_TOKEN, source_position, 0
            fetch = _read_little_unchecked(source, source_position, 3)
            continue

        if destination_position <= last_match_start:
            if source_position >= source_end:
                return destination, QLZ_TRUNCATED_LITERAL, source_position, 0
            destination[destination_position] = source[source_position]
            destination_position += 1
            source_position += 1
            control_word >>= 1

            through = destination_position - 3
            while last_hashed < through:
                last_hashed += 1
                if last_hashed + 3 > output_size:
                    return destination, QLZ_INVALID_HASH_SEQUENCE, last_hashed, 0
                hash_fetch = _read_little_unchecked(destination, last_hashed, 3)
                hash_value = ((hash_fetch >> 12) ^ hash_fetch) & (_HASH_VALUES - 1)
                hash_table[hash_value] = last_hashed
            if source_position + 3 > source_end:
                return destination, QLZ_TRUNCATED_LOOKAHEAD, source_position + 2, 0
            next_byte = int(source[source_position + 2])
            fetch = ((fetch >> 8) & 0xFFFF) | (next_byte << 16)
            continue

        while destination_position < output_size:
            if control_word == 1:
                if source_position + 4 > source_end:
                    return destination, QLZ_TRUNCATED_FINAL_CONTROL_WORD, source_position, 0
                source_position += 4
                control_word = 0x80000000
            if source_position >= source_end:
                return destination, QLZ_TRUNCATED_FINAL_LITERAL, source_position, 0
            destination[destination_position] = source[source_position]
            destination_position += 1
            source_position += 1
            control_word >>= 1
        return destination, QLZ_OK, 0, 0
