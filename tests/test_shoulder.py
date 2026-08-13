import numpy as np
import pytest

from pavement_rut.domain.models import LaneGeometry, ReducedProfile
from pavement_rut.domain.shoulder import remove_shoulders, shoulder_trim_indices


def _profile_from_slopes(slopes: np.ndarray, *, profile_id: str | None = None) -> ReducedProfile:
    slopes = np.asarray(slopes, dtype=np.float64)
    x = 100.0 * np.arange(slopes.size + 1, dtype=np.float64)
    y = np.concatenate(([0.0], np.cumsum(100.0 * slopes)))
    return ReducedProfile(x, y, profile_id=profile_id)


def _whole_profile_lane(profile: ReducedProfile) -> LaneGeometry:
    return LaneGeometry(float(profile.x_inches[0] - 1.0), float(profile.x_inches[-1] + 1.0))


def test_shoulder_helpers_are_exported_from_domain_package() -> None:
    from pavement_rut import domain

    assert domain.remove_shoulders is remove_shoulders
    assert domain.shoulder_trim_indices is shoulder_trim_indices


def test_flat_profile_uses_strict_open_lane_edges() -> None:
    profile = ReducedProfile(np.arange(11.0), np.zeros(11))
    lane = LaneGeometry(2.0, 8.0)

    assert shoulder_trim_indices(profile, lane) == (3, 8)
    trimmed = remove_shoulders(profile, lane)
    np.testing.assert_array_equal(trimmed.x_inches, np.arange(3.0, 8.0))


def test_left_and_right_runs_map_to_the_observed_cut_points() -> None:
    slopes = np.zeros(100)
    slopes[5:10] = 0.2
    slopes[90:95] = -0.2
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile)) == (11, 90)


@pytest.mark.parametrize(
    ("run_length", "expected_start"),
    [(4, 0), (5, 12)],
)
def test_high_slope_must_span_at_least_five_contiguous_segments(
    run_length: int,
    expected_start: int,
) -> None:
    slopes = np.zeros(100)
    slopes[6 : 6 + run_length] = 0.2
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile))[0] == expected_start


@pytest.mark.parametrize(
    ("peak_slope", "expected_start"),
    [(0.169999, 0), (0.17, 11)],
)
def test_high_slope_trigger_is_inclusive_at_seventeen_percent(
    peak_slope: float,
    expected_start: int,
) -> None:
    slopes = np.zeros(100)
    slopes[5:10] = 0.1
    slopes[5] = peak_slope
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile))[0] == expected_start


@pytest.mark.parametrize(
    ("toe_slope", "expected_start"),
    [(0.055, 11), (0.055001, 12)],
)
def test_low_slope_toe_threshold_is_strict(
    toe_slope: float,
    expected_start: int,
) -> None:
    slopes = np.zeros(100)
    slopes[5:10] = 0.1
    slopes[5] = 0.2
    slopes[10] = toe_slope
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile))[0] == expected_start


def test_search_band_is_inclusive_and_innermost_runs_win() -> None:
    slopes = np.zeros(100)
    slopes[2:7] = 0.2
    slopes[15:20] = 0.2  # start == round(101 * 0.15), so it is eligible
    slopes[82:87] = 0.2
    slopes[92:97] = 0.2
    profile = _profile_from_slopes(slopes)

    # The inner left run ends at segment 19, and the inner right run starts at
    # segment 82.  The retained point slice is therefore [21:82].
    assert shoulder_trim_indices(profile, _whole_profile_lane(profile)) == (21, 82)


def test_run_starting_beyond_left_search_band_is_ignored() -> None:
    slopes = np.zeros(100)
    slopes[16:21] = 0.2
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile)) == (0, 101)


def test_geometric_shoulder_result_is_intersected_with_lane_clip() -> None:
    slopes = np.zeros(100)
    slopes[5:10] = 0.2
    profile = _profile_from_slopes(slopes)
    lane = LaneGeometry(2_000.0, 8_000.0)

    # The steep run lies outside this lane.  Its geometric cut is outside the
    # lane too, so the final intersection retains indices 21 through 79.
    assert shoulder_trim_indices(profile, lane) == (21, 80)


def test_edge_search_bands_move_with_lane_center() -> None:
    slopes = np.zeros(100)
    slopes[8:13] = 0.2
    slopes[18:23] = 0.2
    profile = _profile_from_slopes(slopes)

    centered = shoulder_trim_indices(
        profile,
        LaneGeometry(0.0, 10_000.0),
        profile_width_inches=10_000.0,
    )
    shifted = shoulder_trim_indices(
        profile,
        LaneGeometry(1_000.0, 10_000.0),
        profile_width_inches=10_000.0,
    )

    # Center 5000 gives a left search limit of 1500 and selects run 8:12.
    # Center 5500 moves the limit to 2000 and selects inner run 18:22.
    assert centered == (14, 100)
    assert shifted == (24, 100)


def test_original_point_count_recovers_nominal_width_after_edge_crop() -> None:
    x = np.arange(10.0, 91.0)
    slopes = np.zeros(x.size - 1)
    slopes[3:8] = 0.2
    slopes[10:15] = 0.2
    y = np.concatenate(([0.0], np.cumsum(slopes)))
    profile = ReducedProfile(x, y, original_point_count=100)

    # The retained X span is only 80, but the original count and 1-unit sample
    # spacing recover the nominal width 100 and a left search limit of 15.
    assert shoulder_trim_indices(profile, LaneGeometry(0.0, 100.0)) == (9, 81)


def test_remove_shoulders_preserves_profile_diagnostics() -> None:
    profile = ReducedProfile(
        np.arange(11.0),
        np.zeros(11),
        is_noisy=True,
        original_point_count=20,
        valid_fraction=0.75,
        profile_id="sample:7",
        compatibility_notes=("existing note",),
    )
    trimmed = remove_shoulders(profile, LaneGeometry(2.0, 8.0))

    assert trimmed.is_noisy is True
    assert trimmed.original_point_count == 20
    assert trimmed.valid_fraction == 0.75
    assert trimmed.profile_id == "sample:7"
    assert trimmed.compatibility_notes == ("existing note", "clean-room shoulder removal applied")


def test_empty_shoulder_result_is_reported_explicitly() -> None:
    profile = _profile_from_slopes(np.full(100, 0.2))

    with pytest.raises(ValueError, match="left no profile points"):
        remove_shoulders(profile, _whole_profile_lane(profile))


@pytest.mark.parametrize(
    ("point_count", "expected_start", "expected_stop"),
    [
        (1_446, 0, 1_415),
        (1_454, 0, 1_412),
        (1_449, 0, 1_412),
        (1_517, 231, 1_517),
        (1_516, 228, 1_516),
        (1_517, 222, 1_517),
        (1_517, 224, 1_517),
        (1_517, 226, 1_517),
        (1_517, 232, 1_517),
        (1_517, 231, 1_517),
        (1_517, 233, 1_517),
    ],
)
def test_cut_mapping_for_observed_golden_profile_sizes(
    point_count: int,
    expected_start: int,
    expected_stop: int,
) -> None:
    """Keep the 11 independently observed golden slice offsets stable.

    Raw survey values are intentionally not repository fixtures.  These small
    generated profiles exercise the index mapping at the observed sizes and
    boundaries; the private-data black-box validation is documented separately.
    """

    slopes = np.zeros(point_count - 1)
    if expected_start:
        run_end = expected_start - 2
        slopes[run_end - 4 : run_end + 1] = 0.2
    if expected_stop < point_count:
        slopes[expected_stop : expected_stop + 5] = -0.2
    profile = _profile_from_slopes(slopes)

    assert shoulder_trim_indices(profile, _whole_profile_lane(profile)) == (
        expected_start,
        expected_stop,
    )
