# PathView compatibility and Windows oracle

## Scope and clean-room boundary

The cross-platform runtime is an independent implementation of the narrow
processing chain needed by this project:

1. decode a `.3dc` transverse profile;
2. apply the matching camera calibration;
3. reduce/filter the calibrated profile;
4. calculate AASHTO-style cross slope;
5. calculate left/right 6-ft rut-bar geometry and depth; and
6. aggregate those results per file.

It is not a rewrite of the PathView application or of every Pathway assembly.
No PathView DLL, vendor source, decompiled code, calibration file, or survey
file is included in this repository.

During development, a legitimately licensed PathView installation on Windows
may be used as an **oracle**: the repository calls its public .NET API, records
inputs and outputs, and compares the independent implementation against those
results. The oracle is optional development tooling. Production Linux code
must not import `pythonnet`, load a DLL, or invoke PathView.

Before using the oracle, confirm that the applicable software and data
agreements permit local interoperability testing. A real-data oracle export is
derived survey data and must not be committed or published without data-owner
approval.

## Oracle tool

The Windows-only tool is `tools/export_pathview_golden.py`. It requires:

- Windows;
- Python with `pythonnet` and NumPy;
- .NET 8 Desktop Runtime, as required by the installed PathView release; and
- a local, licensed PathView installation.

Supply the assembly directory with `--pathview-dir` or the `PATHVIEW_DIR`
environment variable. The assemblies are loaded in place and are never copied;
the repository contains no workstation-specific default path.

### Export selected real profiles

Always choose the source, calibration directory, profile indices, and output
explicitly. Store private results outside the repository:

```powershell
python tools\export_pathview_golden.py real `
  --input-3dc D:\data\112\93\11201330004C.3dc `
  --calibration-dir D:\data\112 `
  --profile-indices 0,450,899 `
  --output-json D:\private-validation\set112_sample.metadata.json
```

The tool writes:

- a JSON manifest containing assembly hashes, all calculation parameters,
  calibration dimensions, noisy/status flags, cross slope, and complete
  left/right/center rutting geometry; and
- a compressed NPZ containing flat typed arrays for raw `UInt16` heights,
  calibrated heights, intensity, and reduced `(x, y)` profiles.

The NPZ has no object arrays and can be opened safely with:

```python
import numpy as np

arrays = np.load("set112_sample.metadata.arrays.npz", allow_pickle=False)
start, stop = arrays["raw_offsets"][0:2]
first_raw_profile = arrays["raw_height_u16"][start:stop]
```

### Compare a private real-data golden

The comparison tool is pure Python and runs on either Windows or Linux; it
does not load PathView or any DLL. Keep the JSON and NPZ together, then run:

```bash
python tools/compare_pathview_golden.py \
  /private-validation/set112_sample.metadata.json \
  --json-output /private-validation/set112_comparison.json
```

It validates the untrusted manifest and array schema, verifies source hashes,
replays reduction, cross slope, and rut-bar calculations, and reports every
tolerance separately. Exit status `0` means all requested checks passed, `1`
means at least one numerical check failed, and `2` means the fixture or command
was invalid. The default tolerances are the initial acceptance values listed
below; loosening them should be an explicit, reviewed decision rather than a
way to conceal a known compatibility residual.

Absolute source and install paths are redacted by default. Use
`--include-source-paths` only for private provenance records. The source file
name, size, and SHA-256 remain in the manifest so an authorized validation run
can be reproduced.

Relevant calculation controls are also explicit:

```text
--is-double-laser
--roll-degrees
--max-stddev-inches-high-noise
--dark-band-columns
--lane-left-inches / --lane-right-inches
--default-edge-distance-inches
--rut-bar-meters
--calculate-center-rutting
```

Do not compare two implementations unless these values and the calibration
are identical.

### Generate non-survey synthetic golden data

This command uses four deterministic shapes (flat, planar 1% rise, two
Gaussian ruts, and a central V) and writes JSON only:

```powershell
python tools\export_pathview_golden.py synthetic `
  --calculate-center-rutting `
  --output-json D:\private-validation\pathview-synthetic.json
```

The fixture contains the full synthetic input and reduced profile arrays, so
it can be replayed without PathView. It contains no survey or calibration
data. It still records the exact assembly hashes because behavior can change
between vendor versions.

## Public API boundary recorded by the oracle

The tool observes only public constructors, properties, and method results.
The important boundary is:

| Stage | Public PathView API | Recorded result |
| --- | --- | --- |
| Source | `ImageInfo`, `Surface3DImage.GetProfiles()` | selected profile index |
| Raw profile | `GetRawHeightValue`, `HeightInches`, `Color` | raw height, calibrated height, intensity |
| Reduction | `DataReducedProfile.GetFrom3DC` | reduced `(x, y)`, count, noisy flag |
| Lane | `LanePositions` | explicit left/right/default edge inputs |
| Cross slope | `AashtoCrossSlope` | percent and angle |
| Rutting | `RutBarRutting` | L/R/C depth, reference, contact, measurement, and rut points |

Synthetic mode uses the public
`DataReducedProfile(IEnumerable<ICoordinate>, ...)` constructor instead of
`GetFrom3DC`, allowing algorithm behavior to be tested without a vendor data
format or survey file.

The observed public signatures are:

```text
DataReducedProfile.GetFrom3DC(
  ISurface3DProfile profile,
  ISurface3DCalibration calibration,
  bool isDoubleLaser,
  double rollDegrees,
  HashSet<int> darkBandColumns,
  double maxStdDevInchesHighNoise
) -> Profile: IReadOnlyList<ICoordinate>, IsProfileNoisy: bool

AashtoCrossSlope(
  DataReducedProfile dataReduced,
  LanePositions lanePositions
) -> Percent: double, AngleDegrees: double

RutBarRutting(
  DataReducedProfile dataReduced,
  LanePositions lanePositions,
  ICrossSlope crossSlope,
  Length rutBarLength,
  bool calculateCenterRutting
) -> Left/Right/Center: IRuttingGeometry or null
```

Each `IRuttingGeometry` exposes rutting depth plus left/right reference points,
left/right contact points, a measurement point, and a rut point. The oracle
records all six coordinates instead of treating the depth alone as sufficient.

## Measured behavior of the current workstation oracle

These are observations, not a claim about all PathView releases. The manifest
is the authoritative record for each run.

Current assembly versions:

| Assembly | Assembly version | SHA-256 |
| --- | --- | --- |
| `Pathway.Core.dll` | 3.2.3.0 | `5babb73c3b84560b08340bdada3450a5a37b7439071aa43a287a0449ba03abca` |
| `Pathway.Data.dll` | 7.2.25.0 | `4c69def6c7172298f5c60581de6f6f9486e6ff541ddd18b3dfec8b5bf12f7454` |
| `Pathway.Processing.dll` | 2.0.21.0 | `ee19dcb394cbcbbb62c2807f0ffe60df8512bfb2239589f3a9228521e8e3905c` |

Public `RutBarRutting` constants exposed by this build are:

- default rut-bar length: 72 in (6 ft / 1.8288 m);
- rut-path width: 44.291340 in;
- rut-path half-width: 22.145670 in; and
- wheel-path center offset from lane centerline: 34.448819 in.

With a 0.5-in synthetic input grid over 0 to 162 in, the public
`DataReducedProfile` constructor returned 321 points spanning 0.75 to 160.75 in
from 325 input points. A surface rising as `y = 0.01*x` returned approximately
`-1.003125%` cross slope, establishing the sign convention for this specific
coordinate setup. A flat profile returned zero rutting. The two-Gaussian
profile returned approximately 0.477997 in left and 0.764795 in right rutting.
The central V returned center geometry only when center calculation was
enabled.

The reduction result depends on sampling interval. For the same linear profile,
observed input/reduced counts and bounds were:

| Input step | Input count | Reduced count | Reduced X bounds |
| --- | ---: | ---: | --- |
| 0.10 in | 1621 | 1600 | 0.95 to 160.90 in |
| 0.25 in | 649 | 641 | 0.875 to 160.875 in |
| 0.50 in | 325 | 321 | 0.75 to 160.75 in |
| 1.00 in | 163 | 161 | 0.50 to 160.50 in |

This is evidence that reduction is a substantive stage; a replacement must
not treat calibrated samples as already reduced.

### Exact reduction behavior observed for this build

Synthetic `Surface3DProfile` rows and three real profiles from the private Set
112 validation file made the main reduction stages reproducible without
inspecting implementation code. On the 1536-column calibration used here:

1. column `i` becomes `(x, y) = (i * PixelWidth, HeightInches[i])`;
2. the full calibrated Y sequence is summarized by its global mean and standard
   deviation;
3. samples farther than 3.5 standard deviations from the mean, plus indices in
   `darkBandColumns`, are removed;
4. each output coordinate is the arithmetic mean of 19 consecutive retained
   `(x, y)` coordinates; and
5. the final otherwise-valid 19-point window is not emitted.

For uniformly spaced retained samples, this is a centered 19-sample mean: the
output coordinate is centered on the tenth sample in its window.

For a retained sequence of length `M`, this build therefore emits `M - 19`
points. In NumPy notation, both axes match:

```python
kernel = np.ones(19, dtype=np.float64) / 19.0
reduced_x = np.convolve(retained_x, kernel, mode="valid")[:-1]
reduced_y = np.convolve(retained_y, kernel, mode="valid")[:-1]
```

At `PixelWidth = 0.10546875 in`, the 19-sample averaging window spans about 2
inches. Removing raw samples before the mean makes reduced X nonuniform. This
explains the 1-to-3-pixel X increments seen in real profiles and why reduced Y
is not equal to a single calibrated raw height.

The 3.5-sigma keep mask was tested with both sample and population standard
deviation; both produced the same selected samples in the current golden
corpus. The independent implementation uses sample standard deviation and
retains a boundary fixture to detect any difference at an exact threshold.

The noise flag is a separate calculation performed on the unfiltered
calibrated Y sequence:

1. divide the values, in order, into non-overlapping batches of 30;
2. ignore a final batch containing fewer than 30 values;
3. calculate sample standard deviation (`ddof=1`) for each batch; and
4. compare the median batch standard deviation with
   `maxStdDevInchesHighNoise`.

The boundary values returned by the public API matched this statistic to
floating-point precision for flat, ramp, sinusoidal, and localized Gaussian
profiles.

`rollDegrees` is applied after reduction as a rigid clockwise rotation about
the first reduced point. For angle `theta`, the observed transform is:

```text
dx = x - x0; dy = y - y0
x' = x0 + dx*cos(theta) + dy*sin(theta)
y' = y0 - dx*sin(theta) + dy*cos(theta)
```

The maximum coordinate error against the public API was below `3e-14 in` for
rolls of -1, +1, and +2 degrees. With `isDoubleLaser=True`, this calibration
also removed 18 center points after reduction (raw-grid X indices 759 through
776); release- and calibration-specific regression coverage is still needed
before generalizing that seam rule.

For three real Set 112 profiles (indices 0, 450, and 899), replaying the
observed keep mask and 19-point mean reproduced every reduced coordinate. The
maximum absolute X error was `5.7e-14 in` and maximum absolute Y error was
`1.3e-15 in`; their PathView reduced counts were 1446, 1454, and 1449.

The public `Surface3DProfile(byte[], startIndex, calibration)` constructor also
accepted a 4616-byte synthetic row: 8 header bytes, 1536 intensity bytes in
reverse column order, and 1536 little-endian `UInt16` heights in reverse column
order. This observation is useful for constructing oracle probes, but it does
not by itself define the surrounding compressed `.3dc` container.

`GetProfileWithShoulderRemoved` is not a simple clip to `LanePositions`. An
earlier probe incorrectly reported lane-invariant results because it reused a
single `DataReducedProfile`; that object cached its first shoulder result. With
a fresh object for every lane, `GetProfileWithShoulderRemoved` and the public
`Filters.RemoveShoulderHighOrLow` method returned identical point sequences.
A flat synthetic profile has no geometric shoulder and is therefore clipped
only to points strictly inside the requested lane. On three Set 112 profiles,
the full-lane result changed as follows:

| Profile | Reduced count and X bounds | Shoulder-removed count and X bounds |
| ---: | --- | --- |
| 0 | 1446, 0.949219 to 154.395148 in | 1415, 0.949219 to 150.082031 in |
| 450 | 1454, 0.949219 to 154.195312 in | 1412, 0.949219 to 149.765625 in |
| 899 | 1449, 0.949219 to 153.723479 in | 1412, 0.949219 to 149.765625 in |

For the currently tested build, public-API probes reproduced shoulder removal
with the following clean-room rule. For adjacent points in the complete
reduced sequence, calculate `abs(delta_y / delta_x)`. A contiguous run of
slopes strictly greater than `0.055` qualifies when it contains at least five
segments and at least one slope greater than or equal to `0.17`. Center the
edge search on the resolved lane center, with limits 35% of the nominal
profile width to either side. For a centered full-width lane, these are the
inner edges of the outer 15% bands and are equivalent to the observed
`band = round(point_count * 0.15)` rule on a full synthetic grid.

For the left side, choose the eligible run with the greatest start and retain
from point `run_end + 2`. For the right side, choose the eligible run with the
smallest start and retain through point `run_start - 1`. Intersect this
geometric result with the strict-open lane clip `left < x < right`. Shoulder
detection therefore precedes lane clipping; narrow lanes do not redefine the
edge-search bands. The nominal width is recovered from original sample count
and sample spacing after reduction, or can be supplied explicitly for an
externally cropped profile.

Synthetic probes distinguished `0.169999` from the inclusive `0.17` trigger,
`0.055` from the strict `0.055001` continuation threshold, four from five
steep segments, both slope signs, and innermost-run selection. The independent
implementation matched all 11 primary real-profile slices exactly:

| Set | Profiles | Exact retained boundary X values (in) |
| --- | --- | --- |
| 112 | 0, 450, 899 | right: 150.082031, 149.765625, 149.765625 |
| 113 | 0, 99, 423, 450 | left: 25.312500, 25.101562, 24.363281, 24.574219 |
| 113 | 729, 796, 875, 899 | left: 24.785156, 25.417969, 25.312500, 25.523438 |

Fresh-object replay also matched all 27 Set 112 and all 72 Set 113
profile/lane combinations, including resolved lane centers at 72, 80, 81, and
90 inches and narrow-lane strict clipping. Private survey arrays remain outside
the repository; committed tests use generated profiles.

### Cross-slope half-lane fit

After shoulder removal and strict-open lane clipping, the tested public
`AashtoCrossSlope` result is determined by two point-count-weighted means. A
point exactly on the lane center belongs to both halves:

```text
left_mean = mean(y where x <= lane_center)
right_mean = mean(y where x >= lane_center)
run = (last_retained_x - first_retained_x) / 2
fall_per_run = (left_mean - right_mean) / run
Percent = 100 * fall_per_run
AngleDegrees = degrees(atan(fall_per_run))
```

There is no OLS fit, endpoint chord, x-spacing weight, or additional leveling
step. Positive percent means the mean elevation falls toward increasing x.
The denominator uses half the actual retained span, not half the requested
lane width or the distance between half means.

Across 99 private real profile/lane combinations from Set 112 and Set 113, the
independent result had a maximum absolute error of `7.3e-15` percentage points
for percent and `4.2e-15 deg` for angle. Another 120 synthetic public-API cases
had maximum errors of `8.9e-16` percentage points and `5.6e-16 deg`. These are
floating-point roundoff for the tested assembly build, not a guarantee for
every PathView release.

### Rut-point footprint

Rut-bar output is not based on the single reduced Y value nearest the reported
rut-point X. On real profile 0, the reported rut-point Y closely matched the
arithmetic mean over an approximately 4-in neighborhood around that X:

| Geometry | PathView rut-point Y | Observed approximately 4-in mean |
| --- | ---: | ---: |
| Left | 1.82412065 in | 1.82392170 in |
| Right | 1.72009825 in | 1.71956760 in |

The single nearest reduced Y differed by about 0.013 in (left) and 0.026 in
(right). Candidate measurement centers also required their entire 4-in window
to remain inside the public 44.291340-in wheel-path zone. The exact endpoint
inclusion and interpolation behavior remain golden-test targets.

The independent implementation uses a continuous piecewise-linear 4-in moving
average and limits the complete footprint to the wheel-path zone and the two
bar contact points. Against the three private Set 112 profiles above, all 12
contact X/Y coordinates matched the oracle. Rut-depth residuals (independent
minus oracle, in inches) were:

| Profile | Left | Right |
| ---: | ---: | ---: |
| 0 | +0.009864 | +0.003057 |
| 450 | -0.000500 | -0.001936 |
| 899 | +0.000463 | +0.001416 |

Five of six paths are within 0.0031 in. Profile 0 left selects the legal
wheel-zone boundary in the continuous model, while the oracle selects an
interior center. This remains an explicit compatibility gap; the
implementation does not add a fixture-specific inward margin or otherwise
overfit these three profiles.

Changing the requested rut-bar length changed the Gaussian-rut result until
the bar was long enough to span the controlling contact points. Therefore the
bar length belongs in every fixture's identity; a fixture generated with a
different length is not comparable.

## Acceptance criteria

Use several authorized sets covering smooth pavement, severe rutting, noisy or
incomplete profiles, different camera calibrations, and (if applicable) both
single- and double-laser acquisition. Split fixtures into development and
hold-out sets.

Recommended initial tolerances are:

| Quantity | Comparison |
| --- | --- |
| Raw height and intensity | exact value and count |
| Calibrated height | maximum absolute error <= `1e-9 in` |
| Reduced profile status/noisy flag/count | exact |
| Reduced `(x, y)` | maximum absolute error <= `1e-6 in` |
| Cross-slope percent | absolute error <= `1e-4` percentage points |
| Cross-slope angle | absolute error <= `1e-4 deg` |
| Per-profile rut depth | absolute error <= `0.001 in` |
| Non-degenerate contact/rut points | absolute X/Y error <= `0.01 in` |
| Per-file mean left/right/combined rut | absolute error <= `0.001 in` |

Raw decoding and calibration should meet their strict limits before reduction
or rutting differences are investigated. Flat or tied profiles can have many
equivalent contact locations; for those degeneracies, assert depth and
presence/absence of geometry rather than an arbitrary coordinate identity.

No production-compatibility claim should be made from synthetic cases alone.
The implementation should also report pass counts, maximum error, RMSE, and
the worst profile identifiers on the private hold-out corpus.

## Remaining unknown semantics

Public I/O observations do not yet establish all internal choices. The main
unknowns to resolve through independent implementation and golden regression
are:

- invalid raw-height sentinels and interpolation rules;
- whether the observed double-laser seam rule generalizes to other calibrations;
- exact equality behavior at the 3.5-sigma boundary;
- whether the recovered shoulder thresholds and tie-breaking generalize to
  other PathView releases and camera configurations;
- exact 4-in rut-footprint endpoints, contact interpolation, and tolerances;
- wheel-path boundaries and tie-breaking at equal maxima; and
- behavior when left, right, or center geometry cannot be constructed.

These should be described as compatibility findings, not copied from or
attributed to vendor implementation details. If a published specification or
vendor-authorized interchange format becomes available, it takes precedence
over black-box inference.
