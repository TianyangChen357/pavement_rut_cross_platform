# Pavement Rut Cross Platform

Pure Python tooling for reading Pathway `.3dc` pavement profiles, applying the
survey calibration, measuring transverse cross slope and 6-ft rut-bar depth,
and exporting per-file results on Windows or Linux.

The runtime intentionally has no PathView, .NET, `pythonnet`, or Windows-only
dependency. PathView can optionally be used on a licensed Windows workstation
to generate output-level compatibility fixtures; neither its DLLs nor survey
data belong in this repository.

## Status

This repository implements an independently testable processing path:

1. decompress a `.3dc` image and decode its transverse profiles;
2. convert raw heights to calibrated inches;
3. reduce and validate each profile;
4. calculate PathView-compatible transverse cross slope;
5. place a 72-in virtual straightedge in the left and right wheel paths;
6. average profile results for each `.3dc` file;
7. attach interpolated GPS/heading values and export GeoJSON, CSV, and JSON
   metadata;
8. process multiple numeric set directories in parallel.

PathView-compatible behavior is tested at the public input/output boundary.
Exact equivalence cannot be assumed for undocumented filtering and tie-breaking
rules; see the [compatibility notes](docs/compatibility.md) for measured error,
acceptance targets, and remaining assumptions.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/TianyangChen357/pavement_rut_cross_platform.git
cd pavement_rut_cross_platform
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install .
```

Use `python -m pip install -e ".[dev]"` instead when developing or running the
repository test suite.

## Quick start

Inspect one file without writing output:

```bash
pavement-rut inspect-file /data/112/93/11201330004C.3dc \
  --calibration /data/112/3D_Camera.cal
```

Process one set:

```bash
pavement-rut export-set \
  --set-dir /data/112 \
  --out-dir /work/rut/112
```

Run numeric set directories in parallel:

```bash
pavement-rut batch \
  --data-root /data/NC_2018_D05 \
  --out-dir /work/rut/batch_001 \
  --jobs 6
```

By default, each set is expected to contain `3D_Camera.cal`; optional navigation
files are named `gpsdis.<set>` and `heading.<set>`. The `.3dc` files may be in
nested directories. Use `--calibration` to select a different calibration.

Both `export-set` and `batch` resume completed files from crash-tolerant
checkpoints by default. Checkpoints live below `OUT_DIR/.checkpoints`; use
`--no-resume` to start a new journal or `--checkpoint-dir` to place them on a
separate filesystem. A checkpoint is reused only when the source identity,
calibration, and processing options match.

See [long-running exports](docs/operations.md) for checkpoint durability,
immutable-input requirements, parallel-worker guidance, failure semantics, and
exit codes.

Use `--help` on the root command or any subcommand for all options. Inputs,
indexes, generated outputs, proprietary binaries, and local golden results are
excluded by `.gitignore`.

## Outputs

`export-set` writes three files with the same timestamped base name:

- GeoJSON `FeatureCollection`, with one point feature per `.3dc` file when
  navigation is available;
- UTF-8 CSV with `set`, source identity, start frame, navigation, left/right/
  overall rut depth, cross-slope means/counts, and severity; and
- strict JSON metadata containing processing options, diagnostics, counts,
  checkpoint provenance, and output paths.

Rut, profile, and straightedge dimensions are inches.
`cross_slope_average_percent` and `cross_slope_average_angle_degrees` are the
separate arithmetic means of the
finite PathView-compatible per-profile percent and angle results in that
`.3dc` file. `cross_slope_count` is the number of profiles contributing to
both means, and `cross_slope_error_count` records cross-slope-specific failures
after profile reduction. Reduction failures and skipped noisy profiles remain
in their existing row counters. The angle mean is not recomputed from the mean
percent.

Latitude and longitude are decimal degrees, heading follows the source
navigation file, and severity is `0` to `3`; `-1` means a valid two-sided rut
average was unavailable. Missing navigation and non-finite measurements are
serialized as JSON `null`, never non-standard `NaN`; the corresponding CSV
cell is blank.

`batch` additionally writes timestamped CSV and strict JSON summaries for all
selected numeric set directories. Any partial or failed set produces a nonzero
CLI exit status even though its diagnostics and completed records are retained.

These outputs are derived survey data. CSV/GeoJSON include source file identity
and navigation values, while metadata, checkpoint manifests, and batch
summaries also record resolved local paths for reproducibility. Keep them in an
approved output location and review them before sharing. The repository ignore
rules cover the standard output names, but they are not a substitute for your
organization's data-handling policy.

## Python API

The supported library entry points are available from `pavement_rut.io` and
`pavement_rut.domain`. Application-level callers may use
`pavement_rut.app.export_set.export_set` and
`pavement_rut.app.batch.run_batch`; their frozen configuration dataclasses make
all processing inputs explicit. The command-line interface is recommended for
long cluster runs because it also manages indexes, checkpoints, progress, and
exit status.

## Validation

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
python -m build
```

The oracle exporter is a Windows-only development tool, not a runtime
dependency:

```powershell
python tools\export_pathview_golden.py --help
```

The comparator is pure Python and also runs on Linux:

```bash
python tools\compare_pathview_golden.py --help
```

## Output interpretation

Rut depth is reported in inches. The default straightedge is 72 in. Left and
right wheel-path means are calculated independently; the overall value is the
mean of those two finite values. Severity classes retain the existing workflow
thresholds: less than 0.25 in, 0.25–0.5 in, 0.5–1.0 in, and at least 1.0 in.

This is analytical software, not a certification of the underlying survey or
an engineering acceptance decision. Validate results against your governing
specification and approved reference software before production use.

## License status

[NOTICE](NOTICE.md) documents the independent-implementation and vendor
boundary; it is not an open-source license. No open-source license is declared
in this initial repository. Choose and add an explicit license before making
the repository public or authorizing redistribution.
