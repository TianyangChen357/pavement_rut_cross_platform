# Long-running exports

The command-line workflow is designed so a failed or interrupted multi-terabyte
run can continue without recomputing every completed `.3dc` file.

## Checkpoints and resume

`export-set` and `batch` resume by default. After each completed source file,
the parent process appends one strict-JSON record to a per-set journal. The
default `--checkpoint-every 1` flushes and fsyncs every record. A larger value
reduces filesystem synchronization overhead, but a power loss may require the
last few completed files to be processed again. It cannot make a numeric result
partially reusable.

A reusable result is scoped by all of the following:

- checkpoint and processing schema versions;
- the resolved source-set directory;
- the calibration file SHA-256;
- every processing option;
- the source file's relative path, frame interval, size, mtime, and ctime; and
- a SHA-256 probe over up to 4 KiB at the beginning, middle, and end of the
  source file.

The sparse content probe avoids rereading an entire multi-terabyte survey just
to start a resumed run. It is an operational stale-cache guard, not a full-file
integrity checksum. Treat source sets as immutable while processing. The
program verifies each file identity again after its worker returns and aborts
without checkpointing that result if the file changed during processing.

The portable index is also an immutable-set snapshot. If `.3dc` files were
added, removed, renamed, or recopied after the index was built, use
`export-set --rebuild-index` or `batch --rebuild-indexes`. Missing indexed files
are reported as failures rather than silently omitted.

Only fully successful file results are reused. Partial and failed results stay
in the append-only journal for audit purposes and are retried on the next run.
If a crash truncates the final JSONL append, resume removes only that incomplete
tail. Corruption in any complete line fails closed instead of silently dropping
work.

Use `--no-resume` to start a fresh journal for the current processing
fingerprint. Use `--checkpoint-dir` when checkpoints should live on a different
durable volume. A per-set `RUNNING.lock` prevents two writers from using the
same checkpoint area. Stale locks are never deleted automatically; first
confirm that no process is active, then remove the exact lock path shown in the
error.

## Parallelism

For one set, `export-set --jobs N` runs at most `N` `.3dc` workers and keeps a
small bounded submission queue. Results and checkpoints are restored to index
order even when workers finish out of order.

For multiple sets, `batch --jobs N` runs at most `N` set processes. Each set
uses one file worker, avoiding nested process pools and accidental CPU or memory
oversubscription. Normal file failures are isolated within their set, and a set
failure does not prevent the batch summary from recording other set outcomes.

Choose a worker count based on memory and storage throughput, not CPU count
alone. Each worker decodes a complete `.3dc` image and holds calibrated profile
arrays in memory.

## Outputs and status

GeoJSON, metadata JSON, CSV, checkpoint manifests, and fresh journals are
written through a temporary file in the destination directory, fsynced, and
atomically replaced. JSON files reject `NaN` and `Infinity`; unavailable numeric
values are written as `null`. Progress messages go to stderr, leaving stdout as
one machine-readable JSON document containing the output paths.

Each CSV row and GeoJSON feature includes
`cross_slope_average_percent`, `cross_slope_average_angle_degrees`,
`cross_slope_count`, and `cross_slope_error_count`. The averages are calculated
separately over finite per-profile results for that `.3dc` file; the error count
covers cross-slope-specific failures after reduction. If no profile has a valid
cross slope, both JSON averages are `null` (blank in CSV), both rut results and
their counts are still retained, and the file status is `partial` when a valid
rut result remains.

Command exit codes are:

- `0`: every requested file/set completed successfully;
- `1`: processing completed and outputs were written, but at least one
  file/set was partial or failed;
- `2`: invalid configuration, missing input, checkpoint conflict/corruption,
  or another pre-output operational error.

An unexpected programming defect is not converted into a plausible partial
dataset. It aborts the affected command, leaves prior checkpoints reusable, and
surfaces a traceback when called through Python.
