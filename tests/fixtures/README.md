# Compatibility fixtures
Only deterministic synthetic fixtures may be committed here.

Do **not** add:

- `.3dc`, `.psi`, calibration, GPS, heading, or other source survey files;
- real-profile JSON/NPZ oracle exports;
- PathView/Pathway DLLs, runtime files, or other vendor binaries; or
- absolute local input/install paths.

Generate a private real-data golden outside this repository with
`tools/export_pathview_golden.py real`. Generate a reviewable synthetic fixture
with `tools/export_pathview_golden.py synthetic`. See
`docs/compatibility.md` for commands, schema, provenance, and acceptance
tolerances.

Every golden result must identify the exact oracle assembly versions and
SHA-256 hashes. A fixture produced by a different PathView build is a separate
compatibility target and must not silently replace the existing baseline.
