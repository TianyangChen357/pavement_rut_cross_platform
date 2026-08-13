from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
PROHIBITED_SUFFIXES = {
    ".3dc",
    ".cal",
    ".csv",
    ".dll",
    ".exe",
    ".geojson",
    ".npy",
    ".npz",
    ".pdb",
    ".psi",
}


def _release_files() -> list[Path]:
    return [
        path
        for path in REPOSITORY_ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.relative_to(REPOSITORY_ROOT).parts)
    ]


def test_repository_contains_no_survey_data_or_native_binaries() -> None:
    prohibited = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _release_files()
        if path.suffix.lower() in PROHIBITED_SUFFIXES
    ]
    assert prohibited == []


def test_repository_contains_no_personal_windows_home_path() -> None:
    leaked: list[str] = []
    for path in _release_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        normalized = text.replace("/", "\\").lower()
        if ":\\users\\" in normalized:
            leaked.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert leaked == []
