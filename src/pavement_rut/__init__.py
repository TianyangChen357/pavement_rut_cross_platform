"""Cross-platform pavement rut-depth processing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pavement-rut-cross-platform")
except PackageNotFoundError:  # pragma: no cover - source-tree import
    __version__ = "0.1.0"

__all__ = ["__version__"]
