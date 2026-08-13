"""Crash-tolerant per-file checkpoints for long-running set exports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pavement_rut.index import ImageRecord

CHECKPOINT_SCHEMA_VERSION = 2
# Bump this whenever a change can alter a FileRutResult for identical inputs.
PROCESSING_SCHEMA_VERSION = 3

_SOURCE_PROBE_BYTES = 4096


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json_loads(text: str, *, source: Path, line_number: int | None = None) -> Any:
    location = str(source) if line_number is None else f"{source} line {line_number}"
    try:
        return json.loads(text, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Corrupt checkpoint JSON at {location}: {exc}") from exc


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint fingerprint input is not strict JSON: {exc}") from exc
    return text.encode("ascii")


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it all into memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_probe_sha256(path: Path, *, size_bytes: int) -> str:
    """Hash small, deterministic regions to catch metadata-preserving replacements.

    Full-file hashing would force a second read of every source file on every
    resume, which is prohibitive for multi-terabyte surveys.  Size, mtime, and
    ctime remain part of :class:`RecordIdentity`; this probe additionally reads
    up to 4 KiB at the beginning, middle, and end of each file.
    """

    digest = hashlib.sha256()
    digest.update(f"sample-v1:{size_bytes}:".encode("ascii"))
    sample_size = min(_SOURCE_PROBE_BYTES, size_bytes)
    offsets = sorted(
        {
            0,
            max(0, (size_bytes - sample_size) // 2),
            max(0, size_bytes - sample_size),
        }
    )
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            sample = stream.read(sample_size)
            if len(sample) != sample_size:
                raise OSError(
                    f"Source file changed while fingerprinting: {path} "
                    f"(expected {sample_size} bytes at offset {offset}, read {len(sample)})"
                )
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(len(sample).to_bytes(8, "little", signed=False))
            digest.update(sample)
    return digest.hexdigest()


def _normalized_source_root(path: Path) -> str:
    """Return a stable absolute source-root string for the run fingerprint."""

    resolved = str(path.expanduser().resolve())
    return os.path.normcase(resolved) if os.name == "nt" else resolved


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int]:
    return int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after a durable create/replace/unlink."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Durably replace ``path`` without sharing a fixed temporary filename."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def build_checkpoint_manifest(
    *,
    set_label: str,
    set_dir: Path,
    calibration_path: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic manifest whose fingerprint scopes reusable results."""

    resolved_set_dir = set_dir.expanduser().resolve()
    resolved_calibration_path = calibration_path.expanduser().resolve()
    calibration_stat_before = resolved_calibration_path.stat()
    calibration_sha256 = sha256_file(resolved_calibration_path)
    calibration_stat_after = resolved_calibration_path.stat()
    if _stat_signature(calibration_stat_before) != _stat_signature(calibration_stat_after):
        raise OSError(f"Calibration changed while its checkpoint fingerprint was computed: {resolved_calibration_path}")
    fingerprint_input = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "set": set_label,
        "source_set_dir": _normalized_source_root(resolved_set_dir),
        "source_probe": "sha256(first,middle,last;4096-bytes-each)-v1",
        "calibration": {
            "sha256": calibration_sha256,
            "size_bytes": calibration_stat_after.st_size,
        },
        "options": options,
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(fingerprint_input)).hexdigest()
    return {**fingerprint_input, "fingerprint": fingerprint}


@dataclass(frozen=True, slots=True)
class RecordIdentity:
    """Cheap identity for one immutable source file and its index interval."""

    relative_path: str
    start_frame: float
    end_frame: float
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    source_probe_sha256: str

    def __post_init__(self) -> None:
        posix_relative = PurePosixPath(self.relative_path)
        windows_relative = PureWindowsPath(self.relative_path)
        if (
            not self.relative_path
            or "\x00" in self.relative_path
            or posix_relative.is_absolute()
            or windows_relative.is_absolute()
            or windows_relative.drive
            or ".." in posix_relative.parts
            or ".." in windows_relative.parts
        ):
            raise ValueError(f"Unsafe checkpoint relative path: {self.relative_path!r}")
        if not math.isfinite(self.start_frame) or not math.isfinite(self.end_frame):
            raise ValueError("Checkpoint frame bounds must be finite")
        if self.end_frame <= self.start_frame:
            raise ValueError("Checkpoint end_frame must be greater than start_frame")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("Checkpoint size_bytes must be a non-negative integer")
        if isinstance(self.mtime_ns, bool) or not isinstance(self.mtime_ns, int) or self.mtime_ns < 0:
            raise ValueError("Checkpoint mtime_ns must be a non-negative integer")
        if isinstance(self.ctime_ns, bool) or not isinstance(self.ctime_ns, int) or self.ctime_ns < 0:
            raise ValueError("Checkpoint ctime_ns must be a non-negative integer")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_probe_sha256) is None:
            raise ValueError("Checkpoint source_probe_sha256 must be a lowercase SHA-256 digest")

    @property
    def key(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(asdict(self))).hexdigest()

    @classmethod
    def from_record(cls, set_dir: Path, record: ImageRecord) -> RecordIdentity:
        path = record.resolve(set_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Indexed .3dc file does not exist: {path}")
        stat_before = path.stat()
        source_probe = _source_probe_sha256(path, size_bytes=int(stat_before.st_size))
        stat_after = path.stat()
        if _stat_signature(stat_before) != _stat_signature(stat_after):
            raise OSError(f"Source file changed while its checkpoint identity was computed: {path}")
        return cls(
            relative_path=record.relative_path,
            start_frame=float(record.start_frame),
            end_frame=float(record.end_frame),
            size_bytes=int(stat_after.st_size),
            mtime_ns=int(stat_after.st_mtime_ns),
            ctime_ns=int(stat_after.st_ctime_ns),
            source_probe_sha256=source_probe,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> RecordIdentity:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint record identity must be an object")
        expected = {
            "relative_path",
            "start_frame",
            "end_frame",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
            "source_probe_sha256",
        }
        if set(payload) != expected:
            raise ValueError("Checkpoint record identity fields do not match schema")
        if not isinstance(payload["relative_path"], str):
            raise ValueError("Checkpoint relative_path must be a string")
        for name in ("start_frame", "end_frame"):
            if isinstance(payload[name], bool) or not isinstance(payload[name], (int, float)):
                raise ValueError(f"Checkpoint {name} must be numeric")
        for name in ("size_bytes", "mtime_ns", "ctime_ns"):
            if isinstance(payload[name], bool) or not isinstance(payload[name], int):
                raise ValueError(f"Checkpoint {name} must be an integer")
        try:
            return cls(
                relative_path=payload["relative_path"],
                start_frame=float(payload["start_frame"]),
                end_frame=float(payload["end_frame"]),
                size_bytes=payload["size_bytes"],
                mtime_ns=payload["mtime_ns"],
                ctime_ns=payload["ctime_ns"],
                source_probe_sha256=payload["source_probe_sha256"],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid checkpoint record identity: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    identity: RecordIdentity
    result: dict[str, Any]


class CheckpointStore:
    """Single-writer JSONL journal guarded by an atomic per-set lock."""

    def __init__(
        self,
        *,
        root: Path,
        set_label: str,
        manifest: dict[str, Any],
        resume: bool,
        fsync_every: int,
    ) -> None:
        if fsync_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        component = re.sub(r"[^A-Za-z0-9._-]+", "_", set_label).strip("_")
        if not component:
            raise ValueError("set label cannot produce an empty checkpoint directory name")
        fingerprint = manifest.get("fingerprint")
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise ValueError("checkpoint manifest has an invalid fingerprint")

        self.root = root.expanduser().resolve()
        self.set_checkpoint_dir = self.root / f"set_{component}"
        self.run_dir = self.set_checkpoint_dir / fingerprint
        self.manifest_path = self.run_dir / "manifest.json"
        self.journal_path = self.run_dir / "results.jsonl"
        self.lock_path = self.set_checkpoint_dir / "RUNNING.lock"
        self.manifest = manifest
        self.resume = bool(resume)
        self.fsync_every = int(fsync_every)
        self.entries: dict[str, CheckpointEntry] = {}
        self._token = uuid.uuid4().hex
        self._stream: Any = None
        self._since_fsync = 0
        self._lock_owned = False

    def __enter__(self) -> CheckpointStore:
        self.set_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_manifest_and_journal()
            self._stream = self.journal_path.open("a", encoding="utf-8", newline="\n")
            _fsync_directory(self.run_dir)
            return self
        except BaseException:
            self._release_lock()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        close_error: BaseException | None = None
        try:
            if self._stream is not None:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
                self._stream = None
        except BaseException as caught:
            close_error = caught
        finally:
            self._release_lock()
        if close_error is not None and exc is None:
            raise close_error

    def _acquire_lock(self) -> None:
        lock_payload = {
            "schema_version": 1,
            "token": self._token,
            "host": socket.gethostname(),
            "process_id": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.lock_path, flags, 0o644)
        except FileExistsError as exc:
            try:
                owner = self.lock_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                owner = "unreadable"
            raise RuntimeError(
                f"Checkpoint lock already exists: {self.lock_path}. Owner: {owner}. "
                "Confirm that no export is running, then remove this stale lock manually."
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(lock_payload, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.set_checkpoint_dir)
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            _fsync_directory(self.set_checkpoint_dir)
            raise
        self._lock_owned = True

    def _release_lock(self) -> None:
        if not self._lock_owned:
            return
        try:
            payload = _strict_json_loads(
                self.lock_path.read_text(encoding="utf-8"),
                source=self.lock_path,
            )
            if isinstance(payload, dict) and payload.get("token") == self._token:
                self.lock_path.unlink(missing_ok=True)
                _fsync_directory(self.set_checkpoint_dir)
        except (OSError, ValueError):
            # Never remove a lock whose ownership can no longer be proven.
            pass
        finally:
            self._lock_owned = False

    def _prepare_manifest_and_journal(self) -> None:
        if self.resume and self.manifest_path.exists():
            existing = _strict_json_loads(
                self.manifest_path.read_text(encoding="utf-8"),
                source=self.manifest_path,
            )
            if existing != self.manifest:
                raise ValueError(
                    f"Checkpoint manifest does not match the requested processing run: {self.manifest_path}"
                )
        elif self.resume and self.journal_path.exists() and self.journal_path.stat().st_size:
            raise ValueError(f"Checkpoint journal exists without its manifest: {self.journal_path}")
        else:
            self._write_manifest_atomic()

        if not self.resume:
            _write_bytes_atomic(self.journal_path, b"")
            self.entries = {}
            return
        self.entries = self._load_and_repair_journal()

    def _write_manifest_atomic(self) -> None:
        _write_bytes_atomic(
            self.manifest_path,
            json.dumps(
                self.manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8"),
        )

    def _load_and_repair_journal(self) -> dict[str, CheckpointEntry]:
        if not self.journal_path.exists():
            return {}
        data = self.journal_path.read_bytes()
        final_newline = data.rfind(b"\n")
        complete_size = final_newline + 1
        if complete_size != len(data):
            # A crash may cut off only the final append. Preserve every complete line.
            with self.journal_path.open("r+b") as stream:
                stream.truncate(complete_size)
                stream.flush()
                os.fsync(stream.fileno())
            data = data[:complete_size]

        entries: dict[str, CheckpointEntry] = {}
        for line_number, raw_line in enumerate(data.splitlines(), start=1):
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Corrupt checkpoint UTF-8 at {self.journal_path} line {line_number}: {exc}") from exc
            payload = _strict_json_loads(
                text,
                source=self.journal_path,
                line_number=line_number,
            )
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "record_key",
                "identity",
                "result",
            }:
                raise ValueError(
                    f"Checkpoint entry fields do not match schema at {self.journal_path} line {line_number}"
                )
            if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
                raise ValueError(f"Unsupported checkpoint schema at {self.journal_path} line {line_number}")
            identity = RecordIdentity.from_dict(payload["identity"])
            record_key = payload["record_key"]
            if not isinstance(record_key, str) or record_key != identity.key:
                raise ValueError(f"Checkpoint record key mismatch at {self.journal_path} line {line_number}")
            result = payload["result"]
            if not isinstance(result, dict):
                raise ValueError(f"Checkpoint result must be an object at {self.journal_path} line {line_number}")
            entries[record_key] = CheckpointEntry(identity=identity, result=result)
        return entries

    def append(self, identity: RecordIdentity, result: dict[str, Any]) -> None:
        """Append and flush one completed file result from the parent process."""

        if self._stream is None:
            raise RuntimeError("CheckpointStore must be entered before append")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "record_key": identity.key,
            "identity": asdict(identity),
            "result": result,
        }
        try:
            line = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint result is not strict JSON: {exc}") from exc
        self._stream.write(line + "\n")
        self._stream.flush()
        self._since_fsync += 1
        if self._since_fsync >= self.fsync_every:
            os.fsync(self._stream.fileno())
            self._since_fsync = 0
        self.entries[identity.key] = CheckpointEntry(identity=identity, result=result)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PROCESSING_SCHEMA_VERSION",
    "CheckpointEntry",
    "CheckpointStore",
    "RecordIdentity",
    "build_checkpoint_manifest",
    "sha256_file",
]
