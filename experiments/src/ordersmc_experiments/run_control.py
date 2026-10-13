"""Run lifecycle recording that never silently drops an ordinary failure.

The recorder writes a ``running`` manifest before user code executes and a
terminal manifest on success or on a caught Python exception.  Exceptions are
re-raised after their raw failure metadata and traceback have been persisted.
This is intentionally a harness concern: it does not alter model execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Callable, Iterator, ParamSpec, TypeVar

from .manifest import RunManifest


P = ParamSpec("P")
R = TypeVar("R")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(manifest: RunManifest, path: str | Path) -> None:
    """Atomically replace the on-disk lifecycle row with the latest state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


class RunRecorder:
    """Persist one :class:`RunManifest` through its execution lifecycle."""

    def __init__(
        self, manifest: RunManifest, path: str | Path, *, validate: bool = True
    ) -> None:
        self.manifest = manifest
        self.path = Path(path)
        self.validate = bool(validate)
        self._wall_started: float | None = None
        self._cpu_started: float | None = None

    def start(self) -> None:
        if self._wall_started is not None:
            raise RuntimeError("run recorder has already been started")
        if self.validate:
            self.manifest.validate_for_execution()
        self.manifest.status = "running"
        self.manifest.failure_type = None
        self.manifest.failure_message = None
        self.manifest.failure_traceback = None
        self.manifest.failure_traceback_sha256 = None
        self.manifest.started_utc = _utc_now()
        self.manifest.completed_utc = None
        self.manifest.wall_seconds = None
        self.manifest.cpu_seconds = None
        self._wall_started = time.perf_counter()
        self._cpu_started = time.process_time()
        _write_atomic(self.manifest, self.path)

    def finish(self, status: str, error: BaseException | None = None) -> None:
        if self._wall_started is None or self._cpu_started is None:
            raise RuntimeError("run recorder has not been started")
        self.manifest.status = status
        self.manifest.completed_utc = _utc_now()
        self.manifest.wall_seconds = max(0.0, time.perf_counter() - self._wall_started)
        self.manifest.cpu_seconds = max(0.0, time.process_time() - self._cpu_started)
        if error is None:
            self.manifest.failure_type = None
            self.manifest.failure_message = None
            self.manifest.failure_traceback = None
            self.manifest.failure_traceback_sha256 = None
        else:
            self.manifest.failure_type = type(error).__name__
            self.manifest.failure_message = str(error)
            self.manifest.failure_traceback = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
            self.manifest.failure_traceback_sha256 = hashlib.sha256(
                self.manifest.failure_traceback.encode("utf-8")
            ).hexdigest()
            failure_diagnostics = getattr(error, "diagnostics", None)
            if failure_diagnostics is not None:
                self.manifest.diagnostic_record = {
                    **dict(self.manifest.diagnostic_record),
                    "failure_diagnostics": failure_diagnostics,
                }
        _write_atomic(self.manifest, self.path)


@contextmanager
def record_run(
    manifest: RunManifest, path: str | Path, *, validate: bool = True
) -> Iterator[RunManifest]:
    """Record success or a classified failure while preserving propagation.

    ``TimeoutError`` and ``MemoryError`` receive dedicated statuses.  Other
    ordinary ``Exception`` subclasses are recorded as ``error``.  Process-level
    termination such as ``KeyboardInterrupt`` and ``SystemExit`` is deliberately
    not relabelled as an ordinary experiment failure.
    """

    recorder = RunRecorder(manifest, path, validate=validate)
    recorder.start()
    try:
        yield manifest
    except TimeoutError as error:
        recorder.finish("timeout", error)
        raise
    except MemoryError as error:
        recorder.finish("oom", error)
        raise
    except Exception as error:
        recorder.finish("error", error)
        raise
    else:
        recorder.finish("completed")


def run_with_manifest(
    manifest: RunManifest,
    path: str | Path,
    function: Callable[P, R],
    /,
    *args: P.args,
    validate: bool = True,
    **kwargs: P.kwargs,
) -> R:
    """Execute ``function`` under :func:`record_run` and return its result."""

    with record_run(manifest, path, validate=validate):
        return function(*args, **kwargs)
