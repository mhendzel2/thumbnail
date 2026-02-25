from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.image_engine import ImageEngine, ThumbnailResult


class WorkerSignals(QObject):
    finished = pyqtSignal(str, object, dict)
    error = pyqtSignal(str, str)


class MetadataSignals(QObject):
    finished = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)


class DriveScanSignals(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, dict)
    error = pyqtSignal(str, str)


@dataclass(slots=True)
class ThumbnailJob:
    file_path: str
    size: int
    cache_key: str
    slice_request: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ThumbnailWorker(QRunnable):
    def __init__(
        self,
        engine: ImageEngine,
        job: ThumbnailJob,
        *,
        include_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.job = job
        self.include_metadata = include_metadata
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result: ThumbnailResult = self.engine.load_thumbnail(
                self.job.file_path,
                self.job.size,
                slice_request=self.job.slice_request,
                include_metadata=self.include_metadata,
            )
            payload = dict(result.metadata)
            payload["cache_key"] = self.job.cache_key
            payload.update(self.job.extra)
            self.signals.finished.emit(self.job.file_path, result.pixmap, payload)
        except Exception as exc:
            self.signals.error.emit(self.job.file_path, str(exc))


class MetadataWorker(QRunnable):
    def __init__(self, engine: ImageEngine, file_path: str) -> None:
        super().__init__()
        self.engine = engine
        self.file_path = file_path
        self.signals = MetadataSignals()

    def run(self) -> None:
        try:
            metadata = self.engine.load_metadata(self.file_path)
            self.signals.finished.emit(self.file_path, metadata)
        except Exception as exc:
            self.signals.error.emit(self.file_path, str(exc))


class DriveScanWorker(QRunnable):
    def __init__(
        self,
        engine: ImageEngine,
        cache_manager,
        drive_root: str,
        thumbnail_size: int,
        supported_suffixes: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.engine = engine
        self.cache_manager = cache_manager
        self.drive_root = drive_root
        self.thumbnail_size = thumbnail_size
        self.supported_suffixes = tuple(s.lower() for s in supported_suffixes)
        self.signals = DriveScanSignals()
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _safe_emit(self, emitter: Callable, *args) -> None:
        try:
            emitter(*args)
        except RuntimeError:
            pass

    def run(self) -> None:
        started = time.perf_counter()
        scanned = 0
        generated = 0
        skipped_cached = 0

        try:
            for dir_path, _dir_names, file_names in os.walk(self.drive_root):
                if self._stop_requested:
                    return

                for file_name in file_names:
                    if self._stop_requested:
                        return

                    lower = file_name.lower()
                    if not any(lower.endswith(suffix) for suffix in self.supported_suffixes):
                        continue

                    scanned += 1
                    file_path = os.path.join(dir_path, file_name)
                    cache_key = f"thumb::{self.thumbnail_size}::{file_path}"

                    try:
                        if self.cache_manager.get(cache_key) is not None:
                            skipped_cached += 1
                            continue

                        result: ThumbnailResult = self.engine.load_thumbnail(
                            file_path,
                            self.thumbnail_size,
                            include_metadata=False,
                        )
                        if bool(result.metadata.get("broken")):
                            continue
                        self.cache_manager.set(cache_key, result.pixmap)
                        generated += 1
                    except Exception:
                        continue

                    if scanned % 50 == 0:
                        self._safe_emit(
                            self.signals.progress.emit,
                            f"Drive scan {self.drive_root}: scanned {scanned} files, cached {generated}"
                        )

            elapsed_s = round(time.perf_counter() - started, 1)
            self._safe_emit(
                self.signals.finished.emit,
                self.drive_root,
                {
                    "scanned": scanned,
                    "generated": generated,
                    "skipped_cached": skipped_cached,
                    "elapsed_s": elapsed_s,
                },
            )
        except Exception as exc:
            self._safe_emit(self.signals.error.emit, self.drive_root, str(exc))
