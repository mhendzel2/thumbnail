from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.image_engine import ImageEngine, ThumbnailResult


class WorkerSignals(QObject):
    finished = pyqtSignal(str, object, dict)
    error = pyqtSignal(str, str)


class MetadataSignals(QObject):
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
