from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from diskcache import Cache
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QPixmap


PIXMAP_DISK_TAG = "__qt_pixmap_png__"


class LruMemoryCache:
    def __init__(self, max_items: int = 512):
        self._max_items = max_items
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                return None
            value = self._store.pop(key)
            self._store[key] = value
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            self._store[key] = value
            while len(self._store) > self._max_items:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())


class CacheManager:
    def __init__(self, ram_items: int = 512, disk_path: str | None = None):
        cache_dir = Path(disk_path) if disk_path else Path.home() / ".microscopy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory = LruMemoryCache(max_items=ram_items)
        self.disk = Cache(str(cache_dir))

    def get(self, key: str) -> Any | None:
        value = self.memory.get(key)
        if value is not None:
            return value

        value = self.disk.get(key, default=None)
        if value is not None:
            restored = self._deserialize_disk_value(value)
            self.memory.set(key, restored)
            return restored
        return None

    def set(self, key: str, value: Any) -> None:
        self.memory.set(key, value)
        self.disk.set(key, self._serialize_disk_value(value))

    def delete(self, key: str) -> None:
        self.memory.delete(key)
        self.disk.delete(key)

    def clear(self) -> None:
        self.memory.clear()
        self.disk.clear()

    def delete_prefix(self, prefix: str) -> int:
        removed = 0

        for key in self.memory.keys():
            if key.startswith(prefix):
                self.memory.delete(key)
                removed += 1

        for key in list(self.disk.iterkeys()):
            if isinstance(key, str) and key.startswith(prefix):
                self.disk.delete(key)
                removed += 1

        return removed

    def delete_contains(self, token: str) -> int:
        removed = 0

        for key in self.memory.keys():
            if token in key:
                self.memory.delete(key)
                removed += 1

        for key in list(self.disk.iterkeys()):
            if isinstance(key, str) and token in key:
                self.disk.delete(key)
                removed += 1

        return removed

    def close(self) -> None:
        self.disk.close()

    def cache_db_path(self) -> str:
        return str(Path(self.disk.directory) / "cache.db")

    def set_folder_record(self, folder_path: str, payload: dict[str, Any]) -> None:
        key = self._folder_key(folder_path)
        self.disk.set(key, payload)

    def get_folder_record(self, folder_path: str) -> dict[str, Any] | None:
        key = self._folder_key(folder_path)
        value = self.disk.get(key, default=None)
        if isinstance(value, dict):
            return value
        return None

    def _serialize_disk_value(self, value: Any) -> Any:
        if isinstance(value, QPixmap):
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            if buffer.open(QIODevice.OpenModeFlag.WriteOnly):
                ok = value.save(buffer, "PNG")
                buffer.close()
                if ok:
                    return {PIXMAP_DISK_TAG: byte_array.data()}
        return value

    def _deserialize_disk_value(self, value: Any) -> Any:
        if isinstance(value, dict) and PIXMAP_DISK_TAG in value:
            raw = value.get(PIXMAP_DISK_TAG)
            if isinstance(raw, (bytes, bytearray)):
                pixmap = QPixmap()
                if pixmap.loadFromData(raw, "PNG"):
                    return pixmap
            return None
        return value

    def _folder_key(self, folder_path: str) -> str:
        return f"folder::meta::{folder_path}"
