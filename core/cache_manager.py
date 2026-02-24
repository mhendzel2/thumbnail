from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from diskcache import Cache


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
            self.memory.set(key, value)
        return value

    def set(self, key: str, value: Any) -> None:
        self.memory.set(key, value)
        self.disk.set(key, value)

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
