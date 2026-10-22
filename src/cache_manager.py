from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import structlog

from src import CacheError, Err, Ok, Result

logger = structlog.get_logger("cache_manager")


class CacheManager:
    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _safe_key(self, key: str) -> str:
        """Ключ -> безопасное имя файла.

        Простые ключи сохраняют вид ('k' -> 'k.json'), чтобы не ломать
        существующие кэши. Если ключ требует экранирования, добавляем хэш
        исходного: без этого 'a/b' и 'a_b' давали один файл (2026-10-12).
        """
        import hashlib
        import re as _re

        cleaned = key.replace("..", "_")
        cleaned = _re.sub(r"[^\w.\-]", "_", cleaned)
        if cleaned == key:
            return cleaned
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned}-{digest}"

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(key)}.joblib"

    def save(self, key: str, data: Any) -> Result[None, CacheError]:
        path = self._path(key)
        try:
            # Атомарно: tmp + os.replace. Прямая запись оставляла обрезанный
            # pickle при падении/гонке — кэш потом читался как Err навсегда.
            import os
            import tempfile

            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".joblib.tmp")
            os.close(fd)
            try:
                joblib.dump(data, tmp)
                os.replace(tmp, path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return Ok(None)
        except Exception as e:
            return Err(CacheError(message=f"Save failed: {e}", cache_path=str(path)))

    def load(self, key: str) -> Result[Any, CacheError]:
        path = self._path(key)
        if not path.exists():
            return Err(CacheError(message="Cache miss", cache_path=str(path)))
        try:
            data = joblib.load(path)
            return Ok(data)
        except Exception as e:
            return Err(CacheError(message=f"Load failed: {e}", cache_path=str(path)))

    def invalidate(self, key: str) -> Result[bool, CacheError]:
        path = self._path(key)
        if not path.exists():
            return Ok(False)
        try:
            path.unlink()
            return Ok(True)
        except Exception as e:
            return Err(CacheError(message=f"Invalidate failed: {e}", cache_path=str(path)))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def clear_all(self) -> Result[int, CacheError]:
        count = 0
        for p in self.cache_dir.glob("*.joblib"):
            try:
                p.unlink()
                count += 1
            except Exception as e:
                logger.warning("cache_cleanup_failed", path=str(p), error=str(e))
        return Ok(count)


class JsonCacheManager(CacheManager):
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(key)}.json"

    def save(self, key: str, data: Any) -> Result[None, CacheError]:
        path = self._path(key)
        try:
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
            os.close(fd)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return Ok(None)
        except Exception as e:
            return Err(CacheError(message=f"JSON save failed: {e}", cache_path=str(path)))

    def load(self, key: str) -> Result[Any, CacheError]:
        path = self._path(key)
        if not path.exists():
            return Err(CacheError(message="Cache miss", cache_path=str(path)))
        try:
            with open(path, encoding="utf-8") as f:
                return Ok(json.load(f))
        except Exception as e:
            return Err(CacheError(message=f"JSON load failed: {e}", cache_path=str(path)))
