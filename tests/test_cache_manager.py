"""Tests for CacheManager and JsonCacheManager."""
import json
import tempfile
from pathlib import Path
import os

import pytest

from src.cache_manager import CacheManager, JsonCacheManager


class TestCacheManager:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = CacheManager(self.tmp)

    def test_init_creates_dir(self):
        p = self.tmp / "sub"
        CacheManager(p)
        assert p.exists()

    def test_save_and_load(self):
        data = {"key": "value", "num": 42}
        assert self.cache.save("test", data).is_ok()
        result = self.cache.load("test")
        assert result.is_ok()
        assert result.ok() == data

    def test_load_miss(self):
        result = self.cache.load("nonexistent")
        assert result.is_err()

    def test_exists(self):
        assert not self.cache.exists("missing")
        self.cache.save("exists", "val")
        assert self.cache.exists("exists")

    def test_invalidate_existing(self):
        self.cache.save("del", "val")
        result = self.cache.invalidate("del")
        assert result.is_ok()
        assert result.ok() is True
        assert not self.cache.exists("del")

    def test_invalidate_missing(self):
        result = self.cache.invalidate("nope")
        assert result.is_ok()
        assert result.ok() is False

    def test_clear_all(self):
        self.cache.save("a", 1)
        self.cache.save("b", 2)
        count = self.cache.clear_all().ok()
        assert count >= 2
        assert not self.cache.exists("a")

    def test_save_failure(self, monkeypatch):
        cache = CacheManager(self.tmp)

        def fail_dump(*a, **kw):
            raise OSError("disk full")

        import joblib
        monkeypatch.setattr(joblib, "dump", fail_dump)
        result = cache.save("key", "data")
        assert result.is_err()

    def test_load_corrupted(self):
        path = self.cache._path("corrupt")
        path.write_text("not joblib data")
        result = self.cache.load("corrupt")
        assert result.is_err()

    def test_safe_key(self):
        key = "../../etc/passwd"
        path = self.cache._path(key)
        assert ".." not in path.name

    def test_invalidate_failure(self, monkeypatch):
        self.cache.save("test", "val")
        original_unlink = os.unlink

        def fail_unlink(*a, **kw):
            raise OSError("permission denied")

        monkeypatch.setattr(os, "unlink", fail_unlink)
        result = self.cache.invalidate("test")
        assert result.is_err()

    def test_clear_all_empty(self):
        result = self.cache.clear_all()
        assert result.is_ok()

    def test_exists_after_clear(self):
        self.cache.save("x", 1)
        self.cache.clear_all()
        assert not self.cache.exists("x")


class TestJsonCacheManager:
    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = JsonCacheManager(self.tmp)

    def test_save_and_load(self):
        data = {"hello": "world"}
        assert self.cache.save("test", data).is_ok()
        result = self.cache.load("test")
        assert result.is_ok()
        assert result.ok() == data

    def test_load_miss(self):
        result = self.cache.load("missing")
        assert result.is_err()

    def test_uses_json_extension(self):
        self.cache.save("k", "v")
        assert (self.cache.cache_dir / "k.json").exists()

    def test_save_failure(self, monkeypatch):
        def fail_open(*a, **kw):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", fail_open)
        result = self.cache.save("key", "data")
        assert result.is_err()

    def test_load_corrupted(self):
        (self.tmp / "bad.json").write_text("{not json")
        result = self.cache.load("bad")
        assert result.is_err()

    def test_safe_key_json(self):
        key = "../evil"
        path = self.cache._path(key)
        assert ".." not in path.name

    def test_save_unicode(self):
        data = {"русский": "текст"}
        assert self.cache.save("unicode", data).is_ok()
        with open(self.tmp / "unicode.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_load_failure(self, monkeypatch):
        self.cache.save("ok", "value")

        def fail_open(*a, **kw):
            raise OSError("read error")

        monkeypatch.setattr("builtins.open", fail_open)
        result = self.cache.load("ok")
        assert result.is_err()
