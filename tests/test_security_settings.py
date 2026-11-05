"""Тесты src/security_settings.py (без БД)."""

from __future__ import annotations

import json

import pytest

from src import config, security_settings


@pytest.fixture(autouse=True)
def _reset_security_cache():
    """Сбрасывать модульный кэш до и после каждого теста."""
    security_settings.reset_cache_for_tests()
    yield
    security_settings.reset_cache_for_tests()


def test_default_false_when_file_missing(monkeypatch, tmp_path):
    """1. При отсутствии файла флаг по умолчанию False."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert security_settings.two_factor_enabled() is False


def test_save_persists_and_cache_invalidation(monkeypatch, tmp_path):
    """2. Сохранение пишет JSON и cache перечитывается сразу."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    out = security_settings.save_security_settings({"two_factor_enabled": True})

    assert out["two_factor_enabled"] is True
    file = tmp_path / "settings" / "security.json"
    assert file.exists()
    assert json.loads(file.read_text(encoding="utf-8")) == {"two_factor_enabled": True}
    assert security_settings.two_factor_enabled() is True


def test_corrupt_json_returns_defaults(monkeypatch, tmp_path):
    """3. Битый JSON не падает, возвращает defaults."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    file = tmp_path / "settings" / "security.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("not-json", encoding="utf-8")

    assert security_settings.load_security_settings() == {"two_factor_enabled": False}
    assert security_settings.two_factor_enabled() is False


def test_unknown_keys_and_non_bools_ignored(monkeypatch, tmp_path):
    """4. Незнакомые ключи и не-bool значения игнорируются."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    security_settings.save_security_settings(
        {"two_factor_enabled": "yes", "foo": "bar"}
    )

    assert security_settings.load_security_settings() == {"two_factor_enabled": False}
    file = tmp_path / "settings" / "security.json"
    payload = json.loads(file.read_text(encoding="utf-8"))
    assert "foo" not in payload


def test_atomic_write_leaves_no_temp(monkeypatch, tmp_path):
    """5. После атомарной записи не остаётся временных файлов .security-*.json."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    security_settings.save_security_settings({"two_factor_enabled": True})

    settings_dir = tmp_path / "settings"
    assert list(settings_dir.glob(".security-*.json")) == []


def test_reset_cache_forces_reread(monkeypatch, tmp_path):
    """6. reset_cache_for_tests() сбрасывает кэш и читает файл заново."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    security_settings.save_security_settings({"two_factor_enabled": True})
    security_settings.two_factor_enabled()  # заполнить кэш

    file = tmp_path / "settings" / "security.json"
    file.write_text(json.dumps({"two_factor_enabled": False}), encoding="utf-8")

    # симулируем устаревший кэш: ключ совпадает с новым файлом, а значение - старое
    stat = file.stat()
    security_settings._cache_key = (str(file), stat.st_mtime_ns, stat.st_size)
    security_settings._cache_value = {"two_factor_enabled": True}

    assert security_settings.two_factor_enabled() is True  # старое значение из кэша
    security_settings.reset_cache_for_tests()
    assert security_settings.two_factor_enabled() is False
