"""Пул соединений БД: валидация конфига и прокидывание настроек в asyncpg (6.3.4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

SECRET = "x" * 64


def _settings(**over):
    from src.config import Settings

    return Settings(_env_file=None, SECRET_KEY=SECRET, **over)


def test_defaults_are_sane():
    s = _settings()
    assert s.DB_POOL_MIN >= 1
    assert s.DB_POOL_MAX >= s.DB_POOL_MIN
    assert s.DB_COMMAND_TIMEOUT > 0
    assert s.DB_POOL_ACQUIRE_TIMEOUT > 0
    assert s.DB_POOL_MAX_INACTIVE_LIFETIME > 0


@pytest.mark.parametrize(
    "over",
    [
        {"DB_POOL_MIN": 0},
        {"DB_POOL_MIN": 5, "DB_POOL_MAX": 2},
        {"DB_COMMAND_TIMEOUT": 0},
        {"DB_POOL_ACQUIRE_TIMEOUT": 0},
        {"DB_POOL_MAX_INACTIVE_LIFETIME": 0},
    ],
)
def test_invalid_pool_config_fails_fast(over):
    with pytest.raises(ValidationError):
        _settings(**over)


async def test_create_pool_passes_configured_values(monkeypatch):
    import src.db as db

    captured = {}

    async def _fake_create_pool(dsn, **kwargs):
        captured["dsn"] = dsn
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(db.asyncpg, "create_pool", _fake_create_pool)
    monkeypatch.setattr(db, "pool", None)

    await db.create_pool()

    assert captured["min_size"] == db.settings.DB_POOL_MIN
    assert captured["max_size"] == db.settings.DB_POOL_MAX
    assert captured["command_timeout"] == db.settings.DB_COMMAND_TIMEOUT
    assert captured["timeout"] == db.settings.DB_POOL_ACQUIRE_TIMEOUT
    assert captured["max_inactive_connection_lifetime"] == db.settings.DB_POOL_MAX_INACTIVE_LIFETIME
