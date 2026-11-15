"""Fail-fast конфигурации: prod-окружение не принимает dev-дефолты (7.4.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

SECRET = "x" * 64
DEV_DSN = "postgresql+asyncpg://postgres:postgres@localhost:5432/compare_competencies"


def _settings(**over):
    from src.config import Settings

    return Settings(_env_file=None, SECRET_KEY=SECRET, **over)


def test_dev_allows_default_dsn():
    assert _settings().APP_ENV == "dev"
    assert _settings(DATABASE_URL=DEV_DSN).APP_ENV == "dev"


def test_prod_with_default_dsn_fails_fast():
    with pytest.raises(ValidationError) as exc:
        _settings(APP_ENV="prod", DATABASE_URL=DEV_DSN)
    assert "DATABASE_URL" in str(exc.value)


def test_prod_case_insensitive_and_ok_with_real_dsn():
    s = _settings(
        APP_ENV="PRODUCTION",
        DATABASE_URL="postgresql+asyncpg://app:pw@db.internal:5432/cc",
    )
    assert s.APP_ENV.lower() == "production"


def test_weak_secret_still_rejected():
    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, SECRET_KEY="short")
