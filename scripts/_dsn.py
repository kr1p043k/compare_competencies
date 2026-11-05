"""Единое получение DSN для скриптов.

Раньше в каждом скрипте лежал литерал с логином и паролем Postgres
(`postgresql://postgres:...@localhost/...`). Такой литерал попадает в
репозиторий и в историю, а при публикации репозитория утекает вместе с ним.

Теперь порядок поиска:
1. явный аргумент `--dsn`;
2. переменная окружения `DATABASE_URL` (или `COMPARE_DB_DSN`);
3. локальный `.env` (строка `DATABASE_URL=...`);
4. иначе - понятная ошибка с инструкцией, а не тихий дефолт с паролем.

Локальный `.env` в репозиторий не попадает (он в .gitignore), поэтому секрет
живёт только на машине разработчика.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_DRIVER_SUFFIXES = ("+asyncpg", "+psycopg", "+psycopg2")


class DSNNotConfigured(RuntimeError):
    """DSN не найден ни в аргументах, ни в окружении, ни в .env."""


def _read_dotenv() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("DATABASE_URL", "COMPARE_DB_DSN"):
                return value.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _strip_driver(url: str) -> str:
    for suffix in _DRIVER_SUFFIXES:
        url = url.replace(suffix, "")
    return url


def resolve_dsn(explicit: str | None = None) -> str:
    """DSN для синхронного psycopg. Бросает DSNNotConfigured, если пусто."""
    for candidate in (
        explicit,
        os.environ.get("COMPARE_DB_DSN"),
        os.environ.get("DATABASE_URL"),
        _read_dotenv(),
    ):
        if candidate and candidate.strip():
            return _strip_driver(candidate.strip())

    raise DSNNotConfigured(
        "DSN не задан. Укажите --dsn, переменную DATABASE_URL "
        "(или COMPARE_DB_DSN), либо строку DATABASE_URL в .env. "
        "Пароль в коде не хранится намеренно."
    )
