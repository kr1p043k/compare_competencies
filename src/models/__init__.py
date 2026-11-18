"""ORM-модели. Импорт всех модулей регистрирует таблицы в Base.metadata.

Важно для Alembic: `alembic/env.py` импортирует этот пакет, чтобы
autogenerate/`alembic check` видели ВСЕ таблицы (иначе ложный дрейф).
"""
from src.models import (  # noqa: F401
    extra_models,
    krm_models,
)

__all__ = ["krm_models", "extra_models"]
