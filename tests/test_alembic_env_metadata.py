"""Барьер: модели зарегистрированы в Base.metadata (иначе alembic «слеп»).

Проверяем функционально, без запуска alembic: после импорта моделей в
`src.database.Base.metadata` есть таблицы. Это защищает от регресса, когда
`alembic/env.py` импортирует только Base и autogenerate/check видят пустую
метадату («дрейф на удаление всех таблиц»).

Отдельно проверяем таблицы, исторически создававшиеся сырым SQL (auth/TOTP/
кэши/settings): они должны быть в ORM (`src/models/extra_models.py`), иначе
`alembic check` снова покажет дрейф (backlog 6.3.2).
"""

from __future__ import annotations

# Таблицы вне исходного ORM, добавленные в src/models/extra_models.py.
_RAW_SQL_TABLES = (
    "llm_cache",
    "skill_embedding_cache",
    "admin_totp",
    "totp_tokens",
    "trend_precomputed",
    "app_settings",
    "discipline_scope",
    "student_actions",
)


def test_models_registered_in_base_metadata():
    # импорт моделей регистрирует таблицы (как это делает alembic/env.py)
    import src.models  # noqa: F401
    from src.database import Base

    assert len(Base.metadata.tables) >= 35, (
        "Base.metadata почти пуста: модели не импортированы, alembic снова станет «слепым»"
    )
    for table in ("users", "vacancies", "directions"):
        assert table in Base.metadata.tables, table


def test_raw_sql_tables_have_orm_models():
    import src.models  # noqa: F401
    from src.database import Base

    missing = [t for t in _RAW_SQL_TABLES if t not in Base.metadata.tables]
    assert not missing, f"нет ORM-моделей для таблиц: {missing}"
