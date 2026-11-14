"""Барьер: модели зарегистрированы в Base.metadata (иначе alembic «слеп»).

Проверяем функционально, без запуска alembic: после импорта моделей в
`src.database.Base.metadata` есть таблицы. Это защищает от регресса, когда
`alembic/env.py` импортирует только Base и autogenerate/check видят пустую
метадату («дрейф на удаление всех таблиц»).
"""

from __future__ import annotations


def test_models_registered_in_base_metadata():
    # импорт моделей регистрирует таблицы (как это делает alembic/env.py)
    import src.models.krm_models  # noqa: F401
    from src.database import Base

    assert len(Base.metadata.tables) >= 20, (
        "Base.metadata почти пуста: модели не импортированы, alembic снова станет «слепым»"
    )
    assert "users" in Base.metadata.tables
    assert "vacancies" in Base.metadata.tables
