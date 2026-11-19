"""Паритет ORM-моделей и схемы: барьер против «слепого» alembic (6.3.2).

`alembic check` видит только те таблицы, чьи модели импортированы в
`Base.metadata`. Если новый модуль моделей не импортирован в `alembic/env.py`,
дрейф молча перестаёт ловиться - этот тест падает первым.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _model_modules_with_tables() -> set[str]:
    """Модули src/models/*.py, объявляющие классы на Base."""
    mods: set[str] = set()
    for p in (ROOT / "src" / "models").glob("*.py"):
        if p.name == "__init__.py":
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^class\s+\w+\s*\(\s*Base\s*\)", text, re.M):
            mods.add(p.stem)
    return mods


def test_env_imports_every_model_module():
    env = (ROOT / "alembic" / "env.py").read_text(encoding="utf-8", errors="ignore")
    imported = set(re.findall(r"from src\.models import ([^\n#]+)", env))
    names: set[str] = set()
    for chunk in imported:
        names.update(n.strip() for n in chunk.split(",") if n.strip())

    missing = sorted(_model_modules_with_tables() - names)
    assert not missing, (
        "alembic/env.py не импортирует модули моделей (alembic check станет слепым): "
        + ", ".join(missing)
    )


def test_live_totp_column_present_in_orm():
    """last_totp_step - живая колонка (анти-replay TOTP), не legacy."""
    from src.models.extra_models import AdminTotp

    assert "last_totp_step" in AdminTotp.__table__.c


def test_base_metadata_is_populated():
    import src.database as db
    from src.models import extra_models, krm_models  # noqa: F401

    tables = set(db.Base.metadata.tables)
    for required in ("users", "vacancies", "admin_totp", "app_settings", "totp_tokens"):
        assert required in tables, f"нет таблицы {required} в Base.metadata"
