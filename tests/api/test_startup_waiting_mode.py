"""Режим ожидания не должен отключать таксономию.

Найденный баг: `_set_waiting_mode()` ставил `deps.taxonomy = None`, хотя
таксономия - файловая и от снапшотов вакансий не зависит. Итог: при отсутствии
вакансий `/api/taxonomy/*` отдавал 503, хотя сервис жив (/health и /ready = 200).

Второй дефект: ошибка загрузки таксономии глушилась (`except Exception: None`)
без записи в лог, поэтому причину 503 нельзя было увидеть.
"""

import pytest

import src.api_pkg.deps as deps
import src.api_pkg.startup as startup


class _StubClusterer:
    """Кластеру в этом тесте нечего делать: модели не нужны."""

    def load_model(self, _level):
        return True


@pytest.fixture()
def isolated_deps(monkeypatch):
    monkeypatch.setattr(deps, "get_clusterer", lambda: _StubClusterer(), raising=False)
    saved = {
        name: getattr(deps, name, None)
        for name in ("taxonomy", "is_ready", "basic_vacancies", "current_skills_set")
    }
    yield
    for name, value in saved.items():
        setattr(deps, name, value)


async def test_waiting_mode_keeps_taxonomy_loaded(isolated_deps):
    """Таксономия остаётся доступной: она не зависит от вакансий."""
    await startup._set_waiting_mode()

    assert deps.taxonomy is not None, "в режиме ожидания таксономия не должна обнуляться"
    # Она действительно рабочая, а не заглушка.
    assert deps.taxonomy.get_all_categories().is_ok()
    assert deps.is_ready is True


async def test_waiting_mode_still_clears_vacancy_data(isolated_deps):
    """Всё, что зависит от вакансий, по-прежнему очищается."""
    deps.basic_vacancies = [{"id": "x"}]
    await startup._set_waiting_mode()

    assert deps.basic_vacancies == []
    assert deps.vacancy_load_error == "not_found"


async def test_taxonomy_failure_is_logged_not_silent(isolated_deps, monkeypatch):
    """Сбой загрузки таксономии виден в логах, а не проглатывается."""
    calls: list[str] = []

    class _BrokenTaxonomy:
        def __init__(self):
            raise RuntimeError("taxonomy file corrupted")

    class _SpyLogger:
        def exception(self, event, **_kw):
            calls.append(event)

        def __getattr__(self, _name):
            return lambda *a, **kw: None

    monkeypatch.setattr(startup, "SkillTaxonomy", _BrokenTaxonomy)
    monkeypatch.setattr(startup, "logger", _SpyLogger())

    await startup._set_waiting_mode()

    assert deps.taxonomy is None
    assert "taxonomy_load_failed_in_waiting_mode" in calls
