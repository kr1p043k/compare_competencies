"""Кэш SHAP-объяснений (BACKLOG 6.6.4).

Проверяем без модели и БД: фейковый движок считает вызовы; кэш обязан
переиспользовать результат, пока не истёк TTL / не сменились навыки или профиль.
"""

from __future__ import annotations

import pytest

import src.api_pkg.routers.results as res
from src import Ok


class _Engine:
    is_fitted = True
    skill_metadata = {"alpha": {}, "beta": {}}

    def __init__(self) -> None:
        self.calls = 0

    def predict_skill_impact_with_shap(self, _skills, _missing, _flag, _n):
        self.calls += 1
        return Ok(([("beta", 0.9, "потому что")], None, None))


class _Profile:
    def __init__(self, skills: list[str] | None = None) -> None:
        self.skills = ["alpha"] if skills is None else skills
        self.competencies: list[str] = []
        self.technologies: list[str] = []


def _fn():
    # slowapi-декоратор требует request; тестируем исходную функцию
    return getattr(res.get_ml_explanations, "__wrapped__", res.get_ml_explanations)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    res._expl_cache_clear()
    monkeypatch.setattr(res, "_EXPL_CACHE_TTL_S", 300.0)
    yield
    res._expl_cache_clear()


@pytest.fixture
def engine(monkeypatch) -> _Engine:
    eng = _Engine()
    monkeypatch.setattr(res, "_get_or_load_explanation_engine", lambda: eng)
    return eng


async def test_second_call_is_served_from_cache(engine):
    fn = _fn()
    profiles = {"p": _Profile()}

    first = await fn(request=None, profile="p", top=5, profiles=profiles)
    second = await fn(request=None, profile="p", top=5, profiles=profiles)

    assert engine.calls == 1, "второй вызов должен брать результат из кэша"
    assert first == second
    assert first["explanations"][0]["skill"] == "beta"


async def test_changed_skills_recompute(engine):
    fn = _fn()
    first = await fn(request=None, profile="p", top=5, profiles={"p": _Profile()})
    second = await fn(request=None, profile="p", top=5, profiles={"p": _Profile(["alpha", "gamma"])})
    assert engine.calls == 2
    assert first["profile"] == second["profile"] == "p"


async def test_expired_ttl_recomputes(engine, monkeypatch):
    monkeypatch.setattr(res, "_EXPL_CACHE_TTL_S", 0.0)
    fn = _fn()
    profiles = {"p": _Profile()}
    await fn(request=None, profile="p", top=5, profiles=profiles)
    await fn(request=None, profile="p", top=5, profiles=profiles)
    assert engine.calls == 2


async def test_different_profile_has_own_entry(engine):
    fn = _fn()
    profiles = {"p": _Profile(), "q": _Profile()}
    await fn(request=None, profile="p", top=5, profiles=profiles)
    await fn(request=None, profile="q", top=5, profiles=profiles)
    assert engine.calls == 2


def test_cache_is_bounded():
    for i in range(res._EXPL_CACHE_MAX + 10):
        res._expl_cache_put((f"k{i}", 1, i), {"explanations": []})
    assert len(res._expl_cache) <= res._EXPL_CACHE_MAX
