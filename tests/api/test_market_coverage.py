"""Покрытие роутера market: витрина рынка (backlog #49).

Часть ручек уже тронута test_api.py (top-skills / skill/{skill}) и
test_security_p0.py (401). Этот файл добивает непокрытое:
- тело GET /market-competencies: метаданные из пула (fake pool, БД не трогаем),
  отсутствие пула, ошибка пула (глотается, ручка не 500),
- границы: cap top-100 в market-competencies, limit=1/limit=51 в top-skills,
  неизвестый навык в skill/{skill}.

Гейт require_authenticated здесь живой (401-тест). open_all_gates открывает
гейты только для happy-path фикстуры - FastAPI 0.162 включает роутеры лениво,
поэтому dependency_overrides расставляет именно open_all_gates (см.
tests/api/test_vacancies_router_coverage.py).
"""

from __future__ import annotations

import contextlib
from datetime import date

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.market as market_mod
import src.db as src_db
from src.api_pkg import create_app, deps
from tests.conftest import open_all_gates


class _MetaPool:
    """asyncpg-подобный пул: только fetchrow с метаданными вакансий."""

    def __init__(self, *, row=None, error: Exception | None = None):
        self._row = row
        self._error = error
        self.calls = 0

    async def fetchrow(self, _sql, *_args):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._row


@pytest.fixture()
def app_client():
    """(app, client): свежее приложение с открытыми гейтами.

    Overrides кладутся в app.dependency_overrides в каждом тесте и умирают
    вместе с приложением - течь нечему.
    """
    app = create_app()
    open_all_gates(app, role="admin")
    with contextlib.suppress(Exception):
        market_mod.limiter.reset()
    return app, TestClient(app, raise_server_exceptions=False)


# ----------------------------------------------------- market-competencies


def test_market_competencies_with_pool_meta(app_client, monkeypatch):
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.5, "sql": 0.4}
    pool = _MetaPool(row={"n": 42, "mn": date(2026, 1, 1), "mx": date(2026, 2, 1)})
    monkeypatch.setattr(src_db, "get_pool", lambda: pool)

    r = client.get("/api/market-competencies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["skill"] for s in body["skills"]] == ["python", "sql"], "сортировка по весу"
    assert body["total"] == 2
    assert pool.calls == 1
    assert body["vacancy_count"] == 42
    assert body["date_from"] == "2026-01-01"
    assert body["date_to"] == "2026-02-01"


def test_market_competencies_zero_sample_boundary(app_client, monkeypatch):
    """Граница: n=0 и дат нет - мета вырождается честно, без 500."""
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 1.0}
    pool = _MetaPool(row={"n": 0, "mn": None, "mx": None})
    monkeypatch.setattr(src_db, "get_pool", lambda: pool)

    r = client.get("/api/market-competencies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["vacancy_count"] == 0
    assert body["date_from"] is None
    assert body["date_to"] is None


def test_market_competencies_caps_top_100(app_client, monkeypatch):
    """Граница: топ больше 100 навыков режется, total остаётся честным."""
    app, client = app_client
    weights = {f"skill-{i:03d}": float(i) for i in range(150)}
    app.dependency_overrides[deps.get_skill_weights] = lambda: weights
    monkeypatch.setattr(src_db, "get_pool", lambda: None)

    r = client.get("/api/market-competencies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 150
    assert len(body["skills"]) == 100
    assert body["skills"][0]["skill"] == "skill-149", "первым - максимальный вес"
    assert [s["skill"] for s in body["skills"]][-1] == "skill-050", "хвост отсечён"


def test_market_competencies_without_pool(app_client, monkeypatch):
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {}
    monkeypatch.setattr(src_db, "get_pool", lambda: None)

    r = client.get("/api/market-competencies")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "skills": [], "total": 0, "vacancy_count": None,
        "date_from": None, "date_to": None,
    }


def test_market_competencies_pool_error_swallowed(app_client, monkeypatch):
    """Ошибка ветки, доступная только с fake-пулом: пул упал - ручка не 500."""
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 1.0}
    pool = _MetaPool(error=RuntimeError("db down"))
    monkeypatch.setattr(src_db, "get_pool", lambda: pool)

    r = client.get("/api/market-competencies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["skills"][0]["skill"] == "python", "навыки остаются"
    assert body["vacancy_count"] is None, "метаданные опустились"


# ----------------------------------------------------- top-skills (границы)


def test_top_skills_limit_1_boundary(app_client):
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {
        "python": 0.123456, "sql": 0.4,
    }
    app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 100, "sql": 50}

    r = client.get("/api/market/top-skills?limit=1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["skills"]) == 1
    assert body["skills"][0] == {"skill": "sql", "weight": 0.4, "frequency": 50}


def test_top_skills_rounding_boundary(app_client):
    """Вес 0.123456 округляется до 4 знаков, frequency приводится к int."""
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.123456}
    app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 100}

    r = client.get("/api/market/top-skills")
    assert r.status_code == 200, r.text
    skill = r.json()["skills"][0]
    assert skill["weight"] == 0.1235
    assert isinstance(skill["frequency"], int) and skill["frequency"] == 100


def test_top_skills_limit_over_max_is_422(app_client):
    """Граница le=50: limit=51 - ошибка валидации, не тише."""
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 1.0}
    app.dependency_overrides[deps.get_skill_freq] = lambda: {}

    r = client.get("/api/market/top-skills?limit=51")
    assert r.status_code == 422, r.text


# ----------------------------------------------------- skill/{skill}


def test_skill_info_unknown_skill_boundary(app_client):
    """Незнакомый навык: нули и unknown, а не 404 (справочник - не словарь)."""
    app, client = app_client
    app.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 0.5}
    app.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 10}
    app.dependency_overrides[deps.get_taxonomy] = lambda: None

    r = client.get("/api/market/skill/fortran")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "skill": "fortran", "frequency": 0, "weight": 0.0,
        "category": "unknown", "icon": "",
    }


# ----------------------------------------------------- гейт require_authenticated


def test_market_endpoints_require_auth_401():
    """Без токена все три ручки получают 401 (роль не проверяется - 403 тут нет)."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    for path in (
        "/api/market/top-skills",
        "/api/market-competencies",
        "/api/market/skill/python",
    ):
        r = client.get(path)
        assert r.status_code == 401, f"{path} -> {r.status_code}"
