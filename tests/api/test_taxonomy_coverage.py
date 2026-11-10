"""Покрытие роутера taxonomy: список профессий, карточка, KRM-покрытие.

/taxonomy/professions, /taxonomy/profession/{name} и .../krm-coverage не имели
HTTP-тестов (backlog #49): покрыты только /taxonomy/coverage и
/taxonomy/categories. Ручки импортируют ProfessionTaxonomy ВНУТРИ тела из
модуля src.analyzers.skills.profession_taxonomy - подменяем атрибут модуля
monkeypatch-ем, реальную файловую таксономию не читаем.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.analyzers.skills.profession_taxonomy as pt_mod
import src.api_pkg.routers.taxonomy as tax
from src.api_pkg import create_app
from tests.conftest import open_all_gates


class _FakeProfessionTaxonomy:
    """Минимальный контракт, который дёргают три эндпоинта профессий."""

    professions = ["data_engineer", "analyst"]

    def __init__(self, *, fail_info: bool = False, coverage: dict | None = None, fail_coverage: bool = False):
        self._fail_info = fail_info
        self._coverage = coverage if coverage is not None else {}
        self._fail_coverage = fail_coverage
        self.calls: list[tuple] = []

    def get_profession_info(self, name):
        self.calls.append(("info", name))
        if self._fail_info:
            raise RuntimeError("taxonomy boom")
        if name == "data_engineer":
            return {
                "domains": ["Данные"],
                "competency_codes": ["IT-1"],
                "hh_queries": ["инженер данных"],
                "aliases": ["de"],
            }
        if name == "analyst":
            # без остальных ключей - ручка обязана подставить дефолты []
            return {"domains": ["Аналитика"]}
        return None

    def get_profession_skills(self, name):
        self.calls.append(("skills", name))
        if name == "data_engineer":
            return [f"skill_{i}" for i in range(105)]
        return ["sql"]

    def get_profession_competency_codes(self, name):
        self.calls.append(("codes", name))
        return ["IT-1"] if name == "data_engineer" else []

    def get_competency_skills(self, code):
        self.calls.append(("comp_skills", code))
        return [f"krm_{i}" for i in range(25)]

    def compute_krm_coverage(self, name, user_skills):
        self.calls.append(("coverage", name, tuple(user_skills)))
        if self._fail_coverage:
            raise RuntimeError("coverage boom")
        return {k: dict(v) for k, v in self._coverage.items()}


def _client(monkeypatch, factory, *, role="teacher") -> TestClient:
    """Гейты открыты; ProfessionTaxonomy() внутри ручек вернёт наш фейк."""
    app = create_app()
    open_all_gates(app, role=role)
    monkeypatch.setattr(pt_mod, "ProfessionTaxonomy", factory)
    for lim in (getattr(app.state, "limiter", None), getattr(tax, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app, raise_server_exceptions=False)


# ------------------------------------------------------------- гейты (401)

@pytest.mark.parametrize(
    "url",
    [
        "/api/taxonomy/professions",
        "/api/taxonomy/profession/data_engineer",
        "/api/taxonomy/profession/data_engineer/krm-coverage",
    ],
)
def test_profession_endpoints_require_auth_401(url):
    """Router-level Depends(require_authenticated()): без токена - 401."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get(url)
    assert r.status_code == 401, (url, r.text)


# ------------------------------------------------------- /taxonomy/professions


def test_professions_list(monkeypatch):
    fake = _FakeProfessionTaxonomy()
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/professions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    first, second = body["professions"]
    assert first == {
        "name": "data_engineer",
        "domains": ["Данные"],
        "competency_codes": ["IT-1"],
        "hh_queries": ["инженер данных"],
        "aliases": ["de"],
    }
    # у analyst словарь инфо без трёх ключей -> значения по умолчанию []
    assert second["name"] == "analyst"
    assert second["domains"] == ["Аналитика"]
    assert second["competency_codes"] == []
    assert second["hh_queries"] == []
    assert second["aliases"] == []


def test_professions_init_error_500(monkeypatch):
    """Файловая таксономия не поднялась - услуга отвечает 500, а не падает."""

    def _broken_factory():
        raise RuntimeError("init failed")

    client = _client(monkeypatch, _broken_factory)
    r = client.get("/api/taxonomy/professions")
    assert r.status_code == 500
    assert "Попробуйте позже" in r.json()["detail"]


# -------------------------------------------------- /taxonomy/profession/{name}


def test_profession_detail_truncates_lists(monkeypatch):
    fake = _FakeProfessionTaxonomy()
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/data_engineer")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "data_engineer"
    assert body["domains"] == ["Данные"]
    assert body["competency_codes"] == ["IT-1"]
    # граница: 105 навыков -> в ответе первые 100, счётчик - полный
    assert body["skill_count"] == 105
    assert len(body["skills"]) == 100
    assert body["skills"][0] == "skill_0"
    assert body["skills"][-1] == "skill_99"
    krm = body["krm_competencies"]["IT-1"]
    # компетенция KRM режется до 20 навыков, счётчик - полный
    assert krm["skill_count"] == 25
    assert len(krm["skills"]) == 20


def test_profession_detail_unknown_404(monkeypatch):
    fake = _FakeProfessionTaxonomy()
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/ghost")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_profession_detail_info_error_500(monkeypatch):
    fake = _FakeProfessionTaxonomy(fail_info=True)
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/data_engineer")
    assert r.status_code == 500
    assert "Попробуйте позже" in r.json()["detail"]


# ---------------------------------------- /taxonomy/profession/{name}/krm-coverage

_KRM_COVERAGE = {
    "IT-1": {
        "coverage": 0.5,
        "total_required": 4,
        "covered_skills": ["python"],
        "missing_skills": ["rust"],
    }
}


def test_krm_coverage_parses_user_skills(monkeypatch):
    fake = _FakeProfessionTaxonomy(coverage=_KRM_COVERAGE)
    client = _client(monkeypatch, lambda: fake)
    r = client.get(
        "/api/taxonomy/profession/data_engineer/krm-coverage",
        params={"skills": " python , rust ,, sql ,"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # строка навыков разбирается: strip + отбрасывание пустых элементов
    assert body["user_skills"] == ["python", "rust", "sql"]
    assert body["profession"] == "data_engineer"
    assert ("coverage", "data_engineer", ("python", "rust", "sql")) in fake.calls
    item = body["competency_coverage"]["IT-1"]
    assert item["coverage"] == 0.5
    assert item["total_required"] == 4
    assert item["covered_skills"] == ["python"]
    assert item["missing_skills"] == ["rust"]
    assert body["avg_coverage"] == 0.5


def test_krm_coverage_without_skills_param(monkeypatch):
    """Граница: skills не передан - считаем покрытие по пустому набору."""
    fake = _FakeProfessionTaxonomy(coverage=_KRM_COVERAGE)
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/data_engineer/krm-coverage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_skills"] == []
    assert ("coverage", "data_engineer", ()) in fake.calls


def test_krm_coverage_no_data_404(monkeypatch):
    fake = _FakeProfessionTaxonomy(coverage={})
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/data_engineer/krm-coverage")
    assert r.status_code == 404
    assert "Нет KRM-данных" in r.json()["detail"]


def test_krm_coverage_compute_error_500(monkeypatch):
    fake = _FakeProfessionTaxonomy(fail_coverage=True)
    client = _client(monkeypatch, lambda: fake)
    r = client.get("/api/taxonomy/profession/data_engineer/krm-coverage")
    assert r.status_code == 500
    assert "Попробуйте позже" in r.json()["detail"]
