"""Покрытие роутера vacancies: список, карточка, статистика, аналитика.

БД подменяем фейковым пулом: запросы шлём в реальные SQL-строки, но данные
возвращаем каной. Так проверяем разбор строк, фильтры и форму ответа.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.vacancies as vac


def _dt(y=2026, m=1, d=1):
    return datetime(y, m, d, tzinfo=UTC)


class FakePool:
    """Возвращает данные по подстрокам в SQL; пишет, что запрашивали."""

    def __init__(self, *, rows=None, one=None, total=0, scalar=0):
        self._rows = rows or []
        self._one = one
        self._total = total
        self._scalar = scalar
        self.calls: list[str] = []

    async def fetchval(self, sql, *a):
        self.calls.append("fetchval " + " ".join(sql.split())[:60])
        low = sql.lower()
        if "count(*)" in low and "parsed_skills" in low:
            return 3
        if "jsonb_array_length" in low:
            return 5
        if "count(*)" in low:
            return self._total
        return self._scalar

    async def fetchrow(self, sql, *a):
        self.calls.append("fetchrow " + " ".join(sql.split())[:60])
        return self._one

    async def fetch(self, sql, *a):
        self.calls.append("fetch " + " ".join(sql.split())[:60])
        if "group by exp_group" in " ".join(sql.split()).lower():
            return [{"exp_group": "middle", "cnt": 4}, {"exp_group": "junior", "cnt": 2}]
        if "group by lvl" in " ".join(sql.split()).lower():
            return [{"lvl": "middle", "cnt": 3, "avg_sal": 120000}]
        return self._rows


@pytest.fixture()
def client():
    from src.api_pkg import create_app

    app = create_app()
    for route in app.routes:
        dep = getattr(route, "dependant", None)
        for sub in (getattr(dep, "dependencies", None) or []):
            app.dependency_overrides[sub.call] = lambda: {"r": "admin"}
    for lim in (getattr(app.state, "limiter", None), getattr(vac, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app)


def _use_pool(monkeypatch, pool):
    async def _get():
        return pool

    monkeypatch.setattr(vac, "_get_db_pool", _get)


# ---------- чистые хелперы ----------


def test_clean_skill_list_handles_dicts_and_strings():
    out = vac._clean_skill_list([{"name": "Python"}, {"name": " python "}, "SQL", "", None, "sql"])
    assert out == [{"name": "Python"}, "SQL"]


def test_clean_skill_list_empty():
    assert vac._clean_skill_list(None) == []


def test_parse_day_valid_and_invalid():
    assert vac._parse_day(None) is None
    assert vac._parse_day("  ") is None
    assert vac._parse_day("2026-13-01") is None
    dt = vac._parse_day("2026-05-01")
    assert dt is not None and dt.year == 2026 and dt.month == 5


def test_classify_experience_variants():
    assert vac._classify_experience("noExperience", "") == "junior"
    assert vac._classify_experience("moreThan6", "") == "senior"
    assert vac._classify_experience("between1And3", "") == "middle"
    assert vac._classify_experience(None, "Junior разработчик") == "junior"
    # Имя с «ведущий» относится к senior-группе (см. список в _classify_experience).
    assert vac._classify_experience(None, "Ведущий инженер") == "senior"


# ---------- /vacancies ----------


def test_list_vacancies_success(client, monkeypatch):
    row = {
        "hh_id": 101,
        "name": "Python dev",
        "experience": "between1And3",
        "salary_from": 100000,
        "salary_to": 150000,
        "salary_currency": "RUR",
        "employer_name": "ООО",
        "area_name": "Москва",
        "snippet_requirement": "python",
        "snippet_responsibility": "",
        "published_at": _dt(),
        "alternate_url": "https://hh.ru/101",
        "parsed_skills": json.dumps(["python", "sql"]),
        "key_skills": [],
        "employer_logo": None,
    }
    _use_pool(monkeypatch, FakePool(rows=[row], total=1))
    r = client.get("/api/vacancies")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["has_more"] is False
    item = body["items"][0]
    assert item["id"] == "101"
    assert item["experience"] == "middle"
    assert item["skills"] == ["python", "sql"]
    assert item["snippet"]["requirement"] == "python"


def test_list_vacancies_bad_date_400(client, monkeypatch):
    _use_pool(monkeypatch, FakePool())
    r = client.get("/api/vacancies?date_from=01-01-2026")
    assert r.status_code == 400


def test_list_vacancies_503_without_pool(client, monkeypatch):
    _use_pool(monkeypatch, None)
    r = client.get("/api/vacancies")
    assert r.status_code == 503


# ---------- /vacancies/info ----------


def test_info_without_pool(client, monkeypatch):
    _use_pool(monkeypatch, None)
    r = client.get("/api/vacancies/info")
    assert r.status_code == 200
    assert r.json()["db_available"] is False


def test_info_with_pool(client, monkeypatch):
    pool = FakePool(total=10, one={"min_p": _dt(2026, 1, 1), "max_p": _dt(2026, 2, 1),
                                   "completed_at": _dt(), "stats": {"ok": True}})
    _use_pool(monkeypatch, pool)
    r = client.get("/api/vacancies/info")
    assert r.status_code == 200
    body = r.json()
    assert body["db_available"] is True
    assert body["date_range"]["from"] == "2026-01-01"


# ---------- /vacancies/{id} ----------


def test_detail_success(client, monkeypatch):
    row = {
        "hh_id": 5, "name": "Dev", "description": "text", "experience": "noExperience",
        "salary_from": 1, "salary_to": 2, "salary_currency": "RUR",
        "employer_name": "ООО", "employer_id": "e1", "area_name": "Москва",
        "snippet_requirement": "r", "snippet_responsibility": "",
        "published_at": _dt(), "alternate_url": "u", "parsed_skills": '["python"]',
        "key_skills": '["sql"]', "employer_logo": None,
    }
    _use_pool(monkeypatch, FakePool(one=row))
    r = client.get("/api/vacancies/5")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "5"
    assert body["skills"] == ["python"]
    assert body["key_skills"] == ["sql"]


def test_detail_bad_id_400(client, monkeypatch):
    _use_pool(monkeypatch, FakePool())
    r = client.get("/api/vacancies/not-a-number")
    assert r.status_code == 400


def test_detail_not_found_404(client, monkeypatch):
    _use_pool(monkeypatch, FakePool(one=None))
    r = client.get("/api/vacancies/999")
    assert r.status_code == 404


# ---------- stats ----------


def test_stats_summary(client, monkeypatch):
    pool = FakePool(total=10, one={"avg_sal": 123456.7, "min_sal": 100, "max_sal": 900, "cnt": 5})
    _use_pool(monkeypatch, pool)
    r = client.get("/api/vacancies/stats/summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 10
    assert body["by_experience"]["middle"] == 4
    assert body["salary"]["average"] == 123457.0


def test_stats_summary_503(client, monkeypatch):
    _use_pool(monkeypatch, None)
    r = client.get("/api/vacancies/stats/summary")
    assert r.status_code == 503


def test_stats_analytics(client, monkeypatch):
    pool = FakePool(total=10, rows=[{"name": "Москва", "cnt": 5}])
    _use_pool(monkeypatch, pool)
    r = client.get("/api/vacancies/stats/analytics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["salary_by_level"]["middle"]["count"] == 3
    assert body["skills"]["total"] == 10
    assert body["skills"]["percent"] == 30.0


def test_stats_analytics_503(client, monkeypatch):
    _use_pool(monkeypatch, None)
    r = client.get("/api/vacancies/stats/analytics")
    assert r.status_code == 503
