"""Покрытие роутера taxonomy: покрытие, категории.

Таксономию подменяем через dependency_overrides: Depends хранит ссылку на
функцию-провайдер, поэтому monkeypatch модуля не сработал бы.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import src.api_pkg.routers.taxonomy as tax
from src import Err, Ok


class FakeTaxonomy:
    """Управляемая таксономия: категории -> навыки."""

    def __init__(self, data=None, fail_categories=False, fail_skills_for=()):
        self._data = data or {
            "backend": {"label": "Бэкенд", "icon": "🔧", "skills": ["python", "sql"]},
            "empty_cat": {"label": "Пустая", "icon": "📦", "skills": []},
        }
        self._fail_categories = fail_categories
        self._fail_skills_for = set(fail_skills_for)

    def get_all_categories(self):
        if self._fail_categories:
            return Err("boom")
        return Ok(list(self._data))

    def get_skills_in_category(self, cat_id):
        if cat_id in self._fail_skills_for:
            return Err("boom")
        return Ok(list(self._data.get(cat_id, {}).get("skills", [])))

    def get_category_label_by_id(self, cat_id):
        return self._data.get(cat_id, {}).get("label", cat_id)

    def get_category_icon_by_id(self, cat_id):
        return self._data.get(cat_id, {}).get("icon", "")


def _client(monkeypatch, taxonomy, *, skills=None):
    from src.api_pkg import create_app
    from tests.conftest import open_all_gates

    app = create_app()
    open_all_gates(app, role="admin")
    if taxonomy is not None:
        app.dependency_overrides[tax.deps.get_taxonomy] = lambda: taxonomy
    if skills is not None:
        monkeypatch.setattr(tax.deps, "current_skills_set", skills, raising=False)
    for lim in (getattr(app.state, "limiter", None), getattr(tax, "limiter", None)):
        if lim is not None:
            lim.reset()
    return TestClient(app)


# ---------- /taxonomy/coverage ----------


def test_coverage_no_taxonomy_503(monkeypatch):
    from src.api_pkg import create_app
    from tests.conftest import open_all_gates

    app = create_app()
    open_all_gates(app, role="admin")
    app.dependency_overrides[tax.deps.get_taxonomy] = lambda: None
    r = TestClient(app).get("/api/taxonomy/coverage")
    assert r.status_code == 503


def test_coverage_counts_covered(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy(), skills={"python"})
    r = c.get("/api/taxonomy/coverage")
    assert r.status_code == 200, r.text
    cov = r.json()["coverage"]
    assert cov["backend"]["total"] == 2
    assert cov["backend"]["covered"] == 1
    assert cov["backend"]["percent"] == 50.0
    # Пустая категория не попадает в ответ.
    assert "empty_cat" not in cov


def test_coverage_categories_error_500(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy(fail_categories=True), skills={"python"})
    assert c.get("/api/taxonomy/coverage").status_code == 500


def test_coverage_skills_error_500(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy(fail_skills_for=["backend"]), skills={"python"})
    assert c.get("/api/taxonomy/coverage").status_code == 500


# ---------- /taxonomy/categories ----------


def test_categories_lists_non_empty(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy())
    r = c.get("/api/taxonomy/categories")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["categories"][0]["id"] == "backend"
    assert body["categories"][0]["skills"] == ["python", "sql"]


def test_categories_503_without_taxonomy(monkeypatch):
    c = _client(monkeypatch, None)
    assert c.get("/api/taxonomy/categories").status_code == 503


def test_categories_error_500(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy(fail_categories=True))
    assert c.get("/api/taxonomy/categories").status_code == 500


def test_categories_skips_broken_skill_list(monkeypatch):
    c = _client(monkeypatch, FakeTaxonomy(fail_skills_for=["backend"]))
    r = c.get("/api/taxonomy/categories")
    assert r.status_code == 200
    assert r.json()["total"] == 0
