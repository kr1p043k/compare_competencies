"""Барьеры карточки дисциплины KRM: таймаут БД -> 504, нет пула -> 503.

БД не нужна: пул подменяется фейком. Раньше TimeoutError давал 500-стектрейс.
"""

from __future__ import annotations

import contextlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.teacher as teacher_mod
from tests.conftest import open_all_gates


class _Pool:
    def __init__(self, *, disc=None, comps=None, raise_timeout: bool = False):
        self.disc = disc
        self.comps = comps or []
        self.raise_timeout = raise_timeout

    async def fetchrow(self, _sql: str, *args):
        if self.raise_timeout:
            raise TimeoutError("query cancelled due to statement timeout")
        return self.disc

    async def fetch(self, _sql: str, *args):
        if self.raise_timeout:
            raise TimeoutError("query cancelled due to statement timeout")
        return self.comps


def _client(monkeypatch, pool) -> TestClient:
    monkeypatch.setattr(teacher_mod, "get_pool", lambda: pool)
    with contextlib.suppress(Exception):
        teacher_mod.limiter._storage.reset()
    app = FastAPI()
    app.state.limiter = teacher_mod.limiter
    app.include_router(teacher_mod.router, prefix="/api")
    return TestClient(open_all_gates(app), raise_server_exceptions=False)


URL = "/api/teacher/krm/disciplines/Математика"


def test_timeout_returns_504_not_500(monkeypatch):
    client = _client(monkeypatch, _Pool(raise_timeout=True))
    r = client.get(URL)
    assert r.status_code == 504, r.text
    assert "долго" in r.json()["detail"]


def test_missing_pool_returns_503(monkeypatch):
    client = _client(monkeypatch, None)
    r = client.get(URL)
    assert r.status_code == 503, r.text


def test_not_found_discipline_returns_404(monkeypatch):
    client = _client(monkeypatch, _Pool(disc=None))
    r = client.get(URL)
    assert r.status_code == 404, r.text


def test_happy_path_returns_competencies(monkeypatch):
    disc = {"id": "00000000-0000-0000-0000-000000000001", "name": "Математика"}
    comps = [
        {
            "id": "00000000-0000-0000-0000-0000000000aa",
            "code": "УК-1",
            "skills": ["анализ"],
            "knowledge_ids": ["k1"],
            "knowledge": ["знать"],
            "abilities_ids": ["a1"],
            "abilities": ["уметь"],
            "prof_skills_ids": ["s1"],
            "prof_skills": ["владеть"],
        }
    ]
    client = _client(monkeypatch, _Pool(disc=disc, comps=comps))
    r = client.get(URL)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "Математика"
    assert body["year"] == 2025
    assert body["competencies"][0]["code"] == "УК-1"
    assert body["competencies"][0]["ksa"]["knowledge"] == [{"id": "k1", "text": "знать"}]
