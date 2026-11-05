"""Coverage tests for src/api_pkg/routers/trends.py.

Authenticaiton bypassed with a stubbed current-user; dependencies monkeypatched.
"""

from datetime import date
from types import SimpleNamespace

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.trends as trends_mod
from src import Ok
from src.api_pkg import create_app
from src.models.enums import TrendType


class _FakeTrendAnalyzer:
    def get_trending_skills(self, top_n=15, min_change_percent=3.0, **kwargs):
        return Ok({
            TrendType.RISING: [{"skill": "python", "change_pct": 10.0}],
            TrendType.FALLING: [{"skill": "cobol", "change_pct": -12.0}],
            TrendType.EMERGING: [{"skill": "rust", "current_freq": 100}],
        })


async def _fake_trend_analyzer():
    return _FakeTrendAnalyzer()


class _FakeAnalyzerEmpty:
    def get_trending_skills(self, *args, **kwargs):
        return Ok({
            TrendType.RISING: [],
            TrendType.FALLING: [],
            TrendType.EMERGING: [],
        })


async def _fake_trend_analyzer_empty():
    return _FakeAnalyzerEmpty()


async def _fake_current_user(_request):
    return {"u": "t@t.com", "r": "teacher", "uid": "u-1"}


def _make_client(monkeypatch, *, analyzer_factory=_fake_trend_analyzer):
    app = create_app()
    monkeypatch.setattr(auth_mod, "get_current_user", _fake_current_user)
    app.dependency_overrides[trends_mod.deps.get_trend_analyzer] = analyzer_factory
    monkeypatch.setattr(trends_mod, "_get_mappings", lambda: ({}, {}))
    monkeypatch.setattr(
        trends_mod, "skill_words", lambda name, *_args, **_kwargs: name.lower().split()
    )
    trends_mod._clear_trend_caches()
    return TestClient(app, raise_server_exceptions=False)


def _build_fake_session(
    *, with_skill=True,
    snapshots=None,
    comp_skills=None,
):
    _snapshots = snapshots or []
    _comp_skills = comp_skills or []

    class _FakeResult:
        def __init__(self, rows=None):
            self._rows = rows or []

        def scalars(self):
            return self

        def all(self):
            return self._rows

        def __iter__(self):
            yield from self._rows

    class _FakeSession:
        def __init__(self):
            self._calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def execute(self, stmt):
            if "TrendSnapshot" in str(stmt) or "trend_snapshots" in str(stmt).lower():
                return _FakeResult(_snapshots)
            return _FakeResult(_comp_skills)

        async def get(self, *_args, **_kwargs):
            return None

    return _FakeSession


# ----------------------------------------------------------------------------- /trends

def test_get_trends_success(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/api/trends?top_n=5&min_change=4.0")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "trends" in body
    assert body["trends"]["rising"][0]["skill"] == "python"


def test_get_trends_empty(monkeypatch):
    client = _make_client(monkeypatch, analyzer_factory=_fake_trend_analyzer_empty)
    r = client.get("/api/trends")
    assert r.status_code == 200, r.text
    assert r.json()["trends"]["rising"] == []


def test_get_trends_analyzer_exception_returns_500(monkeypatch):
    async def _broken():
        class _Broken:
            def get_trending_skills(self, *args, **kwargs):
                raise RuntimeError("boom")
        return _Broken()
    client = _make_client(monkeypatch, analyzer_factory=_broken)
    r = client.get("/api/trends")
    assert r.status_code == 500, r.text


def test_get_trends_validation_top_n_not_int(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/api/trends?top_n=abc")
    assert r.status_code == 422, r.text


# ----------------------------------------------------------------------------- /competency-trends

def test_get_competency_trends_success(monkeypatch):
    snap1 = SimpleNamespace(
        snapshot_date=date(2026, 10, 1),
        skill_freq={"python": 100, "sql": 50},
    )
    snap2 = SimpleNamespace(
        snapshot_date=date(2026, 11, 1),
        skill_freq={"python": 150, "sql": 60},
    )
    comp_rows = [
        SimpleNamespace(id="c1", code="01.03.01", name="Math", skill_name="Python"),
    ]
    monkeypatch.setattr(
        trends_mod, "async_session_factory",
        _build_fake_session(snapshots=[snap2, snap1], comp_skills=comp_rows),
    )
    client = _make_client(monkeypatch)
    r = client.get("/api/competency-trends?direction=rising&limit=10")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] >= 0
    assert body["trends"] == [] or all(t.get("direction") == "rising" for t in body["trends"])


def test_get_competency_trends_cache_hit(monkeypatch):
    snap = SimpleNamespace(snapshot_date=date(2026, 11, 1), skill_freq={"python": 1})
    comp_rows = [
        SimpleNamespace(id="c1", code="01.03.01", name="Math", skill_name="Python"),
    ]
    monkeypatch.setattr(
        trends_mod, "async_session_factory",
        _build_fake_session(snapshots=[snap], comp_skills=comp_rows),
    )
    client = _make_client(monkeypatch)
    # bypass <2 snapshots by cache pre-population
    trends_mod._CT_CACHE[("falling", 10)] = (1e20, {"total": 0, "trends": []})
    r = client.get("/api/competency-trends?direction=falling&limit=10")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 0


def test_get_competency_trends_not_enough_snapshots(monkeypatch):
    snap = SimpleNamespace(snapshot_date=date(2026, 11, 1), skill_freq={"python": 1})
    comp_rows = [SimpleNamespace(id="c1", code="01.03.01", name="Math", skill_name="Python")]
    monkeypatch.setattr(
        trends_mod, "async_session_factory",
        _build_fake_session(snapshots=[snap], comp_skills=comp_rows),
    )
    client = _make_client(monkeypatch)
    r = client.get("/api/competency-trends")
    assert r.status_code == 400, r.text


def test_get_competency_trends_invalid_direction(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/api/competency-trends?direction=up")
    assert r.status_code == 422, r.text


def test_get_competency_trends_db_error_returns_500(monkeypatch):
    class _BrokenSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *_exc): return False
        async def execute(self, _stmt): raise RuntimeError("db down")
        async def get(self, *_a, **_kw): return None
    monkeypatch.setattr(trends_mod, "async_session_factory", _BrokenSession)
    client = _make_client(monkeypatch)
    r = client.get("/api/competency-trends")
    assert r.status_code == 500, r.text
