"""Покрытие незакрытых GET-эндпоинтов учителя: рыночные и служебные данные.

Идеальный smoke не должен ловить 500 только на живом прогоне. Четыре ручки
/teacher/krm/* не имели ни одного теста на момент написания:

- GET /api/teacher/krm/market-skills - рынок навыков (it_skills.json);
- GET /api/teacher/krm/stats - счётчики направления из БД;
- GET /api/teacher/krm/search-runs - история запусков пайплайна (SQLAlchemy);
- GET /api/teacher/krm/search-runs/{run_id} - детали запуска и анализа.

Подменяется ТОЛЬКО get_current_user (гейт require_any_role остаётся живым:
без него тесты ниже потеряют смысл) и доступ к данным (фейковый пул/сессия по
образцу test_pipeline_ownership.py / test_subscriptions.py). Реальная БД и
файлы проекта не затрагиваются.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.teacher as teacher_mod
import src.database as database_mod
from src.api_pkg import create_app


class _Pool:
    """asyncpg-подобный пул: только fetchrow с фиксированной строкой."""

    def __init__(self, row):
        self._row = row

    async def fetchrow(self, *_args, **_kwargs):
        return self._row


class _FakeResult:
    """SQLAlchemy-подобный Result: scalars().all() и scalars().first()."""

    # Фоновый notifier читает rowcount; без атрибута после ответа
    # всплывает AttributeError в логе задачи.
    rowcount = 0

    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Минимальная async-сессия: execute/get/commit (как в test_subscriptions)."""

    def __init__(self, *, rows=None, get_obj=None):
        self.rows = rows if rows is not None else []
        self.get_obj = get_obj
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, *_a, **_k):
        return _FakeResult(self.rows)

    async def get(self, _model, _pk):
        return self.get_obj

    async def commit(self):
        self.committed = True


_UNSET = object()


def _run(id="run-1", action="full-cycle", status="completed",
         started_at=None, completed_at=_UNSET, stats=None):
    """Строка pipeline_runs для фейковой сессии.

    completed_at по умолчанию - фиксированное время, но явный None сохраняется:
    иначе нельзя проверить незавершённый прогон.
    """
    return {
        "id": id,
        "action": action,
        "status": status,
        "started_at": started_at or datetime(2026, 11, 1, 10, 0, tzinfo=UTC),
        "completed_at": (
            datetime(2026, 11, 1, 11, 30, tzinfo=UTC)
            if completed_at is _UNSET
            else completed_at
        ),
        "stats": stats if stats is not None else {"vacancies": 10},
    }


class _RunRow:
    """Объект ORM-строки (доступ по атрибутам, не по ключам)."""

    def __init__(self, **kw):
        self.__dict__.update(_run(**kw))


class _AnalysisRow:
    def __init__(self, data):
        self.data = data


@pytest.fixture(autouse=True)
def _reset_limiter():
    with contextlib.suppress(Exception):
        teacher_mod.limiter._storage.reset()
    yield


def _client(monkeypatch, role="teacher", *, pool=None, session=None):
    """Живое приложение + подмена только get_current_user (гейт остаётся).

    pool -> teacher_mod.get_pool (asyncpg), session -> src.database.async_session_factory.
    """
    user = None if role is None else {
        "uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None,
    }

    async def _fake_user(_request):
        return user

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    if pool is not None:
        monkeypatch.setattr(teacher_mod, "get_pool", lambda: pool)
    if session is not None:
        monkeypatch.setattr(database_mod, "async_session_factory", lambda: session)
    return TestClient(create_app(), raise_server_exceptions=False)


# ----------------------------------------------------- market-skills

class TestMarketSkills:
    def test_teacher_gets_skills_list(self, monkeypatch):
        client = _client(monkeypatch, role="teacher")
        r = client.get("/api/teacher/krm/market-skills?limit=3")
        assert r.status_code == 200, r.text
        body = r.json()
        assert 0 < len(body) <= 3, "limit=3 должен резать выдачу"
        assert set(body[0]) == {"skill", "frequency"}
        assert isinstance(body[0]["frequency"], int)

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/teacher/krm/market-skills").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        """Гейт admin/teacher/rop: student получает 403, а не данные."""
        client = _client(monkeypatch, role="student")
        assert client.get("/api/teacher/krm/market-skills").status_code == 403


# ----------------------------------------------------- krm/stats

class TestKrmStats:
    _ROW = {
        "total_disciplines": 3,
        "total_competencies": 12,
        "total_skills": 47,
    }

    def test_teacher_gets_direction_counts(self, monkeypatch):
        client = _client(monkeypatch, role="teacher", pool=_Pool(self._ROW))
        r = client.get("/api/teacher/krm/stats?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        assert r.json() == {
            "dir_code": "09.03.02",
            "total_disciplines": 3,
            "total_competencies": 12,
            "total_skills": 47,
        }

    def test_unknown_direction_is_404(self, monkeypatch):
        client = _client(monkeypatch, role="teacher", pool=_Pool(None))
        r = client.get("/api/teacher/krm/stats?dir_code=00.00.00")
        assert r.status_code == 404, r.text
        assert "not found" in r.json()["detail"]

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None, pool=_Pool(self._ROW))
        assert client.get("/api/teacher/krm/stats").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        client = _client(monkeypatch, role="student", pool=_Pool(self._ROW))
        assert client.get("/api/teacher/krm/stats").status_code == 403


# ----------------------------------------------------- search-runs (список)

class TestSearchRunsList:
    def test_teacher_sees_runs_history(self, monkeypatch):
        session = _FakeSession(rows=[
            _RunRow(id="run-2", action="rebuild", status="running",
                    completed_at=None, stats={}),
            _RunRow(id="run-1"),
        ])
        client = _client(monkeypatch, role="teacher", session=session)
        r = client.get("/api/teacher/krm/search-runs?limit=5")
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body) == 2
        # Порядок строк задаёт БД (ORDER BY started_at DESC); фейковая
        # сессия отдаёт их как есть, эндпоинт повторно не сортирует.
        newest, older = body
        assert newest["id"] == "run-2" and newest["completed_at"] is None
        assert older["id"] == "run-1"
        assert older["action"] == "full-cycle"
        assert older["status"] == "completed"
        assert older["started_at"].startswith("2026-11-01T10:00:00")
        assert older["stats"] == {"vacancies": 10}

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/teacher/krm/search-runs").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        client = _client(monkeypatch, role="student")
        assert client.get("/api/teacher/krm/search-runs").status_code == 403


# ----------------------------------------------------- search-runs (детали)

class TestSearchRunDetail:
    def test_run_with_analysis(self, monkeypatch):
        session = _FakeSession(
            rows=[_AnalysisRow({"average_coverage": 0.62})],
            get_obj=_RunRow(),
        )
        client = _client(monkeypatch, role="teacher", session=session)
        r = client.get("/api/teacher/krm/search-runs/run-1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["run"]["id"] == "run-1"
        assert body["run"]["stats"] == {"vacancies": 10}
        assert body["analysis"] == {"average_coverage": 0.62}

    def test_run_without_analysis_returns_null(self, monkeypatch):
        session = _FakeSession(rows=[], get_obj=_RunRow())
        client = _client(monkeypatch, role="teacher", session=session)
        r = client.get("/api/teacher/krm/search-runs/run-1")
        assert r.status_code == 200, r.text
        assert r.json()["analysis"] is None

    def test_unknown_run_is_404(self, monkeypatch):
        client = _client(monkeypatch, role="teacher", session=_FakeSession(rows=[], get_obj=None))
        r = client.get("/api/teacher/krm/search-runs/nope")
        assert r.status_code == 404, r.text
        assert "not found" in r.json()["detail"].lower()

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/teacher/krm/search-runs/run-1").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        client = _client(monkeypatch, role="student")
        assert client.get("/api/teacher/krm/search-runs/run-1").status_code == 403
