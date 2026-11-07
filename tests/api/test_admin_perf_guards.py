"""Барьеры производительности админских эндпоинтов.

Проверяем то, что легко сломать рефакторингом: направления в /admin/users
грузятся только для текущей страницы, а чтение каталога студентов уходит
в отдельный поток (asyncio.to_thread), не блокируя event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api_pkg.routers.admin as admin_mod
import src.api_pkg.routers.auth as auth_mod
import src.config as config
import src.database as database_mod


def _user(i: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"00000000-0000-0000-0000-{i:012d}",
        email=f"user{i}@example.com",
        role="student",
        full_name=f"User {i}",
        is_active=True,
    )


class _FakeResult:
    def __init__(self, rows: list, *, scalar: object = None):
        self._rows = rows
        self._scalar = scalar

    def fetchall(self) -> list:
        return self._rows

    def all(self) -> list:
        return self._rows

    def scalar_one(self):
        return self._scalar


class _RecordingSession:
    """Фейковая сессия, которая запоминает текст выполненных запросов."""

    def __init__(self, users: list, directions: list[tuple[str, str]] | None = None):
        self._users = users
        self._directions = directions or []
        self.queries: list[str] = []

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, stmt, params: dict | None = None):
        try:
            # literal_binds: иначе SQLAlchemy оставляет маркер POSTCOMPILE_user_id_1
            # и в тексте запроса не видно, какие id попали в IN (...).
            sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        except Exception:
            sql = str(stmt)
        self.queries.append(sql)
        if "user_directions" in sql:
            if "WHERE" in sql:
                # Отфильтрованный запрос: отдаём направления только запрошенных id.
                wanted = {str(u.id) for u in self._users}
                return _FakeResult([(k, v) for k, v in self._directions if str(k) in wanted])
            return _FakeResult(self._directions)
        if "COUNT(*)" in sql:
            return _FakeResult([], scalar=len(self._users))
        if "to_regclass" in sql:
            return _FakeResult([], scalar=None)
        if "FROM users" in sql:
            return _FakeResult(self._users)
        return _FakeResult([])


async def _fake_current_user(_request):
    return {"u": "admin@example.com", "r": "admin"}


def _client(monkeypatch, session) -> TestClient:
    monkeypatch.setattr(auth_mod, "get_current_user", _fake_current_user)
    monkeypatch.setattr(database_mod, "async_session_factory", lambda: session)
    monkeypatch.setattr(admin_mod, "get_logs_by_user", lambda: {})
    with contextlib.suppress(Exception):
        admin_mod.limiter._storage.reset()
    app = FastAPI()
    app.state.limiter = admin_mod.limiter
    app.include_router(admin_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def test_users_directions_query_is_bounded_by_page(monkeypatch):
    """Запрос направлений ограничен id текущей страницы, а не всей таблицей."""
    session = _RecordingSession(
        [_user(1), _user(2)],
        directions=[
            ("00000000-0000-0000-0000-000000000001", "09.03.02"),
            ("00000000-0000-0000-0000-000000000099", "01.03.02"),
        ],
    )
    client = _client(monkeypatch, session)
    r = client.get("/admin/users")
    assert r.status_code == 200, r.text

    dir_queries = [q for q in session.queries if "user_directions" in q]
    assert dir_queries, "направления вообще не запрашивались"
    sql = dir_queries[0]
    # Границы заданы: запрос содержит IN по id пользователей, а не всю таблицу.
    assert "WHERE" in sql
    assert "user_id IN" in sql

    body = r.json()
    assert body["total"] == 2
    assert body["users"][0]["directions"] == ["09.03.02"]
    # Направление постороннего пользователя (не из страницы) в ответ не попало.
    assert "01.03.02" not in [d for u in body["users"] for d in u["directions"]]


def test_users_directions_not_queried_for_empty_page(monkeypatch):
    """Пустая страница - запрос направлений не выполняется."""
    session = _RecordingSession([])
    client = _client(monkeypatch, session)
    r = client.get("/admin/users?q=nobody")
    assert r.status_code == 200, r.text
    assert not [q for q in session.queries if "user_directions" in q]


def test_students_scan_uses_thread(monkeypatch, tmp_path):
    """Сканирование каталога студентов выполняется через asyncio.to_thread."""
    students_dir = tmp_path / "students"
    students_dir.mkdir()
    (students_dir / "s1.json").write_text(
        json.dumps({"skills": ["a", "b"]}, ensure_ascii=False), encoding="utf-8"
    )

    calls = {"n": 0}
    real_to_thread = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        calls["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(admin_mod, "asyncio", asyncio)
    monkeypatch.setattr(asyncio, "to_thread", _spy)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    client = _client(monkeypatch, _RecordingSession([]))
    r = client.get("/admin/students")
    assert r.status_code == 200, r.text
    assert calls["n"] >= 1
    assert r.json()["students"][0]["filename"] == "s1.json"
