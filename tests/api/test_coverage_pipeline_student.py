"""Покрытие незакрытых эндпоинтов пайплайна и действий студента.

Без тестов на момент написания остались:

- GET /api/pipeline/status - агрегированный статус (БД + файлы артефактов);
- POST /api/pipeline/refresh-cache - разрушительная чистка кэша (rmtree!);
- POST /api/student/log-action - журнал действий студента (audit-трейл);
- GET /api/student/history - история действий студента.

Подменяется ТОЛЬКО get_current_user для гейтов require_any_role (сами гейты
живые: 401/403 тесты это доказывают). Студенческие ручки ходят с настоящим
подписанным токеном (auth._make_token), middleware-аутентификация не
подменяется. Доступ к данным мокается точечно:

- refresh-cache: BASE_DIR подменён на tmp_path - реальный data/ проекта
  физически не может быть удалён (rmtree работает только внутри tmp);
- статус: src.db.get_pool -> фейковый asyncpg-пул (как в test_security_paths);
- student: _get_pool -> None (записи остаются в in-memory буфере), буфер
  восстанавливается после теста, БД не затрагивается.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.pipeline as pl
import src.api_pkg.routers.student as student_mod
import src.api_pkg.student_actions as actions_mod
import src.db as db_mod
from src.api_pkg import create_app

STUDENT_EMAIL = "s1@uni.example"

_AUTH_POOL_ROW = {
    "completed_at": datetime(2026, 10, 31, 12, 0, tzinfo=UTC),
    "stats": {"vacancies": 1337, "stage": "done"},
}


class _Pool:
    """asyncpg-подобный пул для get_pipeline_status."""

    async def fetchrow(self, *_a, **_k):
        return dict(_AUTH_POOL_ROW)

    async def fetchval(self, *_a, **_k):
        return 4242


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Модульные лимитеры - общие для всех create_app() в процессе."""
    for mod in (pl, student_mod):
        with contextlib.suppress(Exception):
            mod.limiter._storage.reset()
    yield


def _client(monkeypatch, role="teacher"):
    user = None if role is None else {
        "uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None,
    }

    async def _fake_user(_request):
        return user

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    return TestClient(create_app(), raise_server_exceptions=False)


def _seed_cache(root):
    """Имитация data/cache: 2 справочника эмбеддинг/кластеров + один файл."""
    emb = root / "data" / "cache" / "embeddings"
    clusters = root / "data" / "cache" / "clusters"
    emb.mkdir(parents=True)
    clusters.mkdir()
    (emb / "a.joblib").write_bytes(b"x")
    (clusters / "b.joblib").write_bytes(b"x")
    parsed = root / "data" / "cache" / "parsed_skills.joblib"
    parsed.write_bytes(b"x")
    return emb, clusters, parsed


# ----------------------------------------------------- pipeline/status

class TestPipelineStatus:
    def test_teacher_gets_status(self, monkeypatch):
        monkeypatch.setattr(db_mod, "get_pool", lambda: _Pool())
        client = _client(monkeypatch, role="teacher")
        r = client.get("/api/pipeline/status")
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {
            "clusters", "clusters_all_ready", "ltr_model", "recommendations",
            "recommendations_all_ready", "skill_weights", "last_update",
            "total_vacancies", "last_pipeline_stats",
        }
        assert body["last_update"] == "2026-10-31 12:00:00+00:00"
        assert body["total_vacancies"] == 4242
        assert body["last_pipeline_stats"] == {"vacancies": 1337, "stage": "done"}
        assert body["clusters_all_ready"] == all(body["clusters"].values())
        assert body["recommendations_all_ready"] == all(body["recommendations"].values())

    def test_requires_token(self, monkeypatch):
        monkeypatch.setattr(db_mod, "get_pool", lambda: _Pool())
        client = _client(monkeypatch, role=None)
        assert client.get("/api/pipeline/status").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        monkeypatch.setattr(db_mod, "get_pool", lambda: _Pool())
        client = _client(monkeypatch, role="student")
        assert client.get("/api/pipeline/status").status_code == 403


# ----------------------------------------------------- pipeline/refresh-cache

class TestRefreshCache:
    def test_removes_caches_and_recreates_dirs(self, monkeypatch, tmp_path):
        emb, clusters, parsed = _seed_cache(tmp_path)
        monkeypatch.setattr(pl, "BASE_DIR", tmp_path)
        client = _client(monkeypatch, role="teacher")

        r = client.post("/api/pipeline/refresh-cache")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        assert set(body["removed"]) == {str(emb), str(clusters), str(parsed)}
        # каталоги пересоздаются пустыми, одиночный файл-кэш удаляется
        assert emb.is_dir() and clusters.is_dir()
        assert not any(emb.iterdir())
        assert not parsed.exists()

    def test_second_call_is_idempotent(self, monkeypatch, tmp_path):
        """Повторная чистка не падает и не удаляет ничего лишнего."""
        emb, clusters, _parsed = _seed_cache(tmp_path)
        monkeypatch.setattr(pl, "BASE_DIR", tmp_path)
        client = _client(monkeypatch, role="teacher")
        assert client.post("/api/pipeline/refresh-cache").status_code == 200
        r2 = client.post("/api/pipeline/refresh-cache")
        assert r2.status_code == 200, r2.text
        # каталоги пересозданы первой чисткой и снова удаляются; joblib уже нет
        assert set(r2.json()["removed"]) == {str(emb), str(clusters)}

    def test_requires_token(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pl, "BASE_DIR", tmp_path)
        client = _client(monkeypatch, role=None)
        assert client.post("/api/pipeline/refresh-cache").status_code == 401

    def test_student_forbidden(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pl, "BASE_DIR", tmp_path)
        client = _client(monkeypatch, role="student")
        assert client.post("/api/pipeline/refresh-cache").status_code == 403


# ----------------------------------------------------- student log-action

class TestStudentLogAction:
    """Аутентификация - настоящий подписанный токен (гейт не подменяется)."""

    def test_logged_action_recorded(self, monkeypatch):
        token = auth_mod._make_token("s-1", STUDENT_EMAIL, "student")
        client = TestClient(create_app(), raise_server_exceptions=False)
        r = client.post(
            "/api/student/log-action",
            headers={"Authorization": f"Bearer {token}"},
            json={"action_type": "vacancy_search", "profession": "python-разработчик",
                  "region": "Москва", "vacancies_found": 5,
                  "result_ref": "r-1", "profile": "base"},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok"}
        entry = actions_mod._action_buffer[-1]
        assert entry["username"] == STUDENT_EMAIL
        assert entry["action_type"] == "vacancy_search"
        assert entry["vacancies_found"] == 5
        assert entry["profile"] == "base"
        assert entry["timestamp"]

    def test_requires_valid_token(self, monkeypatch):
        client = TestClient(create_app(), raise_server_exceptions=False)
        # без токена и с подделанной подписью middleware не распознаёт пользователя
        assert client.post(
            "/api/student/log-action",
            json={"action_type": "vacancy_search"},
        ).status_code == 401
        tampered = auth_mod._make_token("s-1", STUDENT_EMAIL, "student")
        tampered = tampered.rsplit(".", 1)[0] + ".Zm9yZ2Vk"
        assert client.post(
            "/api/student/log-action",
            headers={"Authorization": f"Bearer {tampered}"},
            json={"action_type": "vacancy_search"},
        ).status_code == 401


# ----------------------------------------------------- student/history

class TestStudentHistory:
    def test_student_sees_only_own_entries(self, monkeypatch):
        actions_mod._action_buffer.clear()
        actions_mod._action_buffer.append({
            "username": STUDENT_EMAIL, "action_type": "search",
            "profession": "", "region": "", "vacancies_found": 1,
            "result_ref": "", "profile": "base", "timestamp": "2026-11-01T00:00:00+00:00",
        })
        actions_mod._action_buffer.append({
            "username": "another@uni.example", "action_type": "search",
            "profession": "", "region": "", "vacancies_found": 2,
            "result_ref": "", "profile": "dc", "timestamp": "2026-11-02T00:00:00+00:00",
        })
        token = auth_mod._make_token("s-1", STUDENT_EMAIL, "student")
        client = TestClient(create_app(), raise_server_exceptions=False)
        r = client.get("/api/student/history", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 1, "чужие записи видны быть не должны"
        assert body["history"][0]["result_ref"] == ""
        assert body["history"][0]["username"] == STUDENT_EMAIL

    def test_limit_keeps_most_recent(self, monkeypatch):
        actions_mod._action_buffer.clear()
        for i in range(3):
            actions_mod._action_buffer.append({
                "username": STUDENT_EMAIL, "action_type": f"act-{i}",
                "profession": "", "region": "", "vacancies_found": i,
                "result_ref": f"r-{i}", "profile": "base",
                "timestamp": f"2026-11-0{i + 1}T00:00:00+00:00",
            })
        token = auth_mod._make_token("s-1", STUDENT_EMAIL, "student")
        client = TestClient(create_app(), raise_server_exceptions=False)
        r = client.get("/api/student/history?limit=1",
                       headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 1
        assert body["history"][0]["action_type"] == "act-2"

    def test_requires_valid_token(self):
        client = TestClient(create_app(), raise_server_exceptions=False)
        assert client.get("/api/student/history").status_code == 401


@pytest.fixture(autouse=True)
def _restore_buffer():
    """Буфер действий - общий ресурс процесса: не мусорим между тестами."""
    saved = list(actions_mod._action_buffer)
    actions_mod._action_buffer.clear()
    yield
    actions_mod._action_buffer.clear()
    actions_mod._action_buffer.extend(saved)


@pytest.fixture(autouse=True)
def _no_db_pool(monkeypatch):
    """История炫действий обязана выше без БД: пул подменён на None."""
    monkeypatch.setattr(actions_mod, "_get_pool", AsyncMock(return_value=None))
    yield
