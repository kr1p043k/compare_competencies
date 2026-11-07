"""Покрытие незакрытых эндпоинтов KRM-доступа преподавателей и истории.

Ни один из них не имел тестов на момент написания:

- GET /api/krm/teacher/my-directions - направления пользователя (authz);
- GET /api/krm/teacher/check-access - проверка доступа к направлению;
- GET /api/krm/teacher/directions/{dir_code} - данные направления (authz);
- GET /api/teacher/history - история снимков покрытия (jsonl + delta).

Подменяется ТОЛЬКО get_current_user (authz-логика _user_directions /
_ensure_dir_access остаётся живой) и источники данных: пул -> None (в БД не
ходим), справочник направлений -> фиксированный список, каталог истории ->
tmp_path. Реальная БД и data/ проекта не затрагиваются.
"""

from __future__ import annotations

import contextlib
import json

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.history as history_mod
import src.api_pkg.routers.krm_teacher as krm_mod
from src.api_pkg import create_app

DIRS = [
    {"dir_code": "09.03.02", "name": "Информационные системы и технологии",
     "profile": "Разработка", "disciplines_count": 2},
    {"dir_code": "09.03.03", "name": "Прикладная информатика",
     "profile": "Аналитика", "disciplines_count": 1},
]

DIRECTION_DATA = {
    "direction_name": "Информационные системы и технологии",
    "profile": "Разработка",
    "disciplines": {"Базы данных": {"competencies": ["ОПК-1"], "skills": {"ОПК-1": ["SQL"]}}},
}


@pytest.fixture(autouse=True)
def _reset_limiter():
    with contextlib.suppress(Exception):
        krm_mod.limiter._storage.reset()
    yield


class _EmptyPool:
    """Пул без привязок: fetch возвращает пустой список."""

    async def fetch(self, *_a, **_kw):
        return []


def _client(monkeypatch, role="teacher", pool=None):
    """Живое приложение; authz-хелперы krm_teacher работают по-настоящему.

    pool по умолчанию None (в БД не ходим). Для проверки fallback-а teacher
    нужен пустой пул: при pool=None _user_directions возвращает [] до ветки
    fallback, потому что при недоступной БД доступ закрывается, а не
    расширяется.
    """
    user = None if role is None else {
        "uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None,
    }

    async def _fake_user(_request):
        return user

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    monkeypatch.setattr(krm_mod, "get_pool", lambda: pool)
    monkeypatch.setattr(krm_mod, "_list_all_dirs", lambda: [dict(d) for d in DIRS])
    monkeypatch.setattr(krm_mod, "_load_krm_direction", lambda _code: DIRECTION_DATA)
    return TestClient(create_app(), raise_server_exceptions=False)


# ----------------------------------------------------- my-directions

class TestMyDirections:
    def test_admin_sees_all_directions(self, monkeypatch):
        r = _client(monkeypatch, role="admin").get("/api/krm/teacher/my-directions")
        assert r.status_code == 200, r.text
        assert r.json() == DIRS

    def test_teacher_without_bindings_fallback_all(self, monkeypatch):
        """Привязок нет: teacher не теряет доступ (fallback на весь справочник)."""
        r = _client(monkeypatch, role="teacher", pool=_EmptyPool()).get(
            "/api/krm/teacher/my-directions"
        )
        assert r.status_code == 200, r.text
        assert r.json() == DIRS

    def test_teacher_with_unavailable_pool_gets_nothing(self, monkeypatch):
        """БД недоступна: доступ закрывается, а не расширяется.

        При pool=None _user_directions возвращает [] до ветки fallback -
        это осознанный fail-closed, иначе падение БД выдавало бы teacher все
        направления.
        """
        r = _client(monkeypatch, role="teacher").get("/api/krm/teacher/my-directions")
        assert r.status_code == 200, r.text
        assert r.json() == []

    def test_rop_without_bindings_gets_nothing(self, monkeypatch):
        """rop без привязок (пул None) направлений не видит: это не fallback-роль."""
        r = _client(monkeypatch, role="rop").get("/api/krm/teacher/my-directions")
        assert r.status_code == 200, r.text
        assert r.json() == []

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/krm/teacher/my-directions").status_code == 401


# ----------------------------------------------------- check-access

class TestCheckAccess:
    def test_admin_has_access(self, monkeypatch):
        r = _client(monkeypatch, role="admin").get(
            "/api/krm/teacher/check-access?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        assert r.json() == {"access": True, "dir_code": "09.03.02"}

    def test_rop_without_bindings_has_no_access(self, monkeypatch):
        r = _client(monkeypatch, role="rop").get(
            "/api/krm/teacher/check-access?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        assert r.json() == {"access": False, "dir_code": "09.03.02"}

    def test_invalid_dir_code_is_access_false_not_500(self, monkeypatch):
        """Невалидный код не роняет ручку: клиенту нужен ложный ответ."""
        r = _client(monkeypatch, role="teacher").get(
            "/api/krm/teacher/check-access?dir_code=../etc")
        assert r.status_code == 200, r.text
        assert r.json() == {"access": False, "dir_code": "../etc"}

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/krm/teacher/check-access").status_code == 401


# ----------------------------------------------------- directions/{dir_code}

class TestDirectionData:
    def test_admin_gets_direction(self, monkeypatch):
        r = _client(monkeypatch, role="admin").get(
            "/api/krm/teacher/directions/09.03.02")
        assert r.status_code == 200, r.text
        assert r.json()["direction_name"] == "Информационные системы и технологии"
        assert r.json()["disciplines"]

    def test_rop_without_bindings_forbidden(self, monkeypatch):
        """rop без привязок не читает направление целиком: 403, не 200."""
        r = _client(monkeypatch, role="rop").get(
            "/api/krm/teacher/directions/09.03.02")
        assert r.status_code == 403, r.text
        assert "запрещ" in r.json()["detail"]

    def test_invalid_dir_code_rejected_400(self, monkeypatch):
        """Код уходит в имя файла: невалидный формат - 400, а не путь в ФС."""
        r = _client(monkeypatch, role="admin").get(
            "/api/krm/teacher/directions/09.03")
        assert r.status_code == 400, r.text
        assert "format" in r.json()["detail"]

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/krm/teacher/directions/09.03.02").status_code == 401


# ----------------------------------------------------- teacher/history

class TestTeacherHistory:
    @staticmethod
    def _write_history(tmp_path, rows, bad_line=False):
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        if bad_line:
            lines.append("corrupted line {")
        (history_dir / "09.03.02_2025.jsonl").write_text("\n".join(lines), encoding="utf-8")
        return history_dir

    def test_delta_between_first_and_last_point(self, monkeypatch, tmp_path):
        rows = [
            {"average_coverage": 0.4, "total_gaps": 10, "total_disciplines": 4},
            {"average_coverage": 0.55, "total_gaps": 7, "total_disciplines": 4},
        ]
        monkeypatch.setattr(history_mod, "HISTORY_DIR", self._write_history(tmp_path, rows))
        r = _client(monkeypatch, role="teacher").get("/api/teacher/history?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["first"]["total_gaps"] == 10
        assert body["last"]["total_gaps"] == 7
        assert body["delta"]["average_coverage"] == pytest.approx(0.15)
        assert body["delta"]["total_gaps"] == -3
        assert body["skipped_lines"] == 0

    def test_corrupted_lines_skipped_not_500(self, monkeypatch, tmp_path):
        rows = [{"average_coverage": 0.5, "total_gaps": 1, "total_disciplines": 1}]
        monkeypatch.setattr(
            history_mod, "HISTORY_DIR", self._write_history(tmp_path, rows, bad_line=True))
        r = _client(monkeypatch, role="teacher").get("/api/teacher/history?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        assert len(r.json()["points"]) == 1
        assert r.json()["skipped_lines"] == 1

    def test_empty_history_returns_null_delta(self, monkeypatch, tmp_path):
        monkeypatch.setattr(history_mod, "HISTORY_DIR", self._write_history(tmp_path, []))
        r = _client(monkeypatch, role="teacher").get("/api/teacher/history?dir_code=09.03.02")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["points"] == []
        assert body["delta"] is None

    def test_invalid_dir_code_rejected_400(self, monkeypatch):
        """dir_code уходит в путь файла: без валидации - обход каталога."""
        r = _client(monkeypatch, role="teacher").get(
            "/api/teacher/history?dir_code=../../etc/passwd")
        assert r.status_code == 400, r.text

    def test_requires_token(self, monkeypatch):
        client = _client(monkeypatch, role=None)
        assert client.get("/api/teacher/history").status_code == 401

    def test_student_forbidden(self, monkeypatch):
        """Гейт admin/teacher/rop: student получает 403, а не снимки покрытия."""
        client = _client(monkeypatch, role="student")
        assert client.get("/api/teacher/history").status_code == 403
