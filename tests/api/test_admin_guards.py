"""Админские write-эндпоинты: только admin, и гейт стоит на каждом.

Две проверки:
  1. статически - у маршрута есть роль-гейт ('admin',) (route_gate_roles);
  2. поведенчески - без токена 401, с ролью teacher 403.

ВАЖНО: здесь НЕ используется open_all_gates - он подменяет гейт на «всегда
пропустить», и тогда тест выполнял бы реальную операцию (запуск пайплайна,
seed БД, запись whitelist). Вместо этого подменяем только get_current_user,
а настоящий require_any_role проверяет роль и отдаёт 403 до вызова handler.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
from src.api_pkg import create_app
from tests.conftest import route_gate_roles

ADMIN_ONLY_WRITES = [
    ("POST", "/api/admin/whitelist/add"),
    ("POST", "/api/admin/whitelist/remove"),
    ("POST", "/api/admin/pipeline/trigger"),
    ("POST", "/api/admin/users/create"),
    ("POST", "/api/admin/db/seed"),
    ("POST", "/api/admin/embeddings/generate"),
    ("POST", "/api/admin/skills/categorize"),
    ("POST", "/api/admin/db/backup"),
    ("PUT", "/api/admin/scheduler/settings"),
    ("POST", "/api/admin/scheduler/collect/now"),
    ("POST", "/api/admin/scheduler/gap/now"),
]


@pytest.mark.parametrize("method,path", ADMIN_ONLY_WRITES)
def test_admin_write_has_admin_gate(method, path):
    """Каждый write-эндпоинт админки закрыт гейтом ('admin',)."""
    roles = route_gate_roles(create_app(), path, method)
    assert ("admin",) in roles, f"{method} {path}: нет admin-гейта, есть {roles}"


@pytest.mark.parametrize("method,path", ADMIN_ONLY_WRITES)
def test_admin_write_rejects_anonymous(method, path):
    """Без токена - 401, а не выполнение операции."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.request(method, path, json={})
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}: {r.text[:200]}"


@pytest.fixture
def teacher_client(monkeypatch):
    """Клиент с «настоящим» пользователем-преподавателем.

    get_current_user подменён, require_any_role остаётся настоящим - поэтому
    гейт срабатывает и handler не выполняется (никаких побочных эффектов).
    """

    async def _teacher(_request):
        return {"uid": "t-1", "u": "teacher@example.com", "r": "teacher", "t": None}

    monkeypatch.setattr(auth_mod, "get_current_user", _teacher)
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("method,path", ADMIN_ONLY_WRITES)
def test_admin_write_rejects_teacher(teacher_client, method, path):
    """Преподаватель получает 403 до выполнения операции."""
    r = teacher_client.request(method, path, json={})
    assert r.status_code == 403, f"{method} {path} -> {r.status_code}: {r.text[:200]}"


def test_public_log_endpoint_is_not_admin_gated():
    """/log пишут все роли - он не должен требовать admin."""
    roles = route_gate_roles(create_app(), "/api/log", "POST")
    assert ("admin",) not in roles
