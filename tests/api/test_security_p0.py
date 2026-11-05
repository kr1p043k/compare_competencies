"""P0 security gates: admin-only user creation, authed export."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from tests.conftest import route_gate_roles


def _route_gate_roles(path: str, method: str = "GET") -> set[tuple]:
    """Роли гейтов маршрута, включая router-level зависимости.

    На FastAPI 0.162+ маршруты лежат во вложенных роутерах, поэтому ходим
    через conftest.route_gate_roles, а не по app.routes.
    """
    return route_gate_roles(create_app(), path, method)


def test_users_create_admin_only():
    roles = _route_gate_roles("/api/admin/users/create", "POST")
    assert roles, "users/create must carry auth dependency"
    assert ("admin",) in roles, f"no admin-only gate, have {roles}"


def test_export_requires_auth():
    roles = _route_gate_roles("/api/teacher/export/vacancies", "GET")
    assert roles, "export must carry auth dependency"


def test_no_token_unauthorized():
    client = TestClient(create_app(), raise_server_exceptions=False)
    r1 = client.post("/api/admin/users/create",
                     json={"email": "x@y.z", "password": "pw", "role": "teacher"})
    assert r1.status_code in (401, 422)
    r2 = client.get("/api/teacher/export/vacancies")
    assert r2.status_code == 401


def test_export_rejects_bad_date_with_400_not_500(monkeypatch):
    """Невалидная дата -> 400, а не 500.

    Регресс: локальный `from fastapi import HTTPException` внутри функции делал
    имя локальным для ВСЕЙ функции, и более ранний raise падал с
    UnboundLocalError (500). Ловилось только живым smoke-прогоном всех GET.
    """
    import src.api_pkg.routers.teacher as teacher
    from tests.conftest import open_all_gates

    class _Pool:
        async def fetch(self, *_a, **_kw):
            return []

    monkeypatch.setattr(teacher, "get_pool", lambda: _Pool())
    app = open_all_gates(create_app(), role="admin")
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/api/teacher/export/vacancies?date_from=2026-13-45")
    assert r.status_code == 400, r.text


def test_market_endpoints_require_auth():
    """Рыночные агрегаты закрыты авторизацией.

    Раньше /market/* отдавались без токена: любой внешний клиент тянул
    агрегаты рынка. UI ходит через api() с токеном, поэтому гейт его не ломает.
    """
    client = TestClient(create_app(), raise_server_exceptions=False)
    for path in (
        "/api/market/top-skills",
        "/api/market-competencies",
        "/api/market/skill/python",
    ):
        r = client.get(path)
        assert r.status_code == 401, f"{path} -> {r.status_code}"
