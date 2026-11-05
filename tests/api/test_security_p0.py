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
