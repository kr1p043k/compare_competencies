"""Unit (v41): scope endpoints validation (fake pool, auth overridden)."""
from fastapi.testclient import TestClient

from src.api_pkg import create_app


class FakePool:
    def __init__(self):
        self.exec_log: list[str] = []

    async def fetch(self, sql, *a):
        self.exec_log.append(sql)
        if "FROM disciplines d2" in sql:
            return [{"name": "Базы данных и СУБД"}, {"name": "Философия"}]
        if "FROM discipline_scope" in sql:
            return []
        return []

    async def execute(self, sql, *a):
        self.exec_log.append(sql)
        return "OK"


def _client(pool):
    app = create_app()
    for route in app.routes:
        dep = getattr(route, "dependant", None)
        for sub in (getattr(dep, "dependencies", None) or []):
            app.dependency_overrides[sub.call] = lambda: {"r": "admin"}
    import src.api_pkg.routers.zun as z
    z.get_pool = lambda: pool
    return TestClient(app)


def test_get_scope_shape():
    r = _client(FakePool()).get(
        "/api/teacher/zun/scope?dir_code=09.03.02")
    assert r.status_code == 200, r.text
    body = r.json()
    by_name = {d['name']: d for d in body['disciplines']}
    assert by_name['Базы данных и СУБД']['in_scope'] is True
    assert by_name['Философия']['in_scope'] is False
    assert by_name['Философия']['source'] == 'methodology'
    assert "Философия" in body["methodology_excluded"]


def test_put_scope_validations():
    c = _client(FakePool())
    assert c.put('/api/teacher/zun/scope',
                 json={"dir_code": "09.03.02", "changes": []}).status_code == 400
    assert c.put('/api/teacher/zun/scope',
                 json={"dir_code": "09.03.02",
                       "changes": [{"discipline_name": "Nope", "included": False}]}).status_code == 400
    assert c.put('/api/teacher/zun/scope',
                 json={"dir_code": "bad", "changes": [{"discipline_name": "X",
                       "included": True}]}).status_code == 400


def test_put_scope_upserts():
    pool = FakePool()
    r = _client(pool).put('/api/teacher/zun/scope',
        json={"dir_code": "09.03.02", "changes": [
            {"discipline_name": "Базы данных и СУБД", "included": False},
            {"discipline_name": "Философия", "included": True}]})
    assert r.status_code == 200, r.text
    assert r.json()['updated'] == 2
    assert any('INSERT INTO discipline_scope' in s for s in pool.exec_log)
