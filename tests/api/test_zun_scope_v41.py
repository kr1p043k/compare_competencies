"""Unit (v41): scope endpoints validation (fake pool, auth overridden)."""
import pytest as _pytest
from fastapi.testclient import TestClient

import src.api_pkg.routers.zun as _zun_mod
from src.api_pkg import create_app


@_pytest.fixture(autouse=True)
def _allow_all_dirs(monkeypatch):
    """IDOR-guard проверяется в tests/api/test_zun_idor.py.

    Здесь - логика эндпоинтов, поэтому доступ к направлению считаем
    разрешённым. Через fixture, а не module-level: иначе подмена протекает в
    соседние тестовые файлы.
    """
    async def _allow(*_args, **_kwargs):
        return None

    monkeypatch.setattr(_zun_mod, "_require_dir_access", _allow)
    monkeypatch.setattr(_zun_mod, "_ensure_competency_access", _allow)
    monkeypatch.setattr(_zun_mod, "_ensure_discipline_access", _allow)
    monkeypatch.setattr(_zun_mod, "_ensure_ksa_access", _allow)



# IDOR-guard проверяется отдельно (tests/api/test_zun_idor.py); здесь - логика
# эндпоинтов, поэтому доступ к направлению считаем разрешённым.
async def _allow_all(_request, _dir_code):
    return None


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
    from tests.conftest import open_all_gates

    app = create_app()
    # Гейты авторизации снимаем общим хелпером: он понимает вложенные
    # роутеры FastAPI 0.162+ (см. tests/conftest.py).
    open_all_gates(app, role="admin")
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


class TestCoverageLevelConservative:
    """ZUN-уровень консервативен, как в teacher reports: min(ratio, weighted)."""

    def test_min_rule(self):
        from src.api_pkg.routers.zun import _coverage_level
        assert _coverage_level(0.55, 0.33) == "medium"
        assert _coverage_level(0.9, 0.8) == "high"
        assert _coverage_level(0.1, 0.05) == "low"

    def test_weighted_none_fallback(self):
        """Старые строки без weighted: поведение как раньше."""
        from src.api_pkg.routers.zun import _coverage_level
        assert _coverage_level(0.9) == "high"
        assert _coverage_level(0.3) == "medium"
        assert _coverage_level(None) is None

    def test_legacy_null_weighted_falls_back_to_ratio(self):
        """Legacy-строка (weighted NULL): уровень по ratio, не low."""
        from src.api_pkg.routers.zun import _coverage_level
        assert _coverage_level(0.85, None) == "high"
        assert _coverage_level(0.55, None) == "high"
