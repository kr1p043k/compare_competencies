"""Unit-тесты security-правок: источники токена, resolve_session, /metrics, WS auth.

Все тесты без сети и реальной БД: пул подменяется MagicMock/AsyncMock,
HTTP/WS - через TestClient(create_app()) без запуска lifespan.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def _req(headers=None, query=b"", cookies=""):
    """ASGI Request без сети (helper как в test_stability_blocks.py)."""
    scope = {
        "type": "http",
        "path": "/",
        "headers": [(k.lower(), v) for k, v in (headers or [])],
        "query_string": query,
    }
    if cookies:
        scope["headers"] = [(b"cookie", cookies.encode())] + scope["headers"]
    return Request(scope)


class TestRequestTokenSources:
    """_request_token: query-параметр игнорируется, Bearer и cookie работают."""

    def test_query_token_ignored_returns_empty(self):
        from src.api_pkg.routers.auth import _request_token

        assert _request_token(_req(query=b"token=SECRET")) == ""

    def test_bearer_accepted(self):
        from src.api_pkg.routers.auth import _request_token

        r = _req([(b"authorization", b"Bearer abc123")])
        assert _request_token(r) == "abc123"

    def test_cookie_accepted(self):
        from src.api_pkg.routers.auth import _request_token

        assert _request_token(_req(cookies="token=ck")) == "ck"

    def test_bearer_wins_over_cookie_and_query(self):
        from src.api_pkg.routers.auth import _request_token

        r = _req([(b"authorization", b"Bearer B")], query=b"token=Q", cookies="token=C")
        assert _request_token(r) == "B"


class TestResolveSession:
    """resolve_session: пул/строка/флаг is_active; активная строка -> uid/u/r."""

    @staticmethod
    def _token() -> str:
        from src.api_pkg.routers.auth import _make_token

        return _make_token("u1", "x@y.z", "admin")

    @staticmethod
    def _pool(row) -> MagicMock:
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=row)
        return pool

    async def test_pool_none_returns_none(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        monkeypatch.setattr(auth, "get_pool", lambda: None)
        assert await auth.resolve_session(self._token()) is None

    async def test_row_none_returns_none(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        monkeypatch.setattr(auth, "get_pool", lambda: self._pool(None))
        assert await auth.resolve_session(self._token()) is None

    async def test_inactive_user_returns_none(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        row = {"id": "u1", "email": "x@y.z", "role": "admin", "is_active": False}
        monkeypatch.setattr(auth, "get_pool", lambda: self._pool(row))
        assert await auth.resolve_session(self._token()) is None

    async def test_active_row_returns_user_dict(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        row = {"id": 42, "email": "x@y.z", "role": "admin", "is_active": True}
        monkeypatch.setattr(auth, "get_pool", lambda: self._pool(row))
        user = await auth.resolve_session(self._token())
        assert user is not None
        assert user["uid"] == "42"  # str(row["id"])
        assert user["u"] == "x@y.z"
        assert user["r"] == "admin"

    async def test_empty_token_returns_none_without_pool(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        monkeypatch.setattr(
            auth,
            "get_pool",
            lambda: pytest.fail("get_pool must not be called for empty token"),
        )
        assert await auth.resolve_session("") is None

    async def test_db_error_returns_none(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        pool = MagicMock()
        pool.fetchrow = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr(auth, "get_pool", lambda: pool)
        assert await auth.resolve_session(self._token()) is None


class TestMetricsRequiresAdmin:
    """GET /metrics закрыт require_any_role("admin"): анонимно -> 401."""

    def test_metrics_anonymous_401(self):
        from src.api_pkg import create_app

        client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_teacher_403(self, monkeypatch):
        """Залогиненный не-админ получает 403, а не 200.

        Аноним даёт 401 на get_current_user; здесь подменяем его на teacher,
        чтобы проверить именно проверку роли в require_any_role.
        """
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _teacher(_request):
            return {"uid": "u-1", "u": "teacher@example.com", "r": "teacher", "t": None}

        monkeypatch.setattr(auth, "get_current_user", _teacher)
        client = TestClient(create_app(), raise_server_exceptions=False)
        assert client.get("/metrics", headers={"Authorization": "Bearer x"}).status_code == 403

    def test_metrics_admin_200(self, monkeypatch):
        """Админ проходит гейт (проверяем, что 403 не «всем подряд»)."""
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _admin(_request):
            return {"uid": "u-2", "u": "admin@example.com", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "get_current_user", _admin)
        client = TestClient(create_app(), raise_server_exceptions=False)
        assert client.get("/metrics", headers={"Authorization": "Bearer x"}).status_code == 200


class TestPipelineWsAuth:
    """WS /api/pipeline/ws без токена закрывается кодом 4001 до accept."""

    def test_ws_without_token_closed_4001(self):
        from src.api_pkg import create_app

        client = TestClient(create_app(), raise_server_exceptions=False)
        with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect("/api/pipeline/ws"):
            pass  # соединение не должно быть принято
        assert exc_info.value.code == 4001

    def test_ws_revoked_session_closed_4001(self, monkeypatch):
        """Токен подписан верно, но сессия в БД отозвана - соединение не принимаем.

        Это регресс на дыру: раньше WS проверял только подпись токена, и logout
        не действовал на открытый сокет до истечения TTL.
        """
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _revoked(_token):
            return None

        monkeypatch.setattr(auth, "resolve_session", _revoked)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect(
            "/api/pipeline/ws", headers={"Authorization": "Bearer signed-but-revoked"}
        ):
            pass
        assert exc_info.value.code == 4001

    def test_ws_valid_session_accepted(self, monkeypatch):
        """Валидная активная сессия - соединение принимается."""
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _ok(_token):
            return {"uid": "u-3", "u": "admin@example.com", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _ok)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with client.websocket_connect(
            "/api/pipeline/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            ws.close()

    def test_ws_query_token_not_accepted(self, monkeypatch):
        """?token= больше не аутентифицирует сокет: адрес утекает в логи."""
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _ok(_token):
            return {"uid": "u-3", "u": "admin@example.com", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _ok)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/pipeline/ws?token=valid"):
                pass
        assert exc_info.value.code == 4001


class TestDisciplineScopeIdor:
    """Scope дисциплины общий (файл), но менять его можно только в своих направлениях."""

    @staticmethod
    def _client(monkeypatch, tmp_path, dir_code: str | None, role="teacher", allow=("09.03.02",)):
        import src.analyzers.discipline_scope as scope_mod
        import src.api_pkg.routers.auth as auth
        import src.api_pkg.routers.teacher as teacher
        from src.api_pkg import create_app

        # Не писать в реальный data/reference/discipline_market_scope.json:
        # без подмены тест менял файл репозитория.
        monkeypatch.setattr(scope_mod, "OVERRIDES_PATH", tmp_path / "scope.json")

        async def _user(_request):
            return {"uid": "u-1", "u": f"{role}@example.com", "r": role, "t": None}

        class Pool:
            async def fetchval(self, *_a, **_kw):
                return dir_code

        monkeypatch.setattr(auth, "get_current_user", _user)
        monkeypatch.setattr(teacher, "get_pool", lambda: Pool())

        from src.api_pkg.routers import krm_teacher

        async def _dirs(_user):
            return list(allow)

        monkeypatch.setattr(krm_teacher, "_user_directions", _dirs)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_foreign_discipline_scope_is_403(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path, dir_code="01.03.01")
        r = client.put(
            "/api/teacher/discipline-scope/Чужая%20дисциплина",
            json={"skills": ["python"], "keywords": []},
        )
        assert r.status_code == 403, r.text

    def test_own_discipline_scope_allowed(self, monkeypatch, tmp_path):
        client = self._client(monkeypatch, tmp_path, dir_code="09.03.02")
        r = client.put(
            "/api/teacher/discipline-scope/Своя%20дисциплина",
            json={"skills": ["python"], "keywords": []},
        )
        assert r.status_code == 200, r.text

    def test_unknown_discipline_is_not_locked(self, monkeypatch, tmp_path):
        """Дисциплины нет в БД (кастомная) - ограничивать нечем, не 403."""
        client = self._client(monkeypatch, tmp_path, dir_code=None)
        r = client.put(
            "/api/teacher/discipline-scope/Кастомная",
            json={"skills": ["python"], "keywords": []},
        )
        assert r.status_code == 200, r.text


class TestAdminLogsWs:
    """WS /admin/logs/ws: сессия из БД, роль admin, токен не из query."""

    @staticmethod
    def _client(monkeypatch, user=None):
        import src.api_pkg.routers.auth as auth
        from src.api_pkg import create_app

        async def _resolve(_token):
            return user

        monkeypatch.setattr(auth, "resolve_session", _resolve)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_no_token_closed_4001(self, monkeypatch):
        client = self._client(monkeypatch)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/api/admin/logs/ws"):
            pass
        assert exc.value.code == 4001

    def test_query_token_not_accepted(self, monkeypatch):
        """?token= больше не аутентифицирует сокет: адрес утекает в логи."""
        client = self._client(
            monkeypatch, {"uid": "u1", "u": "a@b", "r": "admin", "t": None}
        )
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/admin/logs/ws?token=valid"):
                pass
        assert exc.value.code == 4001

    def test_revoked_session_closed_4001(self, monkeypatch):
        client = self._client(monkeypatch, user=None)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/admin/logs/ws", headers={"Authorization": "Bearer revoked"}
        ):
            pass
        assert exc.value.code == 4001

    def test_teacher_forbidden_4003(self, monkeypatch):
        """Роль берётся из БД: понижение в правах закрывает сокет."""
        client = self._client(
            monkeypatch, {"uid": "u2", "u": "t@b", "r": "teacher", "t": None}
        )
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/admin/logs/ws", headers={"Authorization": "Bearer valid"}
        ):
            pass
        assert exc.value.code == 4003
