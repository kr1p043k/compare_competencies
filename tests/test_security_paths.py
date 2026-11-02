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
        from src.api_pkg import create_app
        import src.api_pkg.routers.auth as auth

        async def _teacher(_request):
            return {"uid": "u-1", "u": "teacher@example.com", "r": "teacher", "t": None}

        monkeypatch.setattr(auth, "get_current_user", _teacher)
        client = TestClient(create_app(), raise_server_exceptions=False)
        assert client.get("/metrics", headers={"Authorization": "Bearer x"}).status_code == 403

    def test_metrics_admin_200(self, monkeypatch):
        """Админ проходит гейт (проверяем, что 403 не «всем подряд»)."""
        from src.api_pkg import create_app
        import src.api_pkg.routers.auth as auth

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
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/pipeline/ws"):
                pass  # соединение не должно быть принято
        assert exc_info.value.code == 4001

    def test_ws_revoked_session_closed_4001(self, monkeypatch):
        """Токен подписан верно, но сессия в БД отозвана - соединение не принимаем.

        Это регресс на дыру: раньше WS проверял только подпись токена, и logout
        не действовал на открытый сокет до истечения TTL.
        """
        from src.api_pkg import create_app
        import src.api_pkg.routers.auth as auth

        async def _revoked(_token):
            return None

        monkeypatch.setattr(auth, "resolve_session", _revoked)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/pipeline/ws?token=signed-but-revoked"):
                pass
        assert exc_info.value.code == 4001

    def test_ws_valid_session_accepted(self, monkeypatch):
        """Валидная активная сессия - соединение принимается."""
        from src.api_pkg import create_app
        import src.api_pkg.routers.auth as auth

        async def _ok(_token):
            return {"uid": "u-3", "u": "admin@example.com", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _ok)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with client.websocket_connect("/api/pipeline/ws?token=valid") as ws:
            ws.close()
