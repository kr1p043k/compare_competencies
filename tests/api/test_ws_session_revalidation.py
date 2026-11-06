"""WS re-validation: logout/deactivation must close an already-open socket.

Сейчас сессия проверяется только на хендшейке. После отзыва токена, деактивации
или смены роли открытый сокет продолжал стримить данные. Тесты проверяют, что
сессия ревалидируется внутри стрима (закрытие кодом 4001) и что время жизни
сокета ограничено (закрытие кодом 1000), а также Origin-allowlist на хендшейке.
"""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.api_pkg.routers.auth as auth
from src import config
from src.api_pkg import create_app


def _receive_until_close(ws):
    """Читаем сообщения, пока сервер не закроет сокет (raise WebSocketDisconnect)."""
    while True:
        ws.receive_text()


class TestAdminLogsWsRevalidation:
    @staticmethod
    def _client(monkeypatch, tmp_path, resolve, *, loop=0.05, recheck=0.0, lifetime=1e9):
        import src.api_pkg.routers.admin as admin

        monkeypatch.setattr(auth, "resolve_session", resolve)
        monkeypatch.setattr(admin, "_tail_backend_log", _noop_tail)
        monkeypatch.setattr(config, "LOG_FILE", tmp_path / "backend.log")
        monkeypatch.setattr(admin, "_WS_LOOP_INTERVAL_SEC", loop)
        monkeypatch.setattr(admin, "_WS_SESSION_RECHECK_SEC", recheck)
        monkeypatch.setattr(admin, "_WS_MAX_LIFETIME_SEC", lifetime)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_session_revoked_mid_stream_closes_4001(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        async def _resolve(_token):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"uid": "u1", "u": "a@b", "r": "admin", "t": None}
            return None  # logout / деактивация на втором вызове

        client = self._client(monkeypatch, tmp_path, _resolve)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/admin/logs/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            _receive_until_close(ws)
        assert exc.value.code == 4001
        assert calls["n"] >= 2  # сессия проверена повторно внутри стрима

    def test_role_downgraded_mid_stream_closes_4001(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        async def _resolve(_token):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"uid": "u2", "u": "a@b", "r": "admin", "t": None}
            return {"uid": "u2", "u": "a@b", "r": "teacher", "t": None}

        client = self._client(monkeypatch, tmp_path, _resolve)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/admin/logs/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            _receive_until_close(ws)
        assert exc.value.code == 4001

    def test_lifetime_cap_closes_1000(self, monkeypatch, tmp_path):
        async def _resolve(_token):
            return {"uid": "u3", "u": "a@b", "r": "admin", "t": None}

        client = self._client(
            monkeypatch, tmp_path, _resolve, loop=0.05, recheck=1e9, lifetime=0.0
        )
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/admin/logs/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            _receive_until_close(ws)
        assert exc.value.code == 1000


class TestPipelineWsRevalidation:
    @staticmethod
    def _client(monkeypatch, resolve, *, recheck=0.0, lifetime=1e9):
        import src.api_pkg.routers.pipeline as pipeline

        monkeypatch.setattr(auth, "resolve_session", resolve)
        monkeypatch.setattr(pipeline, "_WS_SESSION_RECHECK_SEC", recheck)
        monkeypatch.setattr(pipeline, "_WS_MAX_LIFETIME_SEC", lifetime)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_session_revoked_mid_stream_closes_4001(self, monkeypatch):
        calls = {"n": 0}

        async def _resolve(_token):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"uid": "u1", "u": "a@b", "r": "teacher", "t": None}
            return None

        client = self._client(monkeypatch, _resolve)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/pipeline/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            _receive_until_close(ws)
        assert exc.value.code == 4001
        assert calls["n"] >= 2

    def test_lifetime_cap_closes_1000(self, monkeypatch):
        async def _resolve(_token):
            return {"uid": "u2", "u": "a@b", "r": "teacher", "t": None}

        client = self._client(monkeypatch, _resolve, recheck=1e9, lifetime=0.0)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/pipeline/ws", headers={"Authorization": "Bearer valid"}
        ) as ws:
            _receive_until_close(ws)
        assert exc.value.code == 1000


class TestWsOriginAllowlist:
    """Origin-allowlist: чужие Origin отклоняются до accept кодом 4003."""

    def test_disallowed_origin_rejected_4003(self, monkeypatch):
        import src.api_pkg.routers.auth as auth

        async def _resolve(_token):
            return {"uid": "u1", "u": "a@b", "r": "admin", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _resolve)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect(
            "/api/pipeline/ws",
            headers={"Authorization": "Bearer valid", "Origin": "http://evil.example"},
        ):
            pass
        assert exc.value.code == 4003

    def test_allowed_origin_accepted(self, monkeypatch):
        async def _resolve(_token):
            return {"uid": "u2", "u": "a@b", "r": "teacher", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _resolve)
        client = TestClient(create_app(), raise_server_exceptions=False)
        with client.websocket_connect(
            "/api/pipeline/ws",
            headers={"Authorization": "Bearer valid", "Origin": "http://localhost:5173"},
        ) as ws:
            ws.close()

    def test_wildcard_origin_allows_any(self, monkeypatch):
        async def _resolve(_token):
            return {"uid": "u3", "u": "a@b", "r": "teacher", "t": None}

        monkeypatch.setattr(auth, "resolve_session", _resolve)
        monkeypatch.setattr(config, "ALLOWED_ORIGINS", "*")
        client = TestClient(create_app(), raise_server_exceptions=False)
        with client.websocket_connect(
            "/api/pipeline/ws",
            headers={"Authorization": "Bearer valid", "Origin": "http://anything.example"},
        ) as ws:
            ws.close()


async def _noop_tail(websocket, stop):
    """Заглушка хвоста лога: тест не должен читать реальный backend.log.

    В суите asyncio.sleep замокан, и реальный _tail_backend_log превратился бы в
    занятый цикл без точек уступки event loop - завешивая тест.
    """
    return None
