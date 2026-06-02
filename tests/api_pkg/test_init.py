"""Тесты для src.api_pkg.__init__ — фабрика FastAPI приложения."""

import pytest
from fastapi import FastAPI, HTTPException
from slowapi.errors import RateLimitExceeded
from starlette.middleware.cors import CORSMiddleware

from src.api_pkg import create_app, app
from src import config


class TestCreateApp:
    def test_returns_fastapi_instance(self):
        result = create_app()
        assert isinstance(result, FastAPI)

    def test_title_and_version(self):
        assert app.title == "Competency Analyzer API"
        assert app.version == "2.0"

    def test_has_rate_limit_exception_handler(self):
        handlers = app.exception_handlers or {}
        exc_types = list(handlers.keys()) if isinstance(handlers, dict) else [h[0] for h in handlers]
        assert RateLimitExceeded in exc_types

    def test_has_http_exception_handler(self):
        handlers = app.exception_handlers or {}
        exc_types = list(handlers.keys()) if isinstance(handlers, dict) else [h[0] for h in handlers]
        assert HTTPException in exc_types

    def test_has_global_exception_handler(self):
        handlers = app.exception_handlers or {}
        exc_types = list(handlers.keys()) if isinstance(handlers, dict) else [h[0] for h in handlers]
        assert Exception in exc_types

    def test_cors_middleware_is_configured(self):
        middlewares = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middlewares

    def test_limiter_in_app_state(self):
        assert hasattr(app.state, "limiter")

    def test_routers_are_included(self):
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in paths

    def test_allowed_origins_defaults_to_star(self, monkeypatch):
        monkeypatch.setattr(config, "ALLOWED_ORIGINS", "*")
        new_app = create_app()
        cors_mw = [m for m in new_app.user_middleware if m.cls == CORSMiddleware]
        assert len(cors_mw) == 1

    def test_allowed_origins_parsed_from_comma_list(self, monkeypatch):
        monkeypatch.setattr(config, "ALLOWED_ORIGINS", "http://a.com,http://b.com")
        new_app = create_app()
        cors_mw = [m for m in new_app.user_middleware if m.cls == CORSMiddleware]
        assert len(cors_mw) == 1

    def test_body_size_limit_middleware_rejects_large(self):
        from starlette.testclient import TestClient
        client = TestClient(app)
        large_body = "x" * (config.MAX_REQUEST_SIZE + 1)
        resp = client.post(
            "/health",
            content=large_body,
            headers={"content-length": str(len(large_body))},
        )
        assert resp.status_code == 413

    def test_create_app_is_callable(self):
        assert callable(create_app)

    def test_app_is_fastapi_instance(self):
        assert isinstance(app, FastAPI)
