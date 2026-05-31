import pytest
from unittest.mock import patch, MagicMock

from src.n8n.auth import N8NAuthMiddleware, N8N_PREFIXES


class TestN8NPrefixes:
    def test_prefixes_are_set(self):
        assert len(N8N_PREFIXES) > 0
        assert "/api/n8n" in N8N_PREFIXES


class TestN8NAuthMiddleware:
    def test_middleware_class_exists(self):
        assert N8NAuthMiddleware is not None

    @pytest.mark.asyncio
    async def test_middleware_dispatch_no_auth_configured(self):
        from fastapi import HTTPException
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/n8n/webhook/test"
        mock_request.headers = {}
        mock_request.client = None

        async def call_next(req):
            return MagicMock(spec=JSONResponse)

        middleware = N8NAuthMiddleware(MagicMock())
        with patch("src.config") as mock_config:
            mock_config.N8N_API_KEY = None
            with pytest.raises(HTTPException) as exc_info:
                await middleware.dispatch(mock_request, call_next)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_middleware_dispatch_invalid_key(self):
        from fastapi import HTTPException
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/n8n/webhook/test"
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        mock_request.client = None

        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "real-key"

        async def call_next(req):
            return MagicMock(spec=JSONResponse)

        middleware = N8NAuthMiddleware(MagicMock())
        with patch("src.config") as mock_config:
            mock_config.N8N_API_KEY = mock_secret
            with pytest.raises(HTTPException) as exc_info:
                await middleware.dispatch(mock_request, call_next)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_middleware_dispatch_success(self):
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/n8n/webhook/test"
        mock_request.headers = {"Authorization": "Bearer real-key"}
        mock_request.client = None

        mock_secret = MagicMock()
        mock_secret.get_secret_value.return_value = "real-key"

        mock_response = MagicMock(spec=JSONResponse)

        async def call_next(req):
            return mock_response

        middleware = N8NAuthMiddleware(MagicMock())
        with patch("src.config") as mock_config:
            mock_config.N8N_API_KEY = mock_secret
            result = await middleware.dispatch(mock_request, call_next)
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_middleware_skips_non_n8n_routes(self):
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/api/health"
        mock_request.client = None

        mock_response = MagicMock(spec=JSONResponse)

        async def call_next(req):
            return mock_response

        middleware = N8NAuthMiddleware(MagicMock())
        result = await middleware.dispatch(mock_request, call_next)
        assert result == mock_response
