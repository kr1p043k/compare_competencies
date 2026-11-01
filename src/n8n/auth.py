"""n8n API key auth middleware."""

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = structlog.get_logger("n8n_auth")

N8N_PREFIXES = {"/api/n8n"}


class N8NAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        for prefix in N8N_PREFIXES:
            if request.url.path.startswith(prefix):
                from src import config

                api_key = config.N8N_API_KEY
                if api_key is None:
                    # 503, а не 500: интеграция не настроена - это не сбой сервера.
                    # Из middleware нельзя поднимать HTTPException - он уходил «сырым»
                    # 500 с трейсбеком; возвращаем JSONResponse.
                    logger.error("n8n_auth_not_configured", path=request.url.path)
                    return JSONResponse(
                        status_code=503, content={"detail": "n8n integration is not configured"}
                    )
                auth = request.headers.get("Authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != api_key.get_secret_value():
                    logger.warning(
                        "n8n_auth_failed", path=request.url.path,
                        client=request.client.host if request.client else "unknown"
                    )
                    return JSONResponse(status_code=401, content={"detail": "Invalid n8n API key"})
                break
        return await call_next(request)
