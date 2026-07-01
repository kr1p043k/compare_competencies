"""API package — фабрика FastAPI приложения."""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from src.monitoring.metrics import get_metrics

from src import config

# Suppress cmdstanpy BEFORE any imports that might trigger Prophet
import logging as _logging
_cmdstan_logger = _logging.getLogger("cmdstanpy")
_cmdstan_logger.setLevel(_logging.WARNING)
for _h in _cmdstan_logger.handlers[:]:
    _cmdstan_logger.removeHandler(_h)
_logging.getLogger("prophet").setLevel(_logging.WARNING)
_logging.getLogger("cmdstanpy.cmdstan").setLevel(_logging.WARNING)

from src.api_pkg import deps as deps  # noqa: F401

logger = structlog.get_logger("api")


def _rate_limit_key(request: Request) -> str:
    """Rate limit by user email (from JWT) if authenticated, else by IP."""
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        email = user.get("u")
        if email:
            return f"user:{email}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    return f"ip:{client_ip}"


limiter = Limiter(key_func=_rate_limit_key)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from src.api_pkg.startup import run_startup

        await run_startup(app)
        yield
        from src.db import close_pool
        from src.database import get_engine
        await close_pool()
        await get_engine().dispose()
        logger.info("API shutting down...")

    app = FastAPI(
        title="Competency Analyzer API",
        version="2.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        lambda request, exc: JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."},
        ),
    )

    allowed_origins = (
        config.ALLOWED_ORIGINS.split(",") if config.ALLOWED_ORIGINS != "*" else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def limit_body_size_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.MAX_REQUEST_SIZE:
            logger.warning("request_body_too_large", size=int(content_length))
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
        return await call_next(request)

    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(
            "Incoming request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            process_time=process_time,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled exception", request_id=request_id, error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
                "error_type": type(exc).__name__,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            "HTTP exception",
            request_id=request_id,
            status_code=exc.status_code,
            detail=exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
        )

    from src.api_pkg.routers.health import router as health_router
    app.include_router(health_router)  # /, /health, /ready (without /api)

    def _mount(router, tag=""):
        app.include_router(router, prefix="/api")
        app.include_router(router, prefix="/api/v1", include_in_schema=False)

    from src.api_pkg.routers.profiles import router as profiles_router
    _mount(profiles_router)
    from src.api_pkg.routers.market import router as market_router
    _mount(market_router)
    from src.api_pkg.routers.clusters import router as clusters_router
    _mount(clusters_router)
    from src.api_pkg.routers.trends import router as trends_router
    _mount(trends_router)
    from src.api_pkg.routers.taxonomy import router as taxonomy_router
    _mount(taxonomy_router)
    from src.api_pkg.routers.vacancies import router as vacancies_router
    _mount(vacancies_router)
    from src.api_pkg.routers.vacancies_by_skill import router as vacancies_skill_router
    _mount(vacancies_skill_router)
    from src.api_pkg.routers.results import router as results_router
    _mount(results_router)
    from src.api_pkg.routers.pipeline import router as pipeline_router
    _mount(pipeline_router)
    from src.api_pkg.routers.admin import router as admin_router
    _mount(admin_router)
    from src.api_pkg.routers.forecast import router as forecast_router
    _mount(forecast_router)
    from src.api_pkg.routers.trends_by_profession import router as trends_prof_router
    _mount(trends_prof_router)
    from src.api_pkg.routers.auth import router as auth_router
    _mount(auth_router)
    from src.api_pkg.routers.teacher import router as teacher_router
    _mount(teacher_router)
    from src.api_pkg.routers.student import router as student_router
    _mount(student_router)

    from src.n8n.webhooks import router as n8n_webhook_router
    app.include_router(n8n_webhook_router)  # no versioning

    from src.api_pkg.request_logger import RequestLogMiddleware

    app.add_middleware(RequestLogMiddleware)

    from src.n8n.auth import N8NAuthMiddleware

    app.add_middleware(N8NAuthMiddleware)

    @app.get("/metrics")
    async def metrics():
        from src.monitoring.metrics import get_metrics
        data, content_type = get_metrics()
        from fastapi.responses import Response
        return Response(content=data, media_type=content_type)

    return app


app = create_app()
