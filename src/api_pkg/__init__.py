"""API package — фабрика FastAPI приложения."""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src import config
from src.api_pkg import deps as deps  # noqa: F401

logger = structlog.get_logger("api")

limiter = Limiter(key_func=get_remote_address)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from src.api_pkg.startup import run_startup

        await run_startup(app)
        yield
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

    app.include_router(health_router)
    from src.api_pkg.routers.profiles import router as profiles_router

    app.include_router(profiles_router)
    from src.api_pkg.routers.market import router as market_router

    app.include_router(market_router)
    from src.api_pkg.routers.clusters import router as clusters_router

    app.include_router(clusters_router)
    from src.api_pkg.routers.trends import router as trends_router

    app.include_router(trends_router)
    from src.api_pkg.routers.taxonomy import router as taxonomy_router

    app.include_router(taxonomy_router)
    from src.api_pkg.routers.vacancies import router as vacancies_router

    app.include_router(vacancies_router)
    from src.api_pkg.routers.results import router as results_router

    app.include_router(results_router)
    from src.api_pkg.routers.pipeline import router as pipeline_router

    app.include_router(pipeline_router)
    from src.api_pkg.routers.admin import router as admin_router

    app.include_router(admin_router)

    return app


app = create_app()
