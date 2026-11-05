"""API package - фабрика FastAPI приложения."""

import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog

# Suppress cmdstanpy BEFORE any imports that might trigger Prophet
from cmdstanpy.utils.logging import disable_logging as _disable_cmdstan
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src import config
from src.api_pkg.client_ip import client_ip
from src.monitoring.metrics import get_metrics

_disable_cmdstan().__enter__()

from src.api_pkg import deps as deps  # noqa: F401,E402

logger = structlog.get_logger("api")


class _BodyTooLargeError(Exception):
    """Тело превысило лимит при стриминге (без Content-Length)."""


def _rate_limit_key(request: Request) -> str:
    """Rate limit by user email (from JWT) if authenticated, else by real IP.

    IP берётся через `client_ip`, который не доверяет `X-Forwarded-For` от
    недоверенных соединений: иначе лимит обходится подменой заголовка
    (15 запросов с разными XFF давали 15 попыток входа).
    """
    user = getattr(request.state, "user", None)
    if user and isinstance(user, dict):
        email = user.get("u")
        if email:
            return f"user:{email}"
    return f"ip:{client_ip(request)}"


limiter = Limiter(key_func=_rate_limit_key)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from src.api_pkg.startup import run_startup

        await run_startup(app)
        yield
        from src.database import get_engine
        from src.db import close_pool
        from src.infrastructure.hub_publications import close_client as close_hub_client

        await close_hub_client()
        await close_pool()
        await get_engine().dispose()
        logger.info("API shutting down...")

    app = FastAPI(
        title="Competency Analyzer API",
        version="2.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    def _ratelimit_exceeded_handler(request, exc: RateLimitExceeded):
        """429 с Retry-After: без заголовка клиент не знает, когда повторять.

        Период берём из самого лимита (10/minute -> 60 c); если разобрать не
        удалось, консервативно отдаём 60.
        """
        retry_after = 60
        try:
            retry_after = int(exc.limit.limit.get_expiry())
        except Exception:
            logger.debug("ratelimit_expiry_unavailable", exc_info=True)
        logger.warning(
            "ratelimit_exceeded",
            path=getattr(getattr(request, "url", None), "path", None),
            retry_after=retry_after,
        )
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."},
            headers={"Retry-After": str(retry_after)},
        )

    app.add_exception_handler(RateLimitExceeded, _ratelimit_exceeded_handler)

    allowed_origins = (
        config.ALLOWED_ORIGINS.split(",") if config.ALLOWED_ORIGINS != "*" else ["*"]
    )
    if allowed_origins == ["*"]:
        # В проде '*' значит, что любой сайт читает API от имени пользователя:
        # падаем на старте, а не отдаём данные кому попало.
        _prod = os.getenv("APP_ENV", "").lower() in ("prod", "production")
        if _prod:
            raise RuntimeError(
                "ALLOWED_ORIGINS='*' недопустим в продакшене: перечислите домены через запятую"
            )
        logger.warning("cors_allow_all_origins", hint="Задайте ALLOWED_ORIGINS списком доменов")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def limit_body_size_middleware(request: Request, call_next):
        limit = config.MAX_REQUEST_SIZE
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    logger.warning("request_body_too_large", size=int(content_length))
                    return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})

        # Chunked/стриминговое тело приходит БЕЗ Content-Length - проверка выше
        # его не ловит. Оборачиваем приём и рвём соединение, как только тело
        # перевалило лимит: иначе клиент шлёт мегабайты мимо проверки (память).
        # Эндпоинты с моделью тела на обрыв ответят 400 «error parsing the body»;
        # если тело читается без валидирующей модели - отдаём 413 ниже.
        received = 0
        original_receive = request._receive

        async def _limited_receive():
            nonlocal received
            message = await original_receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLargeError
            return message

        request._receive = _limited_receive

        def _contains_too_large(exc: BaseException) -> bool:
            if isinstance(exc, _BodyTooLargeError):
                return True
            subs = getattr(exc, "exceptions", None)
            return bool(subs) and any(_contains_too_large(e) for e in subs)

        try:
            return await call_next(request)
        except BaseExceptionGroup as eg:
            # BaseHTTPMiddleware прогоняет эндпоинт в task-group: наше исключение
            # прилетает завёрнутым в BaseExceptionGroup, поэтому ловим и его.
            if _contains_too_large(eg):
                logger.warning("request_body_too_large_streaming", size=received)
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})
            raise
        except _BodyTooLargeError:
            logger.warning("request_body_too_large_streaming", size=received)
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})

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

    @app.middleware("http")
    async def add_security_headers_middleware(request: Request, call_next):
        """Заголовки безопасности на все ответы API.

        До этого отдавали только X-Request-ID/X-Process-Time: как и в аудите
        gigachess, не было ни HSTS, ни X-Frame-Options, ни nosniff, ни
        Referrer-Policy. CSP не ставим: /docs рендерит Swagger с инлайн-скриптами,
        а фронтенд раздаётся отдельным сервером - ломать его нам нечем.
        """
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled exception", request_id=request_id, error_type=type(exc).__name__)
        if request.url.path.startswith("/api/"):
            request.state.system_error_notified = True
            from src.notifications.system import queue_system_error
            queue_system_error(
                f"Необработанная ошибка API: {type(exc).__name__}",
                f"{request.method} {request.url.path} | request_id={request_id} | {str(exc)[:1500]}",
                severity="error",
                article_url=f"api:{request.method} {request.url.path}",
                dedupe=True,
            )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
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
        # headers пробрасываем: 429 от lockout несёт Retry-After, без этого
        # клиент не знает, когда повторять, и долбит сервер в цикле.
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": request_id},
            headers=exc.headers or None,
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
    from src.api_pkg.routers.results import router as results_router
    _mount(results_router)
    from src.api_pkg.routers.pipeline import router as pipeline_router
    _mount(pipeline_router)
    from src.api_pkg.routers.admin import router as admin_router
    _mount(admin_router)
    from src.api_pkg.routers.admin import ws_log_router
    _mount(ws_log_router)
    from src.api_pkg.routers.forecast import router as forecast_router
    _mount(forecast_router)
    from src.api_pkg.routers.trends_by_profession import router as trends_prof_router
    _mount(trends_prof_router)
    from src.api_pkg.routers.auth import router as auth_router
    _mount(auth_router)
    from src.api_pkg.routers.academic import router as academic_router
    _mount(academic_router)
    from src.api_pkg.routers.publications import router as publications_router
    _mount(publications_router)
    from src.api_pkg.routers.academic_trends import router as academic_trends_router
    _mount(academic_trends_router)
    from src.api_pkg.routers.comparison import router as intake_comparison_router
    _mount(intake_comparison_router)
    from src.api_pkg.routers.history import router as history_router
    _mount(history_router)
    from src.api_pkg.routers.security import router as security_router
    _mount(security_router)
    from src.api_pkg.routers.teacher import router as teacher_router
    _mount(teacher_router)
    from src.api_pkg.routers.rpd import router as rpd_router
    _mount(rpd_router)
    from src.api_pkg.routers.zun import router as zun_router
    _mount(zun_router)
    from src.api_pkg.routers.krm_teacher import router as krm_teacher_router
    _mount(krm_teacher_router)
    from src.api_pkg.routers.krm_student import router as krm_student_router
    _mount(krm_student_router)
    from src.api_pkg.routers.student import router as student_router
    _mount(student_router)
    from src.api_pkg.routers.llm import router as llm_router
    _mount(llm_router)

    from src.api_pkg.routers.subscriptions import router as subscriptions_router
    _mount(subscriptions_router)

    from src.n8n.webhooks import router as n8n_webhook_router
    app.include_router(n8n_webhook_router)  # no versioning

    from src.api_pkg.request_logger import RequestLogMiddleware

    app.add_middleware(RequestLogMiddleware)

    from src.n8n.auth import N8NAuthMiddleware

    app.add_middleware(N8NAuthMiddleware)

    from src.api_pkg.routers.auth import require_any_role

    async def _metrics_access(request: Request) -> None:
        """Доступ к /metrics: сервисный токен Prometheus либо роль admin.

        Prometheus не умеет логиниться, поэтому при заданном METRICS_TOKEN он
        присылает его в Authorization. Остальные клиенты проходят обычную
        проверку роли, как и раньше.
        """
        expected = config.METRICS_TOKEN
        if expected is not None:
            import secrets as _secrets

            supplied = request.headers.get("authorization", "")
            if _secrets.compare_digest(supplied, f"Bearer {expected.get_secret_value()}"):
                return
        await require_any_role("admin")(request)

    @app.get("/metrics", dependencies=[Depends(_metrics_access)])
    async def metrics():
        data, content_type = get_metrics()
        from fastapi.responses import Response
        return Response(content=data, media_type=content_type)

    return app


app = create_app()
