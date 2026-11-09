"""Health, readiness и логи."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field, model_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

import src.db as _db
from src import config
from src.api_pkg import deps
from src.model_registry import ModelRegistry
from src.models.api_responses import HealthResponse

logger = structlog.get_logger("api")
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(tags=["monitoring"])


@router.get("/api/version")
async def api_version() -> dict[str, str]:
    """Версия публичного API (контракта). Политика - docs/API-CHANGELOG.md."""
    return {"api_version": config.API_VERSION}


_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


@router.get("/")
async def root():
    """Корневой health-check."""
    v = config.VERSION if hasattr(config, "VERSION") else "2.0"
    return {
        "service": "Compare Competencies API",
        "version": v,
        "docs": "/docs",
        "health": "/health",
        "status": "/api/status",
    }


class LogEntry(BaseModel):
    # level - только известные значения: иначе в лог можно писать произвольные
    # строки (мусор/подделка уровней). json.dumps ниже экранирует переводы строк.
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = Field(max_length=2000)
    data: dict | None = None
    timestamp: str | None = None

    @model_validator(mode="after")
    def _cap_data(self):
        if self.data is not None and (
            len(self.data) > 20
            or len(json.dumps(self.data, ensure_ascii=False, default=str)) > 10000
        ):
            raise ValueError("data too large")
        return self


_LOG_FILE = Path(__file__).parent.parent.parent.parent / "frontend" / "logs" / "app.log"

#: Клиентский лог раньше рос без ограничений (на проде уже 380+ КБ).
#: Держим один бэкап и начинаем заново после порога — этого хватает, чтобы
#: разобрать инцидент, но диск не забивается.
_LOG_MAX_BYTES = 2 * 1024 * 1024


def _write_log_sync(entry: LogEntry) -> None:
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "time": entry.timestamp or datetime.now(UTC).isoformat(),
            "level": entry.level,
            "message": entry.message,
            "data": entry.data,
        },
        ensure_ascii=False,
    )
    try:
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > _LOG_MAX_BYTES:
            backup = _LOG_FILE.with_suffix(".log.1")
            backup.unlink(missing_ok=True)
            _LOG_FILE.rename(backup)
    except OSError:
        # Ротация — не повод терять запись: пишем как есть.
        pass
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@router.post("/api/log")
@limiter.limit("20/minute")
async def write_log(request: Request, entry: LogEntry):
    """Запись клиентского лога."""
    try:
        await asyncio.to_thread(_write_log_sync, entry)
    except Exception as e:
        logger.error("log_write_failed", error=str(e))
    return {"ok": True}


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Готовность сервиса и зависимостей."""
    registry = _get_registry()
    ltr_model = registry.latest("ltr")
    clusters = {
        lvl: registry.latest(f"clusters_{lvl}")
        for lvl in ["junior", "middle", "senior"]
    }
    return {
        "status": "ok",
        "version": getattr(config, "VERSION", "2.0"),
        "evaluator": deps.evaluator is not None,
        "recommendation_engine": deps.recommendation_engine is not None,
        "ltr_model_version": (ltr_model or {}).get("version"),
        "clusters_versions": {
            lvl: (reg or {}).get("version") for lvl, reg in clusters.items()
        },
        "api_ready": deps.is_ready,
    }


class ReadyBody(BaseModel):
    """Ответ /ready: готовность сервиса + список проблем.

    В отличие от исторического ReadyResponse сюда добавлено поле `problems`:
    оркестратору (k8s/proxmox) и людям удобнее видеть, ЧТО именно не готово,
    а не угадывать по HTTP-коду.
    """

    status: str
    components: dict[str, bool]
    problems: list[str] = Field(default_factory=list)


@router.get("/ready", response_model=ReadyBody)
async def ready_check(response: Response) -> ReadyBody:
    """Проверка готовности критических зависимостей (fail-closed).

    Критическими считаются ТОЛЬКО две вещи:
    1) deps.is_ready - False, пока bootstrap не дочитал файл вакансий (startup.py
       `_bootstrap`), либо True после загрузки данных / входа в режим ожидания;
    2) src.db.pool - без пула БД не работают auth/session-эндпоинты.
    Всё остальное (evaluator, recommendation_engine, trend_analyzer) НЕ критично:
    _set_waiting_mode() в startup.py обнуляет их, когда файла вакансий нет, и API
    при этом считается готовым (режим ожидания) - ровнять их в 503 нельзя.
    """
    problems: list[str] = []
    if not deps.is_ready:
        problems.append("data_not_loaded")
    if _db.pool is None:
        problems.append("db_pool_unavailable")

    components = {
        "evaluator": deps.evaluator is not None,
        "recommendation_engine": deps.recommendation_engine is not None
        and deps.recommendation_engine.is_fitted,
        # Кластер теперь ленивый (deps.get_clusterer): не грузим sklearn на /ready.
        "clusterer": deps.get_clusterer().is_fitted,
        "trend_analyzer": deps.trend_analyzer is not None,
    }

    if problems:
        response.status_code = 503
        return ReadyBody(status="not_ready", components=components, problems=problems)
    return ReadyBody(status="ok", components=components, problems=problems)



