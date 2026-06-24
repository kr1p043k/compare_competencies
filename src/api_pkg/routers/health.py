"""Health, readiness, status, regions и логи."""

import json
import time
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.models.api_responses import (
    HealthResponse,
    ReadyResponse,
)

from src.api_pkg import deps
from src import config
from src.model_registry import ModelRegistry

logger = structlog.get_logger("api")

router = APIRouter(tags=["monitoring"])

limiter = Limiter(key_func=get_remote_address)


_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


@router.get("/")
async def root():
    v = config.VERSION if hasattr(config, "VERSION") else "2.0"
    return {
        "service": "Compare Competencies API",
        "version": v,
        "docs": "/docs",
        "health": "/health",
        "status": "/api/status",
    }


class LogEntry(BaseModel):
    level: str = "info"
    message: str
    data: dict | None = None
    timestamp: str | None = None


_LOG_FILE = Path(__file__).parent.parent.parent.parent / "frontend" / "logs" / "app.log"


@router.post("/api/log")
async def write_log(entry: LogEntry):
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "time": entry.timestamp or datetime.now().isoformat(),
                "level": entry.level,
                "message": entry.message,
                "data": entry.data,
            },
            ensure_ascii=False,
        )
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.error("log_write_failed", error=str(e))
    return {"ok": True}


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
async def health_check():
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


@router.get("/ready", response_model=ReadyResponse)
async def ready_check():
    """Проверка готовности всех компонентов."""
    components = {
        "evaluator": deps.evaluator is not None,
        "recommendation_engine": deps.recommendation_engine is not None
        and deps.recommendation_engine.is_fitted,
        "clusterer": deps.clusterer.is_fitted,
        "trend_analyzer": deps.trend_analyzer is not None,
    }
    ready = all(components.values())
    status = "ready" if ready else "not ready"
    return {"status": status, "components": components}



