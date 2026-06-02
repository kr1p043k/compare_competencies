"""Lightweight experiment tracker — логирует метрики и параметры в JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, Result
from src.errors import DomainError
from src.utils import atomic_write_json

logger = structlog.get_logger(__name__)


@dataclass
class ExperimentRun:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class ExperimentTracker:
    def __init__(self, log_dir: Path | None = None):
        self._current: ExperimentRun | None = None
        self._log_dir = log_dir

    def start(self, name: str, params: dict[str, Any] | None = None, tags: dict[str, str] | None = None) -> Result[None, DomainError]:
        if self._current is not None:
            return Err(DomainError(message="experiment already running", detail=self._current.name))
        self._current = ExperimentRun(name=name, params=params or {}, tags=tags or {})
        logger.info("experiment_started", name=name)
        return Ok(None)

    def log_metric(self, key: str, value: float) -> Result[None, DomainError]:
        if self._current is None:
            return Err(DomainError(message="no active experiment"))
        self._current.metrics[key] = value
        logger.debug("metric_logged", name=self._current.name, key=key, value=value)
        return Ok(None)

    def log_params(self, params: dict[str, Any]) -> Result[None, DomainError]:
        if self._current is None:
            return Err(DomainError(message="no active experiment"))
        self._current.params.update(params)
        return Ok(None)

    def finish(self) -> Result[ExperimentRun, DomainError]:
        if self._current is None:
            return Err(DomainError(message="no active experiment"))
        self._current.finished_at = datetime.now(UTC).isoformat()
        run = self._current
        self._persist(run)
        self._current = None
        logger.info("experiment_finished", name=run.name, metrics=run.metrics)
        return Ok(run)

    def _persist(self, run: ExperimentRun) -> None:
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"run_{run.name}_{run.started_at.replace(':', '-')}.json"
        atomic_write_json(run.to_dict(), path)
        logger.debug("experiment_persisted", path=str(path))
