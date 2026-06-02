"""Lightweight model registry — хранит метаданные о версиях моделей."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src import Err, Ok, Result
from src.errors import DomainError

logger = structlog.get_logger(__name__)


@dataclass
class ModelRecord:
    name: str
    version: str
    path: Path
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, list[ModelRecord]] = {}

    def register(self, record: ModelRecord) -> Result[None, DomainError]:
        if record.name not in self._models:
            self._models[record.name] = []
        self._models[record.name].append(record)
        logger.info("model_registered", name=record.name, version=record.version)
        return Ok(None)

    def get(self, name: str, version: str | None = None) -> Result[ModelRecord, DomainError]:
        records = self._models.get(name)
        if not records:
            return Err(DomainError(message=f"model not found: {name}"))
        if version:
            for r in records:
                if r.version == version:
                    return Ok(r)
            return Err(DomainError(message=f"version {version} not found for {name}"))
        return Ok(records[-1])

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def list_versions(self, name: str) -> list[str]:
        records = self._models.get(name, [])
        return [r.version for r in records]

    @property
    def count(self) -> int:
        return sum(len(v) for v in self._models.values())
