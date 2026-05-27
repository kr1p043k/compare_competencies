"""PipelineStage abstraction — единый интерфейс для всех этапов пайплайна."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import structlog

from src.pipeline.progress import write as write_progress
from src.result import Result

T = TypeVar("T")
logger = structlog.get_logger(__name__)


class PipelineStage(ABC, Generic[T]):
    name: str = ""
    pct_range: tuple[int, int] = (0, 100)

    @abstractmethod
    def run(self, **kwargs) -> Result[T, Any]:
        ...

    def validate(self) -> Result[bool, Any]:
        return Result.Ok(True)

    def rollback(self) -> None:
        pass

    def _progress(self, pct: int, msg: str):
        base = self.pct_range[0]
        span = self.pct_range[1] - self.pct_range[0]
        write_progress(base + int(pct / 100 * span), msg)
