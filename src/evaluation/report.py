from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricEntry:
    name: str
    value: float
    threshold: float | None = None
    passed: bool | None = None

    def __post_init__(self):
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold


@dataclass
class EvaluationReport:
    model_name: str
    metrics: list[MetricEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, name: str, value: float, threshold: float | None = None) -> None:
        self.metrics.append(MetricEntry(name=name, value=value, threshold=threshold))

    def add_dict(self, metrics: dict[str, float], threshold: float | None = None) -> None:
        for name, value in metrics.items():
            self.add(name, value, threshold)

    def summary(self) -> dict[str, Any]:
        total = len(self.metrics)
        passed = sum(1 for m in self.metrics if m.passed is True)
        failed = sum(1 for m in self.metrics if m.passed is False)
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "total_metrics": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total else 1.0,
            "metrics": {m.name: {"value": m.value, "threshold": m.threshold, "passed": m.passed} for m in self.metrics},
        }

    def log(self) -> None:
        s = self.summary()
        logger.info(
            "evaluation_report",
            model=self.model_name,
            pass_rate=round(s["pass_rate"], 3),
            passed=s["passed"],
            total=s["total_metrics"],
        )
        for m in self.metrics:
            status = "✓" if m.passed else "✗" if m.passed is False else "~"
            logger.info(f"  {status} {m.name}={m.value:.4f}" + (f" (threshold={m.threshold})" if m.threshold else ""))
