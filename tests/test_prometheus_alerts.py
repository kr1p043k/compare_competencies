"""Правила алертов Prometheus: валидный YAML, ожидаемые алерты, живые метрики.

Ловит опечатку в имени метрики в expr (алерт молча никогда не сработает) и
потерю правил при правках.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALERTS = ROOT / "prometheus" / "alerts.yml"
METRICS_SRC = ROOT / "src" / "monitoring" / "metrics.py"

EXPECTED = {
    "HighErrorRate",
    "HighLatencyP95",
    "PipelineJobFailed",
    "DatabaseUnavailable",
    "HighMatchingLatency",
    "LLMErrorRate",
}

# Метрики, которые алерты используют как «свои» (определены в приложении).
APP_METRICS = {
    "api_requests_total",
    "api_request_duration_seconds",
    "pipeline_runs_total",
    "skill_matching_duration_seconds",
    "llm_requests_total",
}


def _rules() -> list[dict]:
    doc = yaml.safe_load(ALERTS.read_text(encoding="utf-8"))
    out: list[dict] = []
    for group in doc.get("groups", []):
        out.extend(group.get("rules", []))
    return out


def test_alert_names_present() -> None:
    names = {r.get("alert") for r in _rules()}
    missing = sorted(EXPECTED - names)
    assert not missing, f"Пропали правила алертов: {missing}"


def test_every_alert_has_expr_and_annotations() -> None:
    bad = [r.get("alert") for r in _rules() if not r.get("expr") or not r.get("annotations")]
    assert not bad, f"Алерты без expr/annotations: {bad}"


def test_referenced_app_metrics_are_defined() -> None:
    metrics_src = METRICS_SRC.read_text(encoding="utf-8")
    missing = sorted(m for m in APP_METRICS if m not in metrics_src)
    assert not missing, f"Алерты ссылаются на несуществующие метрики: {missing}"
