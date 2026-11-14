"""Бизнес-метрика латентности матчинга (Зона D, 7.3.1).

Проверяем, что метрика экспонируется в /metrics и что вызов матчинга её
наблюдает (инструментирование реально работает, а не только объявлено).
"""
from __future__ import annotations

from src.monitoring.metrics import generate_latest, matching_latency


def test_matching_latency_is_exposed() -> None:
    body = generate_latest().decode("utf-8")
    assert "skill_matching_duration_seconds" in body


def _observed() -> int:
    for metric in matching_latency.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                return int(sample.value)
    return 0


def test_match_observes_latency() -> None:
    from src.analyzers.skill_matcher import SkillMatcher

    before = _observed()
    SkillMatcher().match("python")
    assert _observed() == before + 1
