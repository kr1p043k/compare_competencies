"""Нормализация навыков должна быть идемпотентной.

normalize(normalize(x)) == normalize(x): повторная нормализация уже
канонического имени не должна его менять (иначе расходятся БД/счётчики).
"""
from __future__ import annotations

import pytest

from src.parsing.skills.skill_normalizer import SkillNormalizer

SAMPLES = [
    "Python 3",
    "JS",
    "1C",
    "1С",
    "PostgreSQL",
    "React.js",
    "C++",
    "machine learning",
    "api",
    "Apache Kafka",
    "Docker Compose",
    "REST",
    "c#",
    "MS SQL",
    "Hadoop",
    "hugging face",
    "React",
    "Vue",
    "Kubernetes",
    "SQL",
]


@pytest.mark.parametrize("raw", SAMPLES)
def test_normalize_is_idempotent(raw: str) -> None:
    once = SkillNormalizer.normalize(raw)
    assert once.is_ok(), (raw, once)
    first = once.unwrap()

    twice = SkillNormalizer.normalize(first)
    assert twice.is_ok(), (first, twice)
    assert twice.unwrap() == first, (raw, first, twice.unwrap())


def test_normalize_idempotent_node_family() -> None:
    for raw in ("node", "nodejs", "node.js", "node js", "Node.js"):
        first = SkillNormalizer.normalize(raw).unwrap()
        assert SkillNormalizer.normalize(first).unwrap() == first, raw
