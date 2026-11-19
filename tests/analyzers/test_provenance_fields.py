"""Provenance навыков (BACKLOG 2.3).

Контракт: `SkillMatch` несёт `source` (откуда требование: ksa/whitelist) и
`market_source` (откуда рынок: parsed_skills), а `DisciplineCoverage` —
`gaps_detail` со `source` по каждому пробелу.
"""

from __future__ import annotations

from src.models.teacher_analysis import DisciplineCoverage, SkillMatch


def test_skill_match_has_provenance_fields() -> None:
    m = SkillMatch(skill_name="python", market_match="python")
    assert m.source == "ksa"
    assert m.market_source == "parsed_skills"

    ksa = SkillMatch(skill_name="1с", source="ksa")
    canon = SkillMatch(skill_name="postgresql", source="whitelist")
    assert ksa.source != canon.source


def test_discipline_coverage_gaps_detail() -> None:
    cov = DisciplineCoverage(discipline_id="d1", discipline_name="БД")
    assert cov.gaps_detail == []

    cov = DisciplineCoverage(
        discipline_id="d1",
        discipline_name="БД",
        gaps_list=["kafka"],
        gaps_detail=[{"skill": "kafka", "source": "whitelist"}],
    )
    assert cov.gaps_detail[0]["source"] == "whitelist"
