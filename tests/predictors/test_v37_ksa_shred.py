"""Unit (v37): bullet-strip, conditional shreds, ksa-type routing, matched names."""
from src.analyzers.coverage_analyzer import CoverageAnalyzer
from src.analyzers.skill_matcher import SkillMatcher
from src.models.teacher_analysis import CompetencyCoverage, DisciplineCoverage
from src.pipeline.teacher_analysis_runner import (
    _is_skill_like_ksa,
    _strip_bullet,
)
from src.predictors.curriculum_recommender import CurriculumRecommender


def test_strip_bullet():
    assert _strip_bullet("- Технические средства защиты информации") == \
        "Технические средства защиты информации"
    assert _strip_bullet("python") == "python"
    assert _strip_bullet("  - spaced") == "spaced"


def test_conditional_shreds_rejected():
    assert _is_skill_like_ksa("- при наличии в соответствующем ФГОС/ОС ЮФУ") is False
    assert _is_skill_like_ksa("при наличии в стандарте") is False
    assert _is_skill_like_ksa("Технические средства защиты информации") is True


def test_ksa_knowledge_routes_foundational():
    r = CurriculumRecommender()
    cov = DisciplineCoverage(
        discipline_id="t1", discipline_name="test",
        gaps_list=['положение военной доктрины'],
        competencies=[], coverage_ratio=0.0,
        ksa_types={'положение военной доктрины': 'knowledge'},
    )
    recs = r.generate(cov, cooc=None).ok()
    found = [x for x in recs if x.skill_name == 'положение военной доктрины']
    assert found and found[0].type == 'foundational'


def test_ksa_ability_keeps_text_routing():
    r = CurriculumRecommender()
    cov = DisciplineCoverage(
        discipline_id="t1", discipline_name="test",
        gaps_list=['положение военной доктрины'],
        competencies=[], coverage_ratio=0.0,
        ksa_types={'положение военной доктрины': 'abilities'},
    )
    recs = r.generate(cov, cooc=None).ok()
    found = [x for x in recs if x.skill_name == 'положение военной доктрины']
    assert found and found[0].type in ('foundational', 'review_content')


def test_matched_names_populated():
    m = SkillMatcher(market_skills={'python': 3016, 'sql': 3519})
    cov = CoverageAnalyzer(m).analyze_discipline(
        'd1', 'Test', {'K1': ['python level', 'sql', 'unknown shred xyz']}).unwrap()
    cc = [c for c in cov.competencies if c.code == 'K1'][0]
    assert cc.matched_skills == 2
    assert 'sql' in cc.matched_names and 'python level' in cc.matched_names
    assert 'unknown shred xyz' not in cc.matched_names
