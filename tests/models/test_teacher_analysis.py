from __future__ import annotations

from datetime import datetime

import pytest

from src.models.teacher_analysis import (
    CompetencyCoverage,
    CrossReference,
    DirectionSummary,
    DisciplineCoverage,
    GapAnalysisResult,
    Recommendation,
    SkillMatch,
    TeacherAnalysisReport,
)


class TestSkillMatch:
    def test_minimal(self):
        sm = SkillMatch(skill_name="python")
        assert sm.skill_name == "python"
        assert sm.frequency == 0
        assert sm.market_match is None
        assert sm.match_type == "exact"

    def test_full(self):
        sm = SkillMatch("python", frequency=100, market_match="python", match_type="exact")
        assert sm.frequency == 100
        assert sm.match_type == "exact"


class TestCrossReference:
    def test_create(self):
        cr = CrossReference("docker", 50, "devops")
        assert cr.skill_name == "docker"
        assert cr.frequency == 50
        assert cr.discipline == "devops"


class TestCompetencyCoverage:
    def test_minimal(self):
        cc = CompetencyCoverage(code="C1")
        assert cc.code == "C1"
        assert cc.total_skills == 0
        assert cc.gap_skills == []

    def test_full(self):
        cc = CompetencyCoverage("C1", total_skills=5, matched_skills=3, coverage=0.6, gap_skills=["a", "b"])
        assert cc.coverage == 0.6
        assert len(cc.gap_skills) == 2


class TestDisciplineCoverage:
    def test_minimal(self):
        dc = DisciplineCoverage(discipline_id="d1", discipline_name="CS")
        assert dc.coverage_level == "low"
        assert dc.coverage_ratio == 0.0

    def test_full(self):
        sms = [SkillMatch("python", 100)]
        dc = DisciplineCoverage(
            discipline_id="d1",
            discipline_name="CS",
            total_skills=10,
            market_matched=7,
            gaps=3,
            coverage_ratio=0.7,
            coverage_level="high",
            top_matched=sms,
            gaps_list=["gap1"],
            emerging=[SkillMatch("docker", 50)],
            truly_missing=[SkillMatch("k8s", 30)],
            cross_references=[CrossReference("docker", 50, "Math")],
            competencies=[CompetencyCoverage("C1", 5, 3, 0.6)],
        )
        assert dc.coverage_ratio == 0.7
        assert len(dc.top_matched) == 1
        assert dc.top_matched[0].skill_name == "python"
        assert dc.gaps == 3


class TestRecommendation:
    def test_create(self):
        r = Recommendation(type="update_content", priority="high", message="Update curriculum")
        assert r.type == "update_content"
        assert r.priority == "high"
        assert r.message == "Update curriculum"


class TestDirectionSummary:
    def test_minimal(self):
        ds = DirectionSummary(direction_code="09.03.04", direction_name="SE", profile="dev")
        assert ds.average_coverage == 0.0
        assert ds.total_disciplines == 0

    def test_full(self):
        ds = DirectionSummary(
            direction_code="09.03.04",
            direction_name="SE",
            profile="dev",
            total_disciplines=5,
            average_coverage=0.65,
            total_gaps=12,
            top_cross_discipline_gaps=[{"skill": "docker", "freq": 80}],
            top_emerging=[{"skill": "k8s", "freq": 90}],
            disciplines=[{"id": "d1", "name": "CS"}],
        )
        assert ds.total_disciplines == 5
        assert ds.average_coverage == 0.65


class TestGapAnalysisResult:
    def test_empty(self):
        g = GapAnalysisResult()
        assert g.discipline is None
        assert g.recommendations == []

    def test_with_data(self):
        dc = DisciplineCoverage("d1", "CS")
        rec = Recommendation("update", "high", "msg")
        g = GapAnalysisResult(discipline=dc, recommendations=[rec])
        assert g.discipline is not None
        assert len(g.recommendations) == 1


class TestTeacherAnalysisReport:
    def test_create(self):
        ds = DirectionSummary("09.03.04", "SE", "dev")
        r = TeacherAnalysisReport(direction=ds)
        assert r.direction.direction_code == "09.03.04"
        assert r.discipline_reports == []
        assert r.generated_at == ""
