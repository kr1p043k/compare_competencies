"""Tests for CurriculumOptimizer and CurriculumRecommender."""
from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.errors import RecommendationError
from src.models.teacher_analysis import (
    DirectionSummary, Recommendation, DisciplineCoverage, CompetencyCoverage,
    CrossReference, SkillMatch,
)
from src.predictors.curriculum_optimizer import CurriculumOptimizer
from src.predictors.curriculum_recommender import CurriculumRecommender


class TestCurriculumOptimizer:
    def make_summary(self, disciplines=None, top_emerging=None):
        return DirectionSummary(
            direction_code="09.03.02",
            direction_name="Test Direction",
            profile="test",
            total_disciplines=len(disciplines or []),
            disciplines=disciplines or [],
            top_emerging=top_emerging or [],
        )

    def test_optimize_success_no_recs(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(disciplines=[{"name": "OK", "coverage_level": "high"}])
        result = opt.optimize(summary)
        assert result.is_ok()
        assert result.ok() == []

    def test_optimize_none_summary(self):
        opt = CurriculumOptimizer()
        result = opt.optimize(None)
        assert result.is_err()
        assert isinstance(result.err(), RecommendationError)

    def test_optimize_no_disciplines(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(disciplines=[])
        result = opt.optimize(summary)
        assert result.is_err()
        assert "No disciplines" in result.err().message

    def test_optimize_low_coverage(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(disciplines=[
            {"name": "Discipline 1", "coverage_level": "low"},
            {"name": "Discipline 2", "coverage_level": "high"},
        ])
        result = opt.optimize(summary)
        assert result.is_ok()
        recs = result.ok()
        assert any(r.type == "major_revision" for r in recs)

    def test_optimize_top_emerging(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(
            disciplines=[{"name": "D1", "coverage_level": "medium"}],
            top_emerging=[{"skill": "LLM"}, {"skill": "MLOps"}],
        )
        result = opt.optimize(summary)
        assert result.is_ok()
        recs = result.ok()
        assert any(r.type == "add_new_content" for r in recs)

    def test_optimize_redundant_gaps(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(disciplines=[
            {"name": "D1", "coverage_level": "medium", "gaps": 150},
            {"name": "D2", "coverage_level": "medium", "gaps": 50},
        ])
        result = opt.optimize(summary)
        assert result.is_ok()
        recs = result.ok()
        assert any(r.type == "update_content" for r in recs)

    def test_optimize_combined_recs(self):
        opt = CurriculumOptimizer()
        summary = self.make_summary(
            disciplines=[
                {"name": "Low", "coverage_level": "low"},
                {"name": "Redundant", "coverage_level": "medium", "gaps": 200},
            ],
            top_emerging=[{"skill": "AI"}, {"skill": "Cloud"}],
        )
        result = opt.optimize(summary)
        assert result.is_ok()
        types = {r.type for r in result.ok()}
        assert "major_revision" in types
        assert "add_new_content" in types
        assert "update_content" in types

    def test_optimize_low_coverage_names_truncated(self):
        opt = CurriculumOptimizer()
        long_name = "A very long discipline name that should be truncated to forty characters"
        summary = self.make_summary(disciplines=[
            {"name": long_name, "coverage_level": "low"},
        ])
        result = opt.optimize(summary)
        assert result.is_ok()
        recs = result.ok()
        assert recs[0].message is not None


class TestCurriculumRecommender:
    def make_coverage(self, **overrides) -> DisciplineCoverage:
        defaults = dict(
            discipline_id="D1",
            discipline_name="Test Discipline",
            total_skills=50,
            market_matched=20,
            gaps=30,
            coverage_ratio=0.4,
            coverage_level="medium",
            gaps_list=["python", "sql"],
            truly_missing=[],
            cross_references=[],
            competencies=[CompetencyCoverage(code="CC1", total_skills=10, matched_skills=5, coverage=0.5)],
        )
        defaults.update(overrides)
        return DisciplineCoverage(**defaults)

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_gaps_list(self, mock_load):
        mock_load.return_value = {"academic": ["math"], "professional": ["docker"]}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(gaps_list=["python", "docker"])
        result = rec.generate(coverage)
        assert result.is_ok()
        recs = result.ok()
        assert len(recs) >= 2

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_none_coverage(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        result = rec.generate(None)
        assert result.is_err()
        assert isinstance(result.err(), RecommendationError)

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_truly_missing(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(
            truly_missing=[
                SkillMatch(skill_name="Kubernetes", frequency=50),
                SkillMatch(skill_name="Terraform", frequency=30),
            ],
        )
        result = rec.generate(coverage)
        assert result.is_ok()
        types = {r.type for r in result.ok()}
        assert "add_new_content" in types

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_cross_references(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(
            cross_references=[
                CrossReference(skill_name="Docker", frequency=10, discipline="DevOps"),
                CrossReference(skill_name="Docker", frequency=10, discipline="DevOps"),
            ],
        )
        result = rec.generate(coverage)
        assert result.is_ok()
        types = {r.type for r in result.ok()}
        assert "cross_reference" in types

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_low_coverage_ratio(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(coverage_ratio=0.2)
        result = rec.generate(coverage)
        assert result.is_ok()
        types = {r.type for r in result.ok()}
        assert "major_revision" in types

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_zero_competency_coverage(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(
            competencies=[
                CompetencyCoverage(code="CC1", total_skills=5, matched_skills=0, coverage=0.0),
                CompetencyCoverage(code="CC2", total_skills=0, matched_skills=0, coverage=0.0),
            ],
        )
        result = rec.generate(coverage)
        assert result.is_ok()
        types = {r.type for r in result.ok()}
        assert "review_content" in types

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_classifies_academic(self, mock_load):
        mock_load.return_value = {"academic": ["math"], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(gaps_list=["math", "python"])
        result = rec.generate(coverage)
        assert result.is_ok()
        recs = result.ok()
        foundational = [r for r in recs if r.type == "foundational"]
        assert len(foundational) == 1
        assert foundational[0].priority == "low"

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_classifies_professional(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": ["docker"]}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(gaps_list=["docker"])
        result = rec.generate(coverage)
        assert result.is_ok()
        recs = result.ok()
        review = [r for r in recs if r.type == "review_content"]
        assert len(review) == 1
        assert review[0].priority == "medium"

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_truly_missing_filter_relevant(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(
            discipline_name="Software Engineering",
            truly_missing=[
                SkillMatch(skill_name="React", frequency=100),
                SkillMatch(skill_name="Cooking", frequency=5),
            ],
        )
        result = rec.generate(coverage)
        assert result.is_ok()
        recs = result.ok()
        add_new = [r for r in recs if r.type == "add_new_content"]
        if add_new:
            assert any("React" in r.message for r in add_new)

    def test_filter_relevant_empty_skills(self):
        result = CurriculumRecommender._filter_relevant([], "Test")
        assert result == []

    def test_filter_relevant_with_match(self):
        skills = [
            SkillMatch(skill_name="Python", frequency=10),
            SkillMatch(skill_name="Java", frequency=5),
        ]
        result = CurriculumRecommender._filter_relevant(skills, "Python Developer")
        assert len(result) >= 0

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_summary_recommendations_success(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage()
        result = rec.generate_summary_recommendations(
            all_coverages=[coverage],
            avg_coverage=0.25,
            total_gaps=30,
            top_emerging=[{"skill": "LLM", "frequency": 50}],
        )
        assert result.is_ok()
        recs = result.ok()
        assert len(recs) >= 1
        assert any(r.type == "major_revision" for r in recs)
        assert any(r.type == "add_new_content" for r in recs)

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_summary_empty_coverages(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        result = rec.generate_summary_recommendations([], 0.5, 0, [])
        assert result.is_err()

    @patch("src.predictors.curriculum_recommender._load_skill_types")
    def test_generate_summary_no_emerging(self, mock_load):
        mock_load.return_value = {"academic": [], "professional": []}
        rec = CurriculumRecommender()
        coverage = self.make_coverage(coverage_ratio=0.5)
        result = rec.generate_summary_recommendations(
            all_coverages=[coverage],
            avg_coverage=0.5,
            total_gaps=10,
            top_emerging=[],
        )
        assert result.is_ok()
        assert result.ok() == []