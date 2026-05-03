# tests/analyzers/test_gap_analyzer.py
import pytest
import numpy as np
from src.analyzers.gap_analyzer import GapAnalyzer


class TestGapAnalyzerExtended:
    @pytest.fixture
    def sample_weights_by_level(self):
        return {
            'junior': {'python': 0.8, 'sql': 0.6, 'git': 0.5, 'html': 0.4},
            'middle': {'python': 0.9, 'docker': 0.7, 'sql': 0.5, 'fastapi': 0.4},
            'senior': {'python': 0.9, 'docker': 0.9, 'k8s': 0.8, 'sql': 0.3},
        }

    def test_init_with_empty_weights(self):
        ga = GapAnalyzer({})
        assert ga.skill_weights == {}

    def test_compute_metrics_returns_skills(self, sample_weights_by_level):
        ga = GapAnalyzer(sample_weights_by_level)
        user_skills = ["python", "sql"]
        user_levels = {"python": 0.7, "sql": 0.4}
        metrics = ga.compute_metrics(user_skills, user_levels)

        assert len(metrics) > 0
        for skill, metric in metrics.items():
            assert hasattr(metric, 'gap_j')
            assert hasattr(metric, 'gap_m')
            assert hasattr(metric, 'gap_s')
            assert metric.gap_j >= 0
            assert metric.gap_m >= 0
            assert metric.gap_s >= 0

    def test_compute_metrics_gap_calculation(self, sample_weights_by_level):
        """Gap = max(0, market_weight - user_level)"""
        ga = GapAnalyzer(sample_weights_by_level)
        user_skills = ["python"]
        user_levels = {"python": 0.3}
        metrics = ga.compute_metrics(user_skills, user_levels)

        assert "python" in metrics
        assert metrics["python"].gap_j == pytest.approx(0.5)
        assert metrics["python"].gap_m == pytest.approx(0.6)

    def test_compute_metrics_no_gap_when_full_coverage(self, sample_weights_by_level):
        ga = GapAnalyzer(sample_weights_by_level)
        user_skills = ["python"]
        user_levels = {"python": 1.0}
        metrics = ga.compute_metrics(user_skills, user_levels)

        assert metrics["python"].gap_j == 0.0
        assert metrics["python"].gap_m == 0.0
        assert metrics["python"].gap_s == 0.0

    def test_compute_metrics_empty_user(self, sample_weights_by_level):
        ga = GapAnalyzer(sample_weights_by_level)
        metrics = ga.compute_metrics([], {})
        assert len(metrics) > 0
        for skill, metric in metrics.items():
            if skill == "python":
                assert metric.gap_j == pytest.approx(0.8)
                assert metric.gap_m == pytest.approx(0.9)

    def test_set_weights_by_level(self, sample_weights_by_level):
        ga = GapAnalyzer({})
        ga.set_weights_by_level(sample_weights_by_level)
        assert ga.skill_weights_by_level == sample_weights_by_level

    def test_set_weights_by_level_after_init(self, sample_weights_by_level):
        ga = GapAnalyzer(sample_weights_by_level)
        # skill_weights хранит позиционный аргумент
        assert ga.skill_weights == sample_weights_by_level
        # skill_weights_by_level ещё не установлен
        assert not hasattr(ga, 'skill_weights_by_level')
        # Устанавливаем явно
        ga.set_weights_by_level(sample_weights_by_level)
        assert ga.skill_weights_by_level == sample_weights_by_level