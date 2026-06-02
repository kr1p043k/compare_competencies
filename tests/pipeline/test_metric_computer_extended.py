from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.pipeline.metric_computer import MetricComputer


@pytest.fixture
def mc():
    return MetricComputer({}, [], [], {})


class TestMetricComputerPrepare:
    def test_prepare_ok(self, mc):
        result = mc.prepare({})
        assert result.is_ok()
        assert mc.evaluator is not None

    def test_prepare_exception(self, mc):
        with patch("src.pipeline.metric_computer.ProfileEvaluator",
                   side_effect=ValueError("init error")):
            result = mc.prepare({})
        assert result.is_err()

    def test_prepare_passes_args(self, mc):
        skill_weights = {"py": 1.0}
        vac_skills = [["py"]]
        level_data = [{"skills": ["py"]}]
        hybrid_w = {"py": 0.5}
        level_w = {"junior": {"py": 0.8}}
        mc2 = MetricComputer(skill_weights, vac_skills, level_data, hybrid_w)
        with patch("src.pipeline.metric_computer.ProfileEvaluator") as MockPE:
            result = mc2.prepare(level_w)
        assert result.is_ok()
        MockPE.assert_called_once_with(
            skill_weights=skill_weights,
            vacancies_skills=vac_skills,
            vacancies_skills_dict=level_data,
            hybrid_weights=hybrid_w,
            skill_weights_by_level=level_w,
        )


class TestMetricComputerCompute:
    def test_compute_no_prepare(self, mc):
        result = mc.compute({})
        assert result.is_err()

    def test_compute_ok(self, mc):
        mc.prepare({})
        with patch.object(mc.evaluator, "evaluate_profile",
                          return_value=Ok({"score": 0.8})):
            result = mc.compute({"student1": {}})
        assert result.is_ok()
        data = result.unwrap()
        assert data == {"student1": {"score": 0.8}}

    def test_compute_multiple_students(self, mc):
        mc.prepare({})
        mock_eval = MagicMock()
        mock_eval.evaluate_profile.side_effect = [
            Ok({"score": 0.9}),
            Ok({"score": 0.7}),
        ]
        mc.evaluator = mock_eval
        result = mc.compute({"s1": {}, "s2": {}})
        assert result.is_ok()
        data = result.unwrap()
        assert data["s1"]["score"] == 0.9
        assert data["s2"]["score"] == 0.7

    def test_compute_eval_error_skips(self, mc):
        mc.prepare({})
        mock_eval = MagicMock()
        mock_eval.evaluate_profile.side_effect = [
            Err("fail s1"),
            Ok({"score": 0.5}),
        ]
        mc.evaluator = mock_eval
        result = mc.compute({"s1": {}, "s2": {}})
        assert result.is_ok()
        data = result.unwrap()
        assert "s1" not in data
        assert data["s2"]["score"] == 0.5

    def test_compute_exception(self, mc):
        mc.prepare({})
        with patch.object(mc.evaluator, "evaluate_profile",
                          side_effect=ValueError("unexpected")):
            result = mc.compute({"s1": {}})
        assert result.is_err()
