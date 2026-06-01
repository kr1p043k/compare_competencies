from unittest.mock import patch

from src import Ok, Err
from src.pipeline.metric_computer import MetricComputer


class TestMetricComputer:
    def test_prepare_ok(self):
        mc = MetricComputer({}, [], [], {})
        result = mc.prepare({})
        assert result.is_ok()

    def test_prepare_exception(self):
        mc = MetricComputer({}, [], [], {})
        with patch("src.pipeline.metric_computer.ProfileEvaluator", side_effect=ValueError("x")):
            result = mc.prepare({})
        assert result.is_err()

    def test_compute_no_prepare(self):
        mc = MetricComputer({}, [], [], {})
        result = mc.compute({})
        assert result.is_err()

    def test_compute_ok(self):
        mc = MetricComputer({}, [], [], {})
        mc.prepare({})
        with patch.object(mc.evaluator, "evaluate_profile", return_value=Ok({"score": 0.8})):
            result = mc.compute({"student1": {}})
        assert result.is_ok()
        data = result.unwrap()
        assert "student1" in data

    def test_compute_eval_error(self):
        mc = MetricComputer({}, [], [], {})
        mc.prepare({})
        with patch.object(mc.evaluator, "evaluate_profile", return_value=Err("fail")):
            result = mc.compute({"student1": {}})
        assert result.is_ok()
        assert len(result.unwrap()) == 0
