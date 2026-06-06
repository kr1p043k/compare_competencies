from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src import Ok, Err
from src.evaluation.base import (
    BaseEvaluator,
    CoverageEvaluator,
    EvalReport,
    EvalSuite,
    GapAccuracyEvaluator,
    SkillMatchEvaluator,
)
from src.errors import DomainError


class TestEvalReport:
    def test_create_default(self):
        r = EvalReport(evaluator_name="test", metric_name="acc", metric_value=1.0)
        assert r.evaluator_name == "test"
        assert r.metric_name == "acc"
        assert r.metric_value == 1.0
        assert r.samples == 0
        assert r.details == {}
        assert r.timestamp is not None

    def test_create_full(self):
        r = EvalReport("e1", "m1", 0.5, samples=10, details={"key": "val"})
        assert r.samples == 10
        assert r.details["key"] == "val"


class TestCoverageEvaluator:
    def test_init_default(self):
        e = CoverageEvaluator()
        assert e.name == "coverage"
        assert e.threshold == 0.75

    def test_evaluate_file_not_found(self):
        e = CoverageEvaluator()
        result = e.evaluate()
        assert result.is_err()

    def test_evaluate_ok(self):
        e = CoverageEvaluator()
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        with patch("builtins.open"):
            with patch("json.load", return_value=[{"s": 1}]):
                result = e.evaluate(student_file=mock_file)
        assert result.is_ok()
        report = result.unwrap()
        assert report.metric_name == "coverage_ratio"

    def test_evaluate_exception(self):
        e = CoverageEvaluator()
        mock_file = MagicMock()
        mock_file.exists.side_effect = Exception("boom")
        result = e.evaluate(student_file=mock_file)
        assert result.is_err()


class TestSkillMatchEvaluator:
    def test_precision_recall_f1_perfect(self):
        e = SkillMatchEvaluator()
        result = e.evaluate(predicted=["a", "b"], actual=["a", "b"])
        assert result.is_ok()
        r = result.unwrap()
        assert r.metric_value == 1.0
        assert r.details["precision"] == 1.0
        assert r.details["recall"] == 1.0

    def test_f1_partial(self):
        e = SkillMatchEvaluator()
        result = e.evaluate(predicted=["a", "b", "c"], actual=["a", "d"])
        assert result.is_ok()
        r = result.unwrap()
        assert abs(r.details["precision"] - 1/3) < 0.001
        assert abs(r.details["recall"] - 0.5) < 0.001
        assert abs(r.metric_value - 0.4) < 0.001

    def test_no_actual(self):
        e = SkillMatchEvaluator()
        result = e.evaluate(predicted=["a"], actual=[])
        assert result.is_err()

    def test_no_predicted(self):
        e = SkillMatchEvaluator()
        result = e.evaluate(predicted=[], actual=["a"])
        assert result.is_ok()
        r = result.unwrap()
        assert r.metric_value == 0.0
        assert r.details["precision"] == 0.0

    def test_empty_both(self):
        e = SkillMatchEvaluator()
        result = e.evaluate(predicted=[], actual=[])
        assert result.is_err()


class TestGapAccuracyEvaluator:
    def test_perfect_accuracy(self):
        e = GapAccuracyEvaluator()
        result = e.evaluate(
            recommended=[{"skill": "a"}, {"skill": "b"}],
            validation={"a": True, "b": True},
        )
        assert result.is_ok()
        assert result.unwrap().metric_value == 1.0

    def test_partial_accuracy(self):
        e = GapAccuracyEvaluator()
        result = e.evaluate(
            recommended=[{"skill": "a"}, {"skill": "b"}, {"skill": "c"}],
            validation={"a": True},
        )
        assert result.is_ok()
        assert abs(result.unwrap().metric_value - 1/3) < 0.001

    def test_no_validation(self):
        e = GapAccuracyEvaluator()
        result = e.evaluate(recommended=[], validation={})
        assert result.is_err()

    def test_no_recommendations(self):
        e = GapAccuracyEvaluator()
        result = e.evaluate(recommended=[], validation={"a": True})
        assert result.is_ok()
        assert result.unwrap().metric_value == 0.0


class TestEvalSuite:
    def test_default_evaluators(self):
        s = EvalSuite()
        assert len(s._evaluators) == 3

    def test_add_custom(self):
        e = MagicMock(spec=BaseEvaluator)
        e.name = "custom"
        s = EvalSuite()
        initial = len(s._evaluators)
        s.add(e)
        assert len(s._evaluators) == initial + 1

    def test_run_all(self):
        s = EvalSuite()
        e = MagicMock(spec=BaseEvaluator)
        e.name = "custom"
        e.eval_or_default.return_value = EvalReport("custom", "m1", 1.0)
        s.add(e)
        reports = s.run_all()
        assert len(reports) == len(s._evaluators)

    def test_summary(self):
        s = EvalSuite([])
        reports = [
            EvalReport("cov", "r", 0.8),
            EvalReport("match", "f1", 0.5),
        ]
        summary = s.summary(reports)
        assert summary == {"cov/r": 0.8, "match/f1": 0.5}

    def test_eval_or_default_on_error(self):
        e = MagicMock(spec=BaseEvaluator)
        e.name = "failing"
        e.evaluate.return_value = Err(DomainError("test error"))
        e.eval_or_default = BaseEvaluator.eval_or_default.__get__(e, BaseEvaluator)
        report = e.eval_or_default()
        assert report.metric_value == 0.0
        assert report.evaluator_name == "failing"
        assert report.metric_name == "default"
