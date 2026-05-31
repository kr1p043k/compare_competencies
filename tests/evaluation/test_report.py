from src.evaluation.report import EvaluationReport, MetricEntry


class TestMetricEntry:
    def test_default_passed(self):
        m = MetricEntry(name="p", value=0.8, threshold=0.5)
        assert m.passed is True

    def test_failed_threshold(self):
        m = MetricEntry(name="p", value=0.3, threshold=0.5)
        assert m.passed is False

    def test_no_threshold(self):
        m = MetricEntry(name="p", value=0.5)
        assert m.passed is None


class TestEvaluationReport:
    def test_add_metric(self):
        r = EvaluationReport(model_name="test_model")
        r.add("precision", 0.9, threshold=0.5)
        assert len(r.metrics) == 1
        assert r.metrics[0].name == "precision"

    def test_add_dict(self):
        r = EvaluationReport("m")
        r.add_dict({"p1": 0.8, "p2": 0.6}, threshold=0.5)
        assert len(r.metrics) == 2

    def test_summary(self):
        r = EvaluationReport("m")
        r.add("p", 0.9, threshold=0.5)
        r.add("r", 0.3, threshold=0.5)
        s = r.summary()
        assert s["model"] == "m"
        assert s["total_metrics"] == 2
        assert s["passed"] == 1
        assert s["failed"] == 1
        assert s["pass_rate"] == 0.5

    def test_summary_empty(self):
        r = EvaluationReport("empty")
        s = r.summary()
        assert s["total_metrics"] == 0
        assert s["pass_rate"] == 1.0

    def test_log_no_error(self):
        r = EvaluationReport("m")
        r.add("p", 0.9, threshold=0.5)
        r.log()

    def test_timestamp_set(self):
        r = EvaluationReport("m")
        assert r.timestamp is not None
