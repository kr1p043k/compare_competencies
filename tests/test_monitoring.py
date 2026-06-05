"""Tests for Prometheus monitoring."""

import time
from unittest.mock import patch

import pytest

from src.monitoring.metrics import (
    track_pipeline_stage, pipeline_duration, pipeline_errors,
    recommendations_generated, ltr_model_metrics, get_metrics,
)


class TestMonitoringMetrics:
    def test_get_metrics_returns_data(self):
        data, content_type = get_metrics()
        assert data
        assert "text/plain" in content_type

    def test_track_pipeline_stage_success(self):
        @track_pipeline_stage("test_stage")
        def dummy_func():
            return "ok"

        result = dummy_func()
        assert result == "ok"

    def test_track_pipeline_stage_failure(self):
        @track_pipeline_stage("test_error_stage")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_func()

    def test_recommendations_counter(self):
        recommendations_generated.labels(profile="test_profile").inc(5)
        samples = []
        for metric in recommendations_generated.collect():
            for s in metric.samples:
                if s.name == "recommendations_generated_total":
                    samples.append(s)
        matching = [s for s in samples if s.labels.get("profile") == "test_profile"]
        assert len(matching) > 0

    def test_ltr_model_metrics(self):
        ltr_model_metrics.labels(metric="test_metric").set(0.95)
        for metric in ltr_model_metrics.collect():
            for s in metric.samples:
                if s.labels.get("metric") == "test_metric":
                    assert s.value == 0.95
                    return
        pytest.fail("Metric not found")

    def test_pipeline_errors_counter(self):
        pipeline_errors.labels(stage="test").inc()
        for metric in pipeline_errors.collect():
            for s in metric.samples:
                if s.labels.get("stage") == "test":
                    assert s.value >= 1
                    return
        pytest.fail("Error metric not found")
