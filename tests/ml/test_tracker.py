"""Тесты ExperimentTracker."""
import json
from pathlib import Path
from src.ml.tracker import ExperimentTracker


class TestExperimentTracker:
    def setup_method(self):
        self.tracker = ExperimentTracker()

    def test_start_and_finish(self):
        self.tracker.start("test_run", params={"lr": 0.01})
        self.tracker.log_metric("accuracy", 0.95)
        result = self.tracker.finish()
        assert result.is_ok()
        run = result.unwrap()
        assert run.name == "test_run"
        assert run.metrics["accuracy"] == 0.95
        assert run.finished_at is not None

    def test_finish_without_start_errors(self):
        result = self.tracker.finish()
        assert result.is_err()

    def test_log_metric_without_start_errors(self):
        result = self.tracker.log_metric("acc", 0.5)
        assert result.is_err()

    def test_double_start_errors(self):
        self.tracker.start("run1")
        result = self.tracker.start("run2")
        assert result.is_err()

    def test_persist_to_disk(self, tmp_path):
        tracker = ExperimentTracker(log_dir=tmp_path)
        tracker.start("persist_test", params={"epochs": 10})
        tracker.log_metric("loss", 0.1)
        tracker.finish()

        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["name"] == "persist_test"
        assert data["metrics"]["loss"] == 0.1

    def test_log_params(self):
        self.tracker.start("test")
        self.tracker.log_params({"batch_size": 32, "lr": 0.001})
        result = self.tracker.finish()
        assert result.unwrap().params["batch_size"] == 32
