from unittest.mock import MagicMock, patch

from src import Ok, Err
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineRun, StageResult


class TestPipelineRun:
    def test_elapsed_not_finished(self):
        run = PipelineRun(started_at=1000.0)
        assert run.elapsed >= 0

    def test_elapsed_finished(self):
        run = PipelineRun(started_at=1000.0, finished_at=1010.0)
        assert run.elapsed == 10.0


class TestPipelineOrchestrator:
    def make_stage(self, name, return_value):
        stage = MagicMock()
        stage.name = name
        stage.run.return_value = return_value
        return stage

    def test_run_all_ok(self):
        s1 = self.make_stage("stage1", Ok({"a": 1}))
        s2 = self.make_stage("stage2", Ok({"b": 2}))
        orch = PipelineOrchestrator([s1, s2], num_retries=1)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()
        run = result.unwrap()
        assert run.status == "completed"
        assert len(run.stages) == 2
        assert run.stages[0].status == "ok"
        assert run.stages[1].status == "ok"

    def test_run_stage_fails_no_retry(self):
        s1 = self.make_stage("stage1", Err("fail"))
        orch = PipelineOrchestrator([s1], num_retries=0)
        with patch("src.pipeline.orchestrator.write_progress"):
            with patch.object(s1, "rollback"):
                result = orch.run()
        assert result.is_err()

    def test_run_stage_succeeds_on_retry(self):
        s1 = MagicMock()
        s1.name = "stage1"
        s1.run.side_effect = [Err("fail"), Ok({"a": 1})]
        orch = PipelineOrchestrator([s1], num_retries=1)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()

    def test_stage_returns_non_dict_data(self):
        s1 = self.make_stage("stage1", Ok(42))
        orch = PipelineOrchestrator([s1])
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()

    def test_stage_exception(self):
        s1 = MagicMock()
        s1.name = "stage1"
        s1.run.side_effect = ValueError("boom")
        orch = PipelineOrchestrator([s1], num_retries=0)
        with patch("src.pipeline.orchestrator.write_progress"):
            with patch.object(s1, "rollback"):
                result = orch.run()
        assert result.is_err()

    def test_stage_exception_recovers_on_retry(self):
        s1 = MagicMock()
        s1.name = "stage1"
        s1.run.side_effect = [ValueError("boom"), Ok({"a": 1})]
        orch = PipelineOrchestrator([s1], num_retries=1)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()

    def test_stage_result_as_dataclass(self):
        sr = StageResult(name="s1", status="ok", elapsed=1.0)
        assert sr.name == "s1"
        assert sr.error is None

    def test_stage_result_with_error(self):
        sr = StageResult(name="s1", status="failed", elapsed=1.0, error="oops")
        assert sr.error == "oops"
