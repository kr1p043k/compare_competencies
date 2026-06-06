from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineRun,
    StageResult,
)
from src.pipeline.event_bus import EventBus, PipelineEvent
from src.pipeline.stage import PipelineStage
from src.errors import PipelineError


class TestStageResultData:
    def test_with_data(self):
        sr = StageResult(name="s1", status="ok", elapsed=1.0, data={"key": "val"})
        assert sr.data == {"key": "val"}

    def test_without_data(self):
        sr = StageResult(name="s1", status="ok", elapsed=1.0)
        assert sr.data is None


class TestPipelineRunElapsed:
    def test_default_started_at(self):
        import time
        before = time.time()
        run = PipelineRun()
        assert run.started_at == 0.0

    def test_elapsed_in_flight(self):
        run = PipelineRun(started_at=100.0)
        assert run.elapsed >= 0


class TestOrchestratorEventBus:
    def test_custom_event_bus(self):
        bus = EventBus()
        stages = [MagicMock(spec=PipelineStage)]
        stages[0].name = "s1"
        stages[0].run.return_value = Ok({})
        stages[0].rollback = MagicMock()
        orch = PipelineOrchestrator(stages, num_retries=0, event_bus=bus)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()
        assert len(bus.history) >= 2

    def test_event_bus_emits_start_and_finish(self):
        bus = EventBus()
        events = []
        bus.on("*", lambda e: events.append(e))
        stages = [MagicMock(spec=PipelineStage)]
        stages[0].name = "s1"
        stages[0].run.return_value = Ok({})
        stages[0].rollback = MagicMock()
        orch = PipelineOrchestrator(stages, num_retries=0, event_bus=bus)
        with patch("src.pipeline.orchestrator.write_progress"):
            orch.run()
        statuses = [e.status for e in events]
        assert "start" in statuses
        assert "finish" in statuses

    def test_event_bus_emits_stage_events(self):
        bus = EventBus()
        events = []
        bus.on("*", lambda e: events.append(e))
        stages = [MagicMock(spec=PipelineStage)]
        stages[0].name = "s1"
        stages[0].run.return_value = Ok({})
        stages[0].rollback = MagicMock()
        orch = PipelineOrchestrator(stages, num_retries=0, event_bus=bus)
        with patch("src.pipeline.orchestrator.write_progress"):
            orch.run()
        stage_events = [e for e in events if e.stage == "s1"]
        assert len(stage_events) == 1
        assert stage_events[0].status == "ok"


class TestOrchestratorInitialCtx:
    def test_initial_ctx_passed_to_stages(self):
        stage = MagicMock(spec=PipelineStage)
        stage.name = "s1"
        stage.run.return_value = Ok({"result": "ok"})
        stage.rollback = MagicMock()
        orch = PipelineOrchestrator([stage], num_retries=0)
        with patch("src.pipeline.orchestrator.write_progress"):
            orch.run(initial_ctx={"initial": "value"})
        stage.run.assert_called_once_with(initial="value")


class TestOrchestratorRetryPolicy:
    def test_custom_retry_policy(self):
        bus = EventBus()
        stage = MagicMock(spec=PipelineStage)
        stage.name = "s1"
        stage.run.side_effect = [Err("fail1"), Ok({})]
        stage.rollback = MagicMock()
        orch = PipelineOrchestrator([stage], num_retries=1, event_bus=bus)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_ok()
        assert orch.retry_policy.max_retries == 1

    def test_failure_with_event_logged(self):
        bus = EventBus()
        events = []
        bus.on("*", lambda e: events.append(e))
        stage = MagicMock(spec=PipelineStage)
        stage.name = "s1"
        stage.run.return_value = Err("fatal")
        stage.rollback = MagicMock()
        orch = PipelineOrchestrator([stage], num_retries=0, event_bus=bus)
        with patch("src.pipeline.orchestrator.write_progress"):
            result = orch.run()
        assert result.is_err()
        fail_events = [e for e in events if e.status == "fail"]
        assert len(fail_events) >= 1
