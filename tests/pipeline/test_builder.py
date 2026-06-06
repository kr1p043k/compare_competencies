"""Tests for PipelineBuilder — mostly standalone, needs mocks for PipelineOrchestrator."""
from unittest.mock import patch, MagicMock

import pytest

from src import Ok, Err
from src.pipeline.builder import PipelineBuilder
from src.pipeline.stage import PipelineStage


class DummyStage(PipelineStage):
    name = "dummy"
    pct_range = (0, 100)

    def run(self, **kwargs):
        from src import Ok
        return Ok("ok")


def test_builder_init():
    b = PipelineBuilder()
    assert b._stages == []
    assert b._retries == 1
    assert b._name == "pipeline"


def test_with_stage():
    b = PipelineBuilder()
    s = DummyStage()
    r = b.with_stage(s)
    assert r is b
    assert len(b._stages) == 1


def test_with_stages():
    s1 = DummyStage()
    s2 = DummyStage()
    b = PipelineBuilder().with_stages(s1, s2)
    assert len(b._stages) == 2


def test_with_retries():
    b = PipelineBuilder().with_retries(3)
    assert b._retries == 3


def test_named():
    b = PipelineBuilder().named("test_pipeline")
    assert b._name == "test_pipeline"


@patch("src.pipeline.builder.PipelineOrchestrator")
def test_build(MockOrch):
    MockOrch.return_value = MagicMock()
    b = PipelineBuilder().with_stage(DummyStage())
    orch = b.build()
    assert orch is not None
    MockOrch.assert_called_once()


@patch("src.pipeline.builder.PipelineOrchestrator")
def test_build_no_stages(MockOrch):
    MockOrch.return_value = MagicMock()
    b = PipelineBuilder().build()


@patch("src.pipeline.builder.PipelineOrchestrator")
def test_run_success(MockOrch):
    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_run.stages = []
    mock_orch = MagicMock()
    mock_orch.run.return_value = Ok(mock_run)
    MockOrch.return_value = mock_orch

    b = PipelineBuilder().with_stage(DummyStage())
    result = b.run()
    assert result.is_ok()


@patch("src.pipeline.builder.PipelineOrchestrator")
def test_run_failure(MockOrch):
    mock_run = MagicMock()
    mock_run.status = "failed"
    mock_stage = MagicMock()
    mock_stage.error = "oops"
    mock_run.stages = [mock_stage]
    mock_orch = MagicMock()
    mock_orch.run.return_value = Ok(mock_run)
    MockOrch.return_value = mock_orch

    b = PipelineBuilder().with_stage(DummyStage())
    result = b.run()
    assert result.is_err()


def test_repr():
    b = PipelineBuilder().with_stage(DummyStage()).named("test").with_retries(2)
    r = repr(b)
    assert "PipelineBuilder" in r
    assert "test" in r
