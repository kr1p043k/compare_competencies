# PipelineStage is an ABC — test with a concrete implementation
from src import Ok, Err
from src.errors import DomainError
from src.pipeline.stage import PipelineStage


class ConcreteStage(PipelineStage):
    name = "test_stage"
    pct_range = (10, 50)

    def run(self, **kwargs):
        return Ok("done")


class FailingStage(PipelineStage):
    name = "failing"

    def run(self, **kwargs):
        return Err(DomainError("fail"))


def test_stage_defaults():
    s = ConcreteStage()
    assert s.name == "test_stage"
    assert s.pct_range == (10, 50)


def test_stage_validate_default():
    s = ConcreteStage()
    assert s.validate() == Ok(True)


def test_stage_rollback_noop():
    s = ConcreteStage()
    s.rollback()


def test_stage_progress():
    s = ConcreteStage()
    s._progress(50, "halfway")


def test_stage_run():
    s = ConcreteStage()
    assert s.run() == Ok("done")


def test_failing_stage():
    s = FailingStage()
    result = s.run()
    assert result.is_err()


def test_stage_repr():
    s = ConcreteStage()
    assert hasattr(s, "run")
