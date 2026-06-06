"""E2E integration tests for the full pipeline — orchestrator, stages, progress, events, clean, builder."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.errors import PipelineError
from src.pipeline.builder import PipelineBuilder
from src.pipeline.event_bus import EventBus, PipelineEvent
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineRun, StageResult
from src.pipeline.stage import PipelineStage
from src.models.data_contracts import PipelineContext


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_stage():
    """Returns a factory that creates named MagicMock stages."""
    def _make(name: str, return_value=Ok({})):
        s = MagicMock(spec=PipelineStage)
        s.name = name
        s.run.return_value = return_value
        s.pct_range = (0, 100)
        return s
    return _make


@pytest.fixture
def patch_progress():
    """Patch write_progress in both orchestrator and stage modules."""
    with patch("src.pipeline.orchestrator.write_progress") as p:
        yield p


# =====================================================================
# 1. Mock-based pipeline run
# =====================================================================

def test_pipeline_runs_all_stages(mock_stage, patch_progress):
    """All stages execute successfully, PipelineRun is completed."""
    s1 = mock_stage("load", Ok({"vacancies": []}))
    s2 = mock_stage("extract", Ok({"skills": {"python": 5}}))
    s3 = mock_stage("train", Ok({"model": "ok"}))

    orch = PipelineOrchestrator([s1, s2, s3], num_retries=0)
    result = orch.run()

    assert result.is_ok()
    run = result.unwrap()
    assert run.status == "completed"
    assert len(run.stages) == 3
    for st in run.stages:
        assert st.status == "ok"
    assert run.started_at > 0
    assert run.finished_at > 0
    assert run.elapsed > 0


def test_pipeline_passes_context_between_stages(mock_stage, patch_progress):
    """Context dict is updated from one stage to the next."""
    s1 = mock_stage("a", Ok({"vacancies": ["v1"]}))
    s2 = MagicMock(spec=PipelineStage)
    s2.name = "b"
    s2.pct_range = (0, 100)
    s2.run.return_value = Ok({"model": "trained"})

    orch = PipelineOrchestrator([s1, s2], num_retries=0)
    result = orch.run()

    assert result.is_ok()
    _, kwargs = s2.run.call_args
    assert kwargs.get("vacancies") == ["v1"]


def test_pipeline_with_non_dict_return(mock_stage, patch_progress):
    """If a stage returns non-dict, it is stored under stage name key."""
    s1 = mock_stage("scan", Ok(42))
    s2 = mock_stage("check", Ok({}))
    orch = PipelineOrchestrator([s1, s2], num_retries=0)
    result = orch.run()

    assert result.is_ok()
    run = result.unwrap()
    assert run.stages[0].data == 42
    # Stage "scan" returned 42, so context should have scan_result=42
    # But assert that second stage still got called (just without extra keys)
    assert s2.run.called


def test_pipeline_initial_ctx(mock_stage, patch_progress):
    """Initial context is passed to the first stage."""
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "first"
    s1.pct_range = (0, 100)
    s1.run.return_value = Ok({})
    orch = PipelineOrchestrator([s1], num_retries=0)
    orch.run(initial_ctx={"config": "test"})

    _, kwargs = s1.run.call_args
    assert kwargs.get("config") == "test"


# =====================================================================
# 2. Stage error propagation
# =====================================================================

def test_stage_error_stops_pipeline(mock_stage, patch_progress):
    """When a stage returns Err, the pipeline halts and returns PipelineError."""
    s1 = mock_stage("ok_stage", Ok({"a": 1}))
    s2 = mock_stage("fail_stage", Err("something went wrong"))
    s3 = mock_stage("never_reached", Ok({}))

    orch = PipelineOrchestrator([s1, s2, s3], num_retries=0)

    with patch.object(s1, "rollback") as rb_s1, \
         patch.object(s2, "rollback") as rb_s2:
        result = orch.run()

    assert result.is_err()
    err = result.err()
    assert isinstance(err, PipelineError)
    assert "fail_stage" in err.stage
    assert "something went wrong" in err.detail or "something went wrong" in err.message
    assert not s3.run.called
    # rollback is called on all completed stages up to and including the failed one
    rb_s1.assert_called_once()
    rb_s2.assert_called_once()


def test_stage_exception_stops_pipeline(mock_stage, patch_progress):
    """If a stage raises an exception, the pipeline halts."""
    s1 = mock_stage("crash", None)
    s1.run.side_effect = ValueError("crash")
    s2 = mock_stage("never", Ok({}))

    orch = PipelineOrchestrator([s1, s2], num_retries=0)
    with patch.object(s1, "rollback"):
        result = orch.run()

    assert result.is_err()
    assert not s2.run.called


def test_stage_recovers_on_retry(mock_stage, patch_progress):
    """Stage that fails once then succeeds on retry allows pipeline to complete."""
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "flaky"
    s1.pct_range = (0, 100)
    s1.run.side_effect = [Err("first fail"), Ok({"done": True})]

    orch = PipelineOrchestrator([s1], num_retries=1)
    result = orch.run()

    assert result.is_ok()
    run = result.unwrap()
    assert run.status == "completed"


def test_stage_exception_recovers_on_retry(mock_stage, patch_progress):
    """Stage that throws then succeeds on retry."""
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "flaky_exc"
    s1.pct_range = (0, 100)
    s1.run.side_effect = [ValueError("boom"), Ok({"ok": True})]

    orch = PipelineOrchestrator([s1], num_retries=1)
    result = orch.run()

    assert result.is_ok()


def test_error_propagates_without_retries(mock_stage, patch_progress):
    """With num_retries=0, any error immediately stops the pipeline."""
    s1 = mock_stage("fail", Err("no retry"))
    orch = PipelineOrchestrator([s1], num_retries=0)
    with patch.object(s1, "rollback"):
        result = orch.run()
    assert result.is_err()


# =====================================================================
# 3. Progress tracking
# =====================================================================

def test_progress_from_0_to_100(mock_stage):
    """Progress goes from 0% to 100% throughout pipeline execution."""
    calls = []

    def track(pct, msg):
        calls.append(pct)

    s1 = mock_stage("step1", Ok({"a": 1}))
    s2 = mock_stage("step2", Ok({"b": 2}))

    with patch("src.pipeline.orchestrator.write_progress", side_effect=track):
        orch = PipelineOrchestrator([s1, s2], num_retries=0)
        orch.run()

    assert calls[0] == 0
    assert calls[-1] == 100
    # Progress should be monotonically non-decreasing
    for i in range(1, len(calls)):
        assert calls[i] >= calls[i - 1]


def test_progress_with_failure(mock_stage):
    """Progress is recorded even when a stage fails."""
    calls = []

    def track(pct, msg):
        calls.append(pct)

    s1 = mock_stage("good", Ok({}))
    s2 = mock_stage("bad", Err("fail"))

    with patch("src.pipeline.orchestrator.write_progress", side_effect=track), \
         patch.object(s1, "rollback"):
        orch = PipelineOrchestrator([s1, s2], num_retries=0)
        orch.run()

    assert len(calls) >= 3
    assert 0 in calls


# =====================================================================
# 4. Event bus emissions
# =====================================================================

def test_event_bus_emits_start_and_finish(mock_stage, patch_progress):
    """Pipeline emits __pipeline__ start and finish events."""
    events = []
    bus = EventBus()
    bus.on("*", lambda e: events.append(e))

    s1 = mock_stage("s1", Ok({}))
    orch = PipelineOrchestrator([s1], num_retries=0, event_bus=bus)
    orch.run()

    statuses = [(e.stage, e.status) for e in events]
    assert ("__pipeline__", "start") in statuses
    assert ("__pipeline__", "finish") in statuses


def test_event_bus_emits_stage_events(mock_stage, patch_progress):
    """Each stage emits ok/fail events with metadata."""
    bus = EventBus()
    events = []
    bus.on("ok", lambda e: events.append(e))
    bus.on("fail", lambda e: events.append(e))

    s1 = mock_stage("alpha", Ok({"x": 1}))
    s2 = mock_stage("beta", Err("oops"))

    with patch.object(s1, "rollback"):
        orch = PipelineOrchestrator([s1, s2], num_retries=0, event_bus=bus)
        orch.run()

    stage_events = [(e.stage, e.status) for e in events]
    assert ("alpha", "ok") in stage_events
    assert ("beta", "fail") in stage_events
    for e in events:
        assert e.elapsed >= 0
        assert isinstance(e.metadata, dict)


def test_event_bus_handler_exception_does_not_crash_pipeline(mock_stage, patch_progress):
    """A crashing event handler does not interrupt the pipeline."""
    bus = EventBus()
    bus.on("*", lambda e: (_ for _ in ()).throw(ValueError("bad handler")))

    s1 = mock_stage("s1", Ok({}))
    orch = PipelineOrchestrator([s1], num_retries=0, event_bus=bus)
    result = orch.run()

    assert result.is_ok()


# =====================================================================
# 5. Clean functionality
# =====================================================================

@pytest.fixture(scope="module")
def clean_module():
    """Import src.pipeline.clean with all external deps mocked."""
    fuzz_mock = MagicMock()
    fuzz_mock.token_set_ratio.side_effect = lambda e, tl: 90 if any(w in tl for w in e.split()) else 50
    fuzz_mock.partial_ratio.side_effect = lambda e, tl: 95 if e.lower() in tl.lower() else 50

    with patch.dict("sys.modules", {"rapidfuzz": MagicMock(fuzz=fuzz_mock)}), \
         patch("src.parsing.utils.load_it_skills", return_value=[]), \
         patch("src.parsing.skills.skill_validator.SkillValidator"), \
         patch("builtins.open", MagicMock()), \
         patch("os.path.exists", return_value=False), \
         patch("sys.stdout"):
        from src.pipeline import clean
        clean.combined_list = ["python", "docker", "machine learning", "sql", "git"]
        clean.fuzz = fuzz_mock
        return clean


def test_clean_find_skills_exact_match(clean_module):
    """find_skills finds skills by exact substring match."""
    clean = clean_module
    result = clean.find_skills("python developer with docker experience")
    assert "python" in result
    assert "docker" in result
    assert "sql" not in result  # not present in text


def test_clean_find_skills_empty_or_short(clean_module):
    """find_skills returns empty set for short or empty text."""
    clean = clean_module
    assert clean.find_skills("") == set()
    assert clean.find_skills("ab") == set()


def test_clean_find_skills_fuzzy_match(clean_module):
    """find_skills falls back to fuzzy token_set_ratio for multi-word entries."""
    clean = clean_module
    clean.fuzz.token_set_ratio.return_value = 86  # above 85 threshold
    result = clean.find_skills("machine learning engineer")
    assert "machine learning" in result


def test_clean_find_skills_no_match(clean_module):
    """find_skills returns empty set when nothing matches."""
    clean = clean_module
    clean.fuzz.token_set_ratio.return_value = 50
    clean.fuzz.partial_ratio.return_value = 50
    result = clean.find_skills("quantum computing with aws")
    assert result == set()


def test_clean_competency_skills_flat(clean_module):
    """clean_competency_skills extracts skills from KSA data."""
    clean = clean_module
    data = {
        "disciplines": {
            "it": {
                "competencies": ["C1"],
                "ksa": {
                    "C1": {
                        "knowledge": ["python basics", "docker compose"],
                        "abilities": ["git collaboration"],
                        "skills": ["sql queries"]
                    }
                }
            }
        }
    }
    result = clean.clean_competency_skills(data)
    skills = result["disciplines"]["it"]["skills"]["C1"]
    assert "python" in skills
    assert "docker" in skills
    assert "git" in skills
    assert "sql" in skills


def test_clean_competency_skills_empty(clean_module):
    """clean_competency_skills handles empty data."""
    clean = clean_module
    data = {"disciplines": {}}
    result = clean.clean_competency_skills(data)
    assert result == data


# =====================================================================
# 6. Builder
# =====================================================================

def test_builder_fluent_api():
    """PipelineBuilder fluent methods return self for chaining."""
    b = PipelineBuilder()
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "s1"
    s2 = MagicMock(spec=PipelineStage)
    s2.name = "s2"

    result = b.with_stage(s1).with_stages(s2).with_retries(3).named("e2e")
    assert result is b
    assert len(b._stages) == 2
    assert b._retries == 3
    assert b._name == "e2e"


def test_builder_build_creates_orchestrator():
    """build() returns a PipelineOrchestrator with correct stages."""
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "a"
    s2 = MagicMock(spec=PipelineStage)
    s2.name = "b"

    orch = PipelineBuilder() \
        .with_stages(s1, s2) \
        .with_retries(2) \
        .build()

    assert isinstance(orch, PipelineOrchestrator)
    assert len(orch.stages) == 2
    assert orch.num_retries == 2


def test_builder_run_success(mock_stage, patch_progress):
    """Builder.run returns Ok when all stages succeed."""
    s1 = mock_stage("ok1", Ok({"x": 1}))

    result = PipelineBuilder().with_stage(s1).run()
    assert result.is_ok()


def test_builder_run_failure(mock_stage, patch_progress):
    """Builder.run returns Err when a stage fails."""
    s1 = mock_stage("fail1", Err("builder fail"))

    with patch.object(s1, "rollback"):
        result = PipelineBuilder().with_stage(s1).with_retries(0).run()

    assert result.is_err()


def test_builder_empty_stages():
    """Builder with no stages still creates an orchestrator."""
    orch = PipelineBuilder().build()
    assert isinstance(orch, PipelineOrchestrator)
    assert orch.stages == []


def test_builder_repr():
    """__repr__ includes builder state."""
    s1 = MagicMock(spec=PipelineStage)
    s1.name = "alpha"
    b = PipelineBuilder().with_stage(s1).named("test").with_retries(2)
    r = repr(b)
    assert "PipelineBuilder" in r
    assert "alpha" in r
    assert "test" in r


# =====================================================================
# PipelineRun & StageResult
# =====================================================================

def test_pipeline_run_elapsed():
    """PipelineRun.elapsed works before and after finish."""
    run = PipelineRun(started_at=100.0)
    assert run.elapsed >= 0

    run.finished_at = 105.0
    assert run.elapsed == 5.0


def test_stage_result_defaults():
    """StageResult default field values."""
    sr = StageResult(name="test", status="ok", elapsed=0.5)
    assert sr.error is None
    assert sr.data is None


def test_stage_result_with_error():
    """StageResult stores error string."""
    sr = StageResult(name="test", status="failed", elapsed=0.5, error="error msg")
    assert sr.error == "error msg"
