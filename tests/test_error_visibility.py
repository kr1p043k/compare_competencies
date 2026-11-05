"""Error-visibility tests for the silent-except hardening (merge/prod-hardening).

Before this change every ``except Exception`` block in the LLM/pipeline/predictor
modules dropped the error with no log and no re-raise, so a real failure read as
"no data" instead of "something broke".

These tests pin the two invariants of the fix per representative handler:

1. a log record IS emitted - asserted through a logger spy, because structlog is
   NOT wired to stdlib logging in the test process (``setup_structlog()`` runs
   only at app startup in ``src/api_pkg/startup.py``), so ``caplog`` cannot see it;
2. the fallback value is returned unchanged (control flow preserved).

Also covered: the ``psycopg2`` optional-dependency guard stays silent at import.
"""
from __future__ import annotations

import sys
from importlib import reload
from unittest.mock import patch

from src import config
from src.pipeline import background_collector
from src.services import llm_extract, llm_recommend


class SpyLogger:
    """Records the (method, event, kwargs) of every logger call it receives.

    Patched over a module's ``logger`` global, it is deterministic regardless of
    how structlog is (or is not) configured in the running process.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(event=None, *args, **kwargs):
            self.calls.append((name, event, kwargs))
            return None

        return record

    def warned(self, event: str) -> list[dict]:
        return [kw for m, ev, kw in self.calls if m == "warning" and ev == event]

    def errored(self, event: str) -> list[dict]:
        return [kw for m, ev, kw in self.calls if m == "exception" and ev == event]


class _BoomMessage:
    @property
    def content(self) -> str:
        raise RuntimeError("content boom")


class _BoomChoice:
    message = _BoomMessage()


class _BoomResponse:
    choices = [_BoomChoice()]


class _BoomClient:
    def chat(self, *args, **kwargs):
        return _BoomResponse()


# ---------------------------------------------------------------------------
# Representative handler 1: outer wrapper uses logger.exception + fallback.
# ---------------------------------------------------------------------------
def test_student_recs_outer_failure_logs_and_returns_base(monkeypatch):
    spy = SpyLogger()
    monkeypatch.setattr(llm_recommend, "logger", spy)
    monkeypatch.setattr(
        llm_recommend, "_enhance_core",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("enhance core blew up")),
    )

    base = {"recommendations": [{"skill": "Python", "gap": 0.8}]}
    got = llm_recommend.enhance_student_recs("profile", base, client=None, use_cache=False)

    assert got == base
    assert spy.errored("student_recs_failed")


def test_teacher_recs_outer_failure_logs_and_returns_base(monkeypatch):
    spy = SpyLogger()
    monkeypatch.setattr(llm_recommend, "logger", spy)
    monkeypatch.setattr(
        llm_recommend, "_enhance_core",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("enhance core blew up")),
    )

    base = {"recommendations": [{"skill": "SQL", "gap": 0.2}]}
    got = llm_recommend.enhance_teacher_recs("Math", [0.5], base, client=None, use_cache=False)

    assert got == base
    assert spy.errored("teacher_recs_failed")
    # discipline is a stable identifier; it must be attached, not the prompt.
    assert spy.errored("teacher_recs_failed")[0].get("discipline") == "Math"


# ---------------------------------------------------------------------------
# Representative handler 2: inner fallback uses logger.warning(exc_info=True).
# ---------------------------------------------------------------------------
def test_extract_chat_content_failure_logs_warning(monkeypatch):
    spy = SpyLogger()
    monkeypatch.setattr(llm_extract, "logger", spy)

    got = llm_extract._chat_guarded(
        _BoomClient(), [{"role": "user", "content": "x"}], timeout_s=20
    )

    assert got == ""
    calls = spy.warned("llm_chat_content_unavailable")
    assert calls, "expected a warning for the failed content access"
    assert calls[0].get("exc_info") is True


def test_extract_skills_outer_failure_logs_and_returns_empty(monkeypatch):
    spy = SpyLogger()
    monkeypatch.setattr(llm_extract, "logger", spy)

    def boom(client, messages, timeout_s):
        raise RuntimeError("chat broke")

    monkeypatch.setattr(llm_extract, "_chat_guarded", boom)

    got = llm_extract.extract_skills("some text", ["Python", "SQL"], client=object(), use_cache=False)

    assert got == []
    assert spy.errored("skills_extract_failed")


# ---------------------------------------------------------------------------
# Representative handler 3: settings seed degrades with a warning, not a crash.
# ---------------------------------------------------------------------------
def test_seed_defaults_logs_warning_on_settings_failure(monkeypatch):
    spy = SpyLogger()
    monkeypatch.setattr(background_collector, "logger", spy)

    class BoomSettings:
        @property
        def background_collector_enabled(self) -> bool:
            raise RuntimeError("settings broken")

        SCHEDULER_COLLECT_INTERVAL_HOURS = 24
        SCHEDULER_DAILY_GAP_ENABLED = True

    monkeypatch.setattr(config, "settings", BoomSettings())

    got = background_collector._seed_defaults()

    # fallback preserved: the broken flag defaults to False, the rest is intact.
    assert got["collector_enabled"] is False
    assert got["collect_interval_hours"] == 24
    assert got["daily_gap_enabled"] is True
    calls = spy.warned("scheduler_seed_enabled_unavailable")
    assert calls, "expected a warning for the failed settings read"
    assert calls[0].get("exc_info") is True


# ---------------------------------------------------------------------------
# Import guard: psycopg2 missing must NOT raise and must NOT log.
# ---------------------------------------------------------------------------
def test_psycopg2_import_guard_is_silent():
    import src.predictors.curriculum_recommender as cr

    # Setting a sys.modules entry to None makes `import psycopg2` raise ImportError.
    with patch.dict(sys.modules, {"psycopg2": None}):
        mod = reload(cr)
        assert mod.psycopg2 is None

    # Restore the real binding for any later use in this session.
    reload(cr)
    assert cr.psycopg2 is not None
