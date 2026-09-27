"""LLM Phase 2 tests — offline (no network, no Ollama endpoint).

Binds REAL implementations:
  src.services.llm_extract._validate / extract_skills
  (client protocol: .chat(messages, temperature, max_tokens)
   -> obj with .choices[0].message.content)
  src.services.llm_cache.get_cached / put_cached (real Postgres tables
  llm_cache / skill_embedding_cache; rows cleaned up after)
  src.services.llm_recommend.enhance_student_recs / enhance_teacher_recs
  src.config LLM_* flags (safe defaults: everything OFF except master)

Covers:
  (a) extract validator: drops out-of-vocab names, dedupes, [] on failure
  (b) cache get/put roundtrip incl. miss -> None (real DB)
  (c) recommend enhancer: base unchanged on LLM failure, never new names
  (d) flags OFF by default
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import config as cfg
from src.services import llm_cache as dbcache
from src.services.llm_extract import _validate, extract_skills
from src.services.llm_recommend import enhance_student_recs, enhance_teacher_recs

VOCAB = ["Python", "SQL", "Docker", "FastAPI"]


class FakeChatClient:
    """Scripted stand-in with LLMClient.chat(message, ...) protocol."""

    def __init__(self, payload="", exc=None):
        self.payload = payload
        self.exc = exc
        self.calls = 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.payload))]
        )


def _msg(text):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# (a) extract validator + extract_skills
# ---------------------------------------------------------------------------
class TestExtractValidator:
    def test_drops_out_of_vocab(self):
        assert _validate(["Python", "cobol-xyz", "SQL", "unicorn-lang"], VOCAB) == [
            "Python", "SQL",
        ]

    def test_dedupes_case_insensitive(self):
        assert _validate(["python", "Python", " PYTHON ", "SQL", "sql"], VOCAB) == [
            "Python", "SQL",
        ]

    def test_extract_returns_empty_on_client_exception(self):
        client = FakeChatClient(exc=RuntimeError("endpoint down"))
        assert extract_skills("some text", VOCAB, client=client, use_cache=False) == []

    def test_extract_bad_json_returns_empty(self):
        client = FakeChatClient(payload="not-json{{{")
        assert extract_skills("text", VOCAB, client=client, use_cache=False) == []

    def test_extract_roundtrip_filters_vocab(self):
        client = FakeChatClient(payload=json.dumps(["Python", "cobol-xyz", "SQL", "SQL"]))
        assert extract_skills("text", VOCAB, client=client, use_cache=False) == [
            "Python", "SQL",
        ]

    def test_extract_empty_inputs(self):
        client = FakeChatClient(payload=json.dumps(["Python"]))
        assert extract_skills("", VOCAB, client=client, use_cache=False) == []
        assert extract_skills("text", [], client=client, use_cache=False) == []
        assert client.calls == 0


# ---------------------------------------------------------------------------
# (b) cache — REAL Postgres tables (cleaned up)
# ---------------------------------------------------------------------------
class TestCache:
    TASK = "phase2-test"

    def _purge(self):
        import psycopg2

        con = psycopg2.connect(
            dbname="compare_competencies", user="postgres",
            password="Admin_123!", host="127.0.0.1", port=5432,
        )
        con.autocommit = True
        cur = con.cursor()
        cur.execute("DELETE FROM llm_cache WHERE task=%s", (self.TASK,))
        con.close()

    def test_put_get_roundtrip(self):
        self._purge()
        try:
            dbcache.put_cached(self.TASK, "t-model", "hello", "world",
                               {"prompt_tokens": 1, "completion_tokens": 1})
            assert dbcache.get_cached(self.TASK, "t-model", "hello") == "world"
        finally:
            self._purge()

    def test_miss_returns_none(self):
        assert dbcache.get_cached(self.TASK, "t-model", "no-such-prompt-xyz") is None

    def test_overwrite_keeps_first(self):
        self._purge()
        try:
            dbcache.put_cached(self.TASK, "t-model", "k", "v1")
            dbcache.put_cached(self.TASK, "t-model", "k", "v2")
            assert dbcache.get_cached(self.TASK, "t-model", "k") == "v1"
        finally:
            self._purge()


# ---------------------------------------------------------------------------
# (c) recommend enhancer
# ---------------------------------------------------------------------------
class TestRecommendEnhancer:
    BASE = [
        {"skill": "Python", "gap": 0.8},
        {"skill": "SQL", "gap": 0.5},
        {"skill": "Docker", "gap": 0.3},
    ]

    def test_student_failure_returns_base_unchanged(self):
        client = FakeChatClient(exc=TimeoutError("ollama timeout"))
        got = enhance_student_recs("profile", self.BASE, client=client, use_cache=False)
        assert got == self.BASE

    def test_teacher_failure_returns_base_unchanged(self):
        client = FakeChatClient(exc=TimeoutError("ollama timeout"))
        got = enhance_teacher_recs("disc", [0.5], self.BASE, client=client,
                                   use_cache=False)
        assert got == self.BASE

    def test_never_introduces_new_names(self):
        payload = json.dumps([
            {"skill": "Python", "llm_reason": "x"},
            {"skill": "Rust", "llm_reason": "y"},
            {"skill": "SQL", "llm_reason": "z"},
        ])
        client = FakeChatClient(payload=payload)
        got = enhance_student_recs("profile", self.BASE, client=client, use_cache=False)
        names = [r["skill"] if isinstance(r, dict) else r for r in got]
        assert set(names) <= {"Python", "SQL", "Docker"}
        assert "Rust" not in names

    def test_none_client_returns_base(self):
        assert enhance_student_recs("p", self.BASE, client=None,
                                    use_cache=False) == self.BASE
        assert enhance_teacher_recs("d", [0.1], self.BASE, client=None,
                                    use_cache=False) == self.BASE


# ---------------------------------------------------------------------------
# (d) flags OFF by default
# ---------------------------------------------------------------------------
class TestFlagsOff:
    def test_safe_defaults(self):
        assert cfg.LLM_ENABLED is True
        assert cfg.LLM_ENHANCE_STUDENT is False
        assert cfg.LLM_ENHANCE_TEACHER is False
        assert cfg.LLM_EXTRACT is False
        assert float(cfg.LLM_TIMEOUT_S) == 20.0
