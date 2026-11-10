"""Coverage for src/services/llm_cache.py (backlog #49).

Offline unit tests: the SQLAlchemy engine is replaced by an in-memory fake, so
no Postgres, no network and no LLM are touched. The module contract is
"all public functions degrade gracefully to None on any DB error", so
failure paths are asserted as (returned None + logged warning + no raise).

Layers exercised:
  * pure helpers: _prompt_hash, _sync_database_url, _format_vector,
    _parse_vector, _extract_usage;
  * DB helpers via a fake engine: get_cached / put_cached (record cache) and
    get_embedding / put_embedding (pgvector-style string round-trip), plus
    the never-raise degradation contract on every public function.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from src.services import llm_cache


class _RaisingResult:
    def fetchone(self):
        raise RuntimeError("fetchone failed")


class _SelectResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, engine: _FakeEngine):
        self._engine = engine

    def execute(self, statement: Any, params: dict | None = None):
        if self._engine.error is not None:
            raise self._engine.error
        text = str(statement)
        if text.lstrip().upper().startswith("SELECT"):
            self._engine.selects.append((text, params))
            return _SelectResult(self._engine.select_rows.pop(0) if self._engine.select_rows else None)
        self._engine.executions.append((text, params))
        return None


class _ConnContext:
    def __init__(self, conn: _FakeConnection):
        self._conn = conn

    def __enter__(self) -> _FakeConnection:
        return self._conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeEngine:
    """Stand-in for llm_cache._engine(): records I/O, no real DB."""

    def __init__(self):
        self.error: Exception | None = None
        self.select_rows: list = []
        self.selects: list = []
        self.executions: list = []

    def connect(self) -> _ConnContext:
        return _ConnContext(_FakeConnection(self))

    def begin(self) -> _ConnContext:
        return _ConnContext(_FakeConnection(self))


@pytest.fixture
def engine(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(llm_cache, "_engine", lambda: fake)
    return fake


class TestPromptHash:
    def test_matches_sha256_hexdigest(self):
        assert llm_cache._prompt_hash("abc") == hashlib.sha256(b"abc").hexdigest()

    def test_is_deterministic_64_char_hex(self):
        first = llm_cache._prompt_hash("кэш, prompt #1")
        second = llm_cache._prompt_hash("кэш, prompt #1")
        assert first == second
        assert len(first) == 64
        int(first, 16)

    def test_different_prompts_hash_apart(self):
        assert llm_cache._prompt_hash("a") != llm_cache._prompt_hash("b")


class TestSyncDatabaseUrl:
    def test_asyncpg_url_is_rewritten_to_psycopg2(self, monkeypatch):
        monkeypatch.setattr(
            llm_cache.settings,
            "DATABASE_URL",
            "postgresql+asyncpg://user:pw@host:5432/db",
            raising=False,
        )
        assert (
            llm_cache._sync_database_url()
            == "postgresql+psycopg2://user:pw@host:5432/db"
        )

    def test_plain_postgres_url_passthrough(self, monkeypatch):
        url = "postgresql://user:pw@other:5432/prod"
        monkeypatch.setattr(llm_cache.settings, "DATABASE_URL", url, raising=False)
        assert llm_cache._sync_database_url() == url

    def test_conversion_affects_only_driver_prefix(self, monkeypatch):
        monkeypatch.setattr(
            llm_cache.settings,
            "DATABASE_URL",
            "postgresql+asyncpg://user:pw@host:5432/db?ssl=verify-full",
            raising=False,
        )
        result = llm_cache._sync_database_url()
        assert result.startswith("postgresql+psycopg2://")
        assert result.endswith("?ssl=verify-full")


class TestFormatVector:
    def test_formats_pgvector_literal(self):
        assert llm_cache._format_vector([0.25, -1.5, 2.0]) == "[0.25,-1.5,2.0]"

    def test_empty_vector_renders_empty_vector(self):
        assert llm_cache._format_vector([]) == "[]"

    def test_integers_are_coerced_to_float(self):
        assert llm_cache._format_vector([1, 2]) == "[1.0,2.0]"


class TestParseVector:
    def test_round_trips_format_vector(self):
        assert llm_cache._parse_vector(llm_cache._format_vector([0.5, 1.5])) == [0.5, 1.5]

    def test_none_returns_none(self):
        assert llm_cache._parse_vector(None) is None

    def test_accepts_native_sequences(self):
        assert llm_cache._parse_vector([0.1, 0.9]) == [0.1, 0.9]
        assert llm_cache._parse_vector((2, 3)) == [2.0, 3.0]

    def test_accepts_unwrapped_comma_list(self):
        assert llm_cache._parse_vector(" 0.25, -0.75 ") == [0.25, -0.75]

    def test_empty_string_returns_none(self):
        assert llm_cache._parse_vector("") is None

    def test_garbage_returns_none(self):
        assert llm_cache._parse_vector("not-a-vector") is None


class TestExtractUsage:
    def test_dict_usage(self):
        assert llm_cache._extract_usage({"prompt_tokens": 5, "completion_tokens": 7}) == (5, 7)

    def test_object_usage(self):
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=13)
        assert llm_cache._extract_usage(usage) == (11, 13)

    def test_none_returns_none_pair(self):
        assert llm_cache._extract_usage(None) == (None, None)

    def test_partial_dict_maps_missing_to_none(self):
        assert llm_cache._extract_usage({"completion_tokens": 4}) == (None, 4)

    def test_non_numeric_tokens_degrade_to_none_pair(self):
        assert llm_cache._extract_usage({"prompt_tokens": {"a": 1}}) == (None, None)


class TestGetCached:
    def test_hit_returns_row_response(self, engine):
        engine.select_rows = [("expected answer",)]
        result = llm_cache.get_cached("extract", "model-x", "ping")
        assert result == "expected answer"
        assert len(engine.selects) == 1
        statement, params = engine.selects[0]
        assert "FROM llm_cache" in statement
        assert params == {
            "task": "extract",
            "model": "model-x",
            "ph": llm_cache._prompt_hash("ping"),
        }

    def test_miss_returns_none(self, engine):
        assert llm_cache.get_cached("extract", "model-x", "never cached") is None

    def test_db_error_degrades_to_none(self, engine):
        engine.error = RuntimeError("db down")
        assert llm_cache.get_cached("extract", "model-x", "ping") is None

    def test_db_error_logs_structlog_warning(self, engine, monkeypatch):
        engine.error = RuntimeError("select blew up")
        recorded: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            llm_cache,
            "logger",
            SimpleNamespace(warning=lambda event, **kw: recorded.append((event, kw))),
        )
        assert llm_cache.get_cached("tt", "mm", "pp") is None
        assert recorded and recorded[0][0] == "llm_cache.get_cached_failed"


class TestPutCached:
    def test_inserts_all_fields_with_hashed_prompt(self, engine):
        llm_cache.put_cached("extract", "model-x", "cached prompt", "answer", {"prompt_tokens": 5, "completion_tokens": 7})
        assert len(engine.executions) == 1
        statement, params = engine.executions[0]
        assert "INSERT INTO llm_cache" in statement
        assert "ON CONFLICT (task, model, prompt_hash) DO NOTHING" in statement
        assert params == {
            "task": "extract",
            "model": "model-x",
            "ph": llm_cache._prompt_hash("cached prompt"),
            "pt": "cached prompt",
            "rt": "answer",
            "pto": 5,
            "cto": 7,
        }

    def test_usage_defaults_to_none_columns(self, engine):
        llm_cache.put_cached("t", "m", "p", "r")
        _, params = engine.executions[0]
        assert (params["pto"], params["cto"]) == (None, None)

    def test_db_error_returns_none_without_raise(self, engine):
        engine.error = RuntimeError("db down")
        assert llm_cache.put_cached("t", "m", "p", "r") is None

    def test_weird_usage_degrades_but_row_still_written(self, engine):
        llm_cache.put_cached("t", "m", "p", "r", object())
        _, params = engine.executions[0]
        assert params["rt"] == "r"
        assert (params["pto"], params["cto"]) == (None, None)


class TestGetEmbedding:
    def test_returns_parsed_vector_from_pg_text(self, engine):
        engine.select_rows = [("[0.5,-0.25]",)]
        assert llm_cache.get_embedding("python", "emb-model") == [0.5, -0.25]
        statement, params = engine.selects[0]
        assert "FROM skill_embedding_cache" in statement
        assert params == {"k": "python", "m": "emb-model"}

    def test_miss_returns_none(self, engine):
        assert llm_cache.get_embedding("nosuch", "emb-model") is None

    def test_db_error_degrades_to_none(self, engine):
        engine.error = RuntimeError("db down")
        assert llm_cache.get_embedding("python", "emb-model") is None


class TestPutEmbedding:
    def test_inserts_cast_to_vector(self, engine):
        llm_cache.put_embedding("python", "emb-model", [0.5, 0.25])
        assert len(engine.executions) == 1
        statement, params = engine.executions[0]
        assert "INSERT INTO skill_embedding_cache" in statement
        assert "CAST(:emb AS vector)" in statement
        assert params == {"k": "python", "m": "emb-model", "emb": "[0.5,0.25]"}

    def test_empty_vector_writes_nothing(self, engine):
        llm_cache.put_embedding("python", "emb-model", [])
        assert engine.executions == []
        assert engine.selects == []

    def test_db_error_returns_none_without_raise(self, engine):
        engine.error = RuntimeError("db down")
        assert llm_cache.put_embedding("python", "emb-model", [0.5]) is None
