"""Tests for src/services/llm_chat.py (offline, no live LLM / no live hub).

Covers the acceptance criteria:
  (a) build_context assembles the right fields from injected loaders
  (b) answer_text uses config.LLM_CHAT_MODEL, answer_blocks uses
      config.LLM_CHAT_BLOCKS_MODEL
  (c) invalid block JSON falls back to a single text block
  (d) assess with LLM_ASSESS_ENABLED=False returns a non-empty deterministic
      text block and does not call the LLM
  (e) append_turn writes exactly one ChatMessage row with the next seq
  (f) build_context enforces the 50/40/15/10 limits
  (g) build_context with empty sources yields an empty context without
      raising (fallback to context_key as title, no phantom data)
  (h) block parsing edge cases: empty blocks list / no "blocks" key /
      JSON embedded in prose
"""

from __future__ import annotations

import json

from src.models.krm_models import ChatMessage
from src.services import llm_chat

# ── моки LLMClient ─────────────────────────────────────────────────────────

class _RecordingClient:
    """Записывает model= при конструировании и отдаёт фиксированный ответ."""

    instances: list[_RecordingClient] = []

    def __init__(self, model: str | None = None, **kwargs) -> None:
        self.model = model
        self.response = kwargs.get("response", "сводка")
        _RecordingClient.instances.append(self)

    def generate(self, prompt: str) -> str:
        return self.response


class _CapturingClient:
    """Запоминает model= в один слот (для проверки одиночного вызова)."""

    def __init__(self, model: str | None = None, **kwargs) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        return '{"blocks": [{"type": "text", "content": "x"}]}'


# ── моки сессий для persist-функций ────────────────────────────────────────

class _MaxResult:
    def scalar_one(self) -> int:
        return 0


class _AppendSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)

    async def execute(self, statement):
        return _MaxResult()

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _ThreadRows:
    def first(self):
        return None


class _ThreadResult:
    def scalars(self) -> _ThreadRows:
        return _ThreadRows()


class _ThreadSession:
    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)

    async def execute(self, statement):
        return _ThreadResult()

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ── config defaults ────────────────────────────────────────────────────────

def test_config_defaults_exported():
    from src import config

    assert config.LLM_ASSESS_ENABLED is False
    assert config.LLM_CHAT_BLOCKS_MODEL == "deepseek-ai/DeepSeek-V4-Pro-0813"
    assert config.LLM_CHAT_CONTEXT_TOKEN_BUDGET == 8000


# ── build_context ──────────────────────────────────────────────────────────

_TEACHER_SUMMARY = {
    "direction_name": "Информатика и вычислительная техника",
    "average_coverage": 0.82,
    "average_quality_coverage": 0.71,
    "coverage_level": "high",
    "total_gaps_across_all": 3,
    "top_cross_discipline_gaps": [
        {"skill": "Kafka", "disciplines": 2},
        {"skill": "Kubernetes", "disciplines": 1},
    ],
    "top_emerging_across_all": [{"skill": "Rust", "change_pct": 40}],
    "trends": {"rising": [{"skill": "LLM Ops"}]},
    "disciplines": [
        {
            "matched_market": ["Python", "SQL"],
            "top_matched": [{"skill_name": "Docker"}],
        }
    ],
}


def _teacher_summary_loader(dir_code: str) -> dict:
    return dict(_TEACHER_SUMMARY)


def _krm_loader(context_type: str, context_key: str) -> list[dict]:
    return [{"code": "ПК-1", "skills": ["Python", "SQL"]}]


async def _pubs_loader(query: str, limit: int) -> list[dict]:
    return [
        {"id": 1, "title": "Публикация A"},
        {"id": 2, "title": "Публикация B"},
    ]


async def test_build_context_teacher_fields():
    ctx = await llm_chat.build_context(
        "teacher",
        "09.03.02",
        summary_loader=_teacher_summary_loader,
        krm_loader=_krm_loader,
        pubs_loader=_pubs_loader,
    )

    assert ctx["context_type"] == "teacher"
    assert ctx["context_key"] == "09.03.02"
    assert ctx["title"] == "Информатика и вычислительная техника"
    assert ctx["metrics"]["coverage"] == 0.82
    assert ctx["metrics"]["quality"] == 0.71
    assert "Python" in ctx["skills"]
    assert ctx["gaps"][0]["skill"] == "Kafka"
    assert ctx["competencies"] == [{"code": "ПК-1", "skills": ["Python", "SQL"]}]
    assert len(ctx["publications"]) == 2
    assert ctx["frontiers"][0]["skill"] in {"Rust", "LLM Ops"}


async def test_build_context_student_fields():
    payload = {
        "profile": "base",
        "summary": {"coverage_strict": 64.0, "readiness_score": 55.0, "avg_gap": 12.0},
        "gaps": {"FastAPI": {"category": "missing", "gap_j": 0.6}},
        "recommendations": [
            {"skill": "FastAPI", "category": "missing", "market_frequency_percent": 87.0},
            {"skill": "Docker", "category": "weak", "market_frequency_percent": 40.0},
        ],
    }

    ctx = await llm_chat.build_context(
        "student", "base", recs_loader=lambda p: payload, pubs_loader=_pubs_loader
    )

    assert ctx["context_type"] == "student"
    assert ctx["title"] == "base"
    assert ctx["metrics"]["coverage"] == 64.0
    assert "FastAPI" in ctx["skills"]
    assert ctx["gaps"][0]["skill"] == "FastAPI"
    assert ctx["frontiers"][0]["skill"] == "FastAPI"


async def test_build_context_enforces_limits():
    many_gaps = [{"skill": f"g{i}", "disciplines": 1} for i in range(80)]
    many_skills = [{"matched_market": [f"s{i}"]} for i in range(60)]

    async def pubs(query: str, limit: int) -> list[dict]:
        return [{"id": i, "title": f"p{i}"} for i in range(30)]

    summary = dict(_TEACHER_SUMMARY)
    summary["top_cross_discipline_gaps"] = many_gaps
    summary["disciplines"] = many_skills
    summary["top_emerging_across_all"] = [{"skill": f"f{i}"} for i in range(20)]

    ctx = await llm_chat.build_context(
        "teacher", "09.03.02", summary_loader=lambda d: summary, pubs_loader=pubs
    )

    assert len(ctx["gaps"]) <= 50
    assert len(ctx["skills"]) <= 40
    assert len(ctx["publications"]) <= 15
    assert len(ctx["frontiers"]) <= 10


async def test_build_context_teacher_default_loader(tmp_path, monkeypatch):
    await _use_default_teacher_loader(tmp_path, monkeypatch)


async def _use_default_teacher_loader(tmp_path, monkeypatch):
    summary_path = tmp_path / "teacher" / "09.03.02"
    summary_path.mkdir(parents=True)
    (summary_path / "_summary.json").write_text(
        json.dumps(
            {
                "direction_name": "Тест",
                "average_coverage": 0.5,
                "disciplines": [{"matched_market": ["Python"]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_chat.config, "DATA_RESULT_DIR", tmp_path)

    async def pubs(query: str, limit: int) -> list[dict]:
        return []

    ctx = await llm_chat.build_context("teacher", "09.03.02", pubs_loader=pubs)

    assert ctx["title"] == "Тест"
    assert "Python" in ctx["skills"]


# ── (g) build_context без данных: пустой контекст, без исключения ───────────

async def _no_pubs(query: str, limit: int) -> list[dict]:
    return []


async def test_build_context_teacher_empty_payload_is_safe():
    """Пустой summary/KRM/публикации -> пустой контекст, исключения нет."""
    ctx = await llm_chat.build_context(
        "teacher",
        "09.03.02",
        summary_loader=lambda dir_code: {},
        krm_loader=lambda ct, ck: [],
        pubs_loader=_no_pubs,
    )

    assert ctx["context_type"] == "teacher"
    assert ctx["title"] == "09.03.02"  # fallback на context_key
    assert ctx["metrics"]["coverage"] is None
    assert ctx["metrics"]["quality"] is None
    assert ctx["skills"] == []
    assert ctx["gaps"] == []
    assert ctx["frontiers"] == []
    assert ctx["competencies"] == []
    assert ctx["publications"] == []


async def test_build_context_student_empty_payload_is_safe():
    """Пустые рекомендации -> пустой контекст, исключения нет."""
    ctx = await llm_chat.build_context(
        "student",
        "somebody",
        recs_loader=lambda profile: {},
        krm_loader=lambda ct, ck: [],
        pubs_loader=_no_pubs,
    )

    assert ctx["context_type"] == "student"
    assert ctx["title"] == "somebody"
    assert ctx["metrics"]["coverage"] is None
    assert ctx["metrics"]["readiness"] is None
    assert ctx["skills"] == []
    assert ctx["gaps"] == []
    assert ctx["frontiers"] == []
    assert ctx["competencies"] == []
    assert ctx["publications"] == []


# ── ответы LLM: выбор моделей ──────────────────────────────────────────────

def test_answer_text_uses_text_model(monkeypatch):
    _RecordingClient.instances = []
    monkeypatch.setattr(llm_chat, "LLMClient", _RecordingClient)
    monkeypatch.setattr(llm_chat.config, "LLM_CHAT_MODEL", "text-model")

    out = llm_chat.answer_text({"metrics": {}})

    assert out == "сводка"
    assert any(c.model == "text-model" for c in _RecordingClient.instances)


def test_answer_blocks_uses_blocks_model(monkeypatch):
    captured: dict = {}

    class Fake:
        def __init__(self, model=None, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

        def generate(self, prompt):
            return '{"blocks": [{"type": "text", "content": "x"}]}'

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)
    monkeypatch.setattr(llm_chat.config, "LLM_CHAT_BLOCKS_MODEL", "blocks-model")

    blocks = llm_chat.answer_blocks({"metrics": {}}, "сводка")

    assert captured["model"] == "blocks-model"
    assert blocks == [{"type": "text", "content": "x"}]


def test_answer_blocks_invalid_json_falls_back_to_text(monkeypatch):
    class Fake:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt):
            return "это не JSON"

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    blocks = llm_chat.answer_blocks({}, "сводка")

    assert blocks == [{"type": "text", "content": "сводка"}]


def test_answer_blocks_wrong_type_falls_back_to_text(monkeypatch):
    class Fake:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt):
            return '{"blocks": [{"type": "bogus", "x": 1}]}'

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    blocks = llm_chat.answer_blocks({}, "сводка")

    assert blocks == [{"type": "text", "content": "сводка"}]


# ── (h) парсинг блоков: граничные варианты невалидного/обёрнутого JSON ──────

def test_answer_blocks_empty_blocks_list_falls_back_to_text(monkeypatch):
    """Валидный JSON с пустым списком блоков -> один text-блок."""
    class Fake:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt):
            return '{"blocks": []}'

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    blocks = llm_chat.answer_blocks({}, "сводка")

    assert blocks == [{"type": "text", "content": "сводка"}]


def test_answer_blocks_json_without_blocks_key_falls_back_to_text(monkeypatch):
    """Валидный JSON без ключа blocks -> один text-блок."""
    class Fake:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt):
            return '{"items": [{"type": "text", "content": "x"}]}'

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    blocks = llm_chat.answer_blocks({}, "сводка")

    assert blocks == [{"type": "text", "content": "сводка"}]


def test_answer_blocks_recovers_json_embedded_in_prose(monkeypatch):
    """JSON обёрнут разговорным текстом - регекс-аварийный путь достаёт блоки."""
    class Fake:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt):
            return (
                'Вот ваш ответ: {"blocks": [{"type": "metric", "name": "coverage", '
                '"value": 82}]}. Надеюсь, помог.'
            )

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    blocks = llm_chat.answer_blocks({}, "сводка")

    assert blocks == [{"type": "metric", "name": "coverage", "value": 82}]


# ── assess: fallback при выключенном флаге ─────────────────────────────────

async def test_assess_disabled_returns_deterministic_text(monkeypatch):
    monkeypatch.setattr(llm_chat.config, "LLM_ASSESS_ENABLED", False)

    ctx = {
        "context_type": "teacher",
        "metrics": {"coverage": 50.0, "quality": 60.0},
        "gaps": [{"skill": "Kafka"}],
    }

    async def fake_build(*args, **kwargs):
        return ctx

    monkeypatch.setattr(llm_chat, "build_context", fake_build)
    # Гарантируем, что LLM-путь не вызывается: answer_text подменена на раис.
    monkeypatch.setattr(llm_chat, "answer_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))

    result = await llm_chat.assess("teacher", "09.03.02")

    assert result["llm_used"] is False
    assert result["blocks"] and result["blocks"][0]["type"] == "text"
    assert result["blocks"][0]["content"]
    assert "Kafka" in result["blocks"][0]["content"]


async def test_assess_enabled_calls_two_models(monkeypatch):
    monkeypatch.setattr(llm_chat.config, "LLM_ASSESS_ENABLED", True)
    monkeypatch.setattr(llm_chat.config, "LLM_CHAT_MODEL", "text-model")
    monkeypatch.setattr(llm_chat.config, "LLM_CHAT_BLOCKS_MODEL", "blocks-model")

    models: list[str] = []

    class Fake:
        def __init__(self, model=None, **kwargs):
            models.append(model)

        def generate(self, prompt):
            # текст для первого вызова, блоки - для второго по модели
            return "сводка"

    monkeypatch.setattr(llm_chat, "LLMClient", Fake)

    ctx = {"metrics": {}, "gaps": []}

    async def fake_build(*args, **kwargs):
        return ctx

    monkeypatch.setattr(llm_chat, "build_context", fake_build)

    result = await llm_chat.assess("teacher", "09.03.02")

    assert result["llm_used"] is True
    assert "text-model" in models
    assert "blocks-model" in models


# ── персистенция ───────────────────────────────────────────────────────────

async def test_append_turn_writes_one_row():
    session = _AppendSession()

    message = await llm_chat.append_turn(
        "thread-1", "assistant", {"blocks": []}, "blocks-model", session=session
    )

    assert isinstance(message, ChatMessage)
    assert len(session.added) == 1
    assert session.added[0].role == "assistant"
    assert session.added[0].seq == 1
    assert session.added[0].blocks_json == {"blocks": []}
    assert session.added[0].model_used == "blocks-model"


async def test_get_or_create_thread_creates_when_absent():
    session = _ThreadSession()

    thread = await llm_chat.get_or_create_thread(
        "user-1", "teacher", "09.03.02", session=session
    )

    assert thread is session.added[0]
    assert thread.context_type == "teacher"
    assert thread.context_key == "09.03.02"
