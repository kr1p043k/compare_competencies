"""Интеграция RAG в llm_chat: контекст, бюджет, цитаты, деградация."""

from __future__ import annotations

from types import SimpleNamespace

from src.rag.types import Citation, RagContext, RetrievalHit
from src.services import llm_chat


def _hit(text: str = "УК-3 формирует способность работать в команде.") -> RetrievalHit:
    return RetrievalHit(
        chunk_id="c1",
        document_id="d1",
        source="rpd",
        title="Математический анализ",
        text=text,
        dense_score=0.81,
        section_path="Компетенции",
        url="https://hub/1",
    )


class FakeRag:
    def __init__(self, *, fail: bool = False, empty: bool = False):
        self.fail = fail
        self.empty = empty
        self.calls: list[dict] = []

    async def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.fail:
            raise RuntimeError("index down")
        if self.empty:
            context = RagContext(degraded="no_hits")
        else:
            hit = _hit()
            context = RagContext(
                evidence=[hit],
                citations=[Citation(n=1, source="rpd", title=hit.title, url=hit.url,
                                    section=hit.section_path, quote=hit.text[:100])],
                prompt_block="<evidence>\n[1] РПД · Математический анализ\n"
                             "УК-3 формирует способность работать в команде.\n</evidence>",
                top_score=0.81,
                token_count=40,
            )
        return SimpleNamespace(context=context, degraded=context.degraded, log_id=None)


class FakeClient:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt, temperature=None, max_tokens=None):
        self.prompts.append(prompt)
        if "JSON:" in prompt:
            return '{"blocks":[{"type":"text","content":"Компетенция покрыта [1]."}]}'
        return "Компетенция УК-3 покрыта дисциплиной [1]."


_TEACHER_PAYLOAD = {
    "direction_name": "Информатика и вычислительная техника",
    "average_coverage": 0.82,
    "average_quality_coverage": 0.71,
    "top_cross_discipline_gaps": [{"skill": "Kafka", "disciplines": 3}],
    "disciplines": [{"matched_market": ["Python"]}],
}


async def _no_pubs(query, limit):
    return []


async def test_build_context_adds_evidence_and_citations():
    rag = FakeRag()
    ctx = await llm_chat.build_context(
        "teacher",
        "09.03.02",
        summary_loader=lambda key: _TEACHER_PAYLOAD,
        krm_loader=lambda ctx_type, key: [],
        pubs_loader=_no_pubs,
        rag=rag,
        rag_query="почему УК-3 не покрыт",
        rag_role="teacher",
    )
    assert ctx["evidence"].startswith("<evidence>")
    assert ctx["citations"][0]["type"] == "citation"
    assert ctx["rag"]["hits"] == 1
    assert rag.calls[0]["query"] == "почему УК-3 не покрыт"
    assert rag.calls[0]["role"] == "teacher"
    assert rag.calls[0]["log"] is False


async def test_build_context_survives_rag_failure():
    ctx = await llm_chat.build_context(
        "teacher",
        "09.03.02",
        summary_loader=lambda key: _TEACHER_PAYLOAD,
        krm_loader=lambda ctx_type, key: [],
        pubs_loader=_no_pubs,
        rag=FakeRag(fail=True),
        rag_query="вопрос",
    )
    assert ctx["evidence"] == ""
    assert ctx["citations"] == []
    assert ctx["metrics"]["coverage"] == 0.82


async def test_context_text_puts_evidence_first():
    ctx = {
        "title": "Тест",
        "metrics": {"coverage": 50},
        "evidence": "<evidence>[1] цитата</evidence>",
        "citations": [{"type": "citation", "n": 1}],
        "rag": {"hits": 1},
    }
    text = llm_chat._context_text(ctx)
    assert text.startswith("<evidence>")
    assert "Факты:" in text
    assert "citations" not in text


async def test_assess_appends_citation_blocks():
    client = FakeClient()
    result = await llm_chat.assess(
        "teacher",
        "09.03.02",
        client=client,
        message="почему УК-3 не покрыт?",
        enabled=True,
        summary_loader=lambda key: _TEACHER_PAYLOAD,
        krm_loader=lambda ctx_type, key: [],
        pubs_loader=_no_pubs,
        use_rag=True,
        rag=FakeRag(),
    )
    assert result["llm_used"] is True
    assert result["rag"]["hits"] == 1
    types = [block["type"] for block in result["blocks"]]
    assert types[-1] == "citation"
    assert result["blocks"][-1]["n"] == 1


async def test_assess_without_rag_is_unchanged():
    client = FakeClient()
    result = await llm_chat.assess(
        "teacher",
        "09.03.02",
        client=client,
        message="что с покрытием?",
        enabled=True,
        summary_loader=lambda key: _TEACHER_PAYLOAD,
        krm_loader=lambda ctx_type, key: [],
        pubs_loader=_no_pubs,
        use_rag=False,
    )
    assert result["context"]["evidence"] == ""
    assert all(block["type"] != "citation" for block in result["blocks"])
    assert result["rag"] == {}


async def test_assess_falls_back_when_rag_empty():
    client = FakeClient()
    result = await llm_chat.assess(
        "teacher",
        "09.03.02",
        client=client,
        message="вопрос без индекса",
        enabled=True,
        summary_loader=lambda key: _TEACHER_PAYLOAD,
        krm_loader=lambda ctx_type, key: [],
        pubs_loader=_no_pubs,
        use_rag=True,
        rag=FakeRag(empty=True),
    )
    assert result["context"]["evidence"] == ""
    assert result["blocks"]
    assert all(block["type"] != "citation" for block in result["blocks"])
