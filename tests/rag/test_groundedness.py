"""Проверки целостности ответа: аудит цитат, отсечение выдуманных источников."""

from __future__ import annotations

from src.services import llm_chat


def test_citation_audit_all_valid():
    citations = [{"type": "citation", "n": 1}, {"type": "citation", "n": 2}]
    valid, groundedness = llm_chat._citation_audit("Покрытие 82% [1]. Пробел Kafka [2].", citations)
    assert valid == {1, 2}
    assert groundedness == 1.0


def test_citation_audit_flags_invented_numbers():
    citations = [{"type": "citation", "n": 1}]
    valid, groundedness = llm_chat._citation_audit("Факт [1] и выдумка [7].", citations)
    assert valid == {1}
    assert groundedness == 0.5


def test_citation_audit_without_references_is_zero():
    citations = [{"type": "citation", "n": 1}]
    valid, groundedness = llm_chat._citation_audit("Ответ без ссылок.", citations)
    assert valid == set()
    assert groundedness == 0.0


def test_parse_blocks_drops_model_supplied_citations():
    raw = (
        '{"blocks": ['
        '{"type": "text", "content": "Покрытие 82% [1]."},'
        '{"type": "citation", "title": "Выдуманный источник", "url": "https://evil.example"}'
        "]}"
    )
    blocks = llm_chat._parse_blocks(raw, "fallback")
    assert [block["type"] for block in blocks] == ["text"]
    assert all("evil.example" not in str(block) for block in blocks)


def test_with_citations_keeps_only_referenced_numbers():
    blocks = [{"type": "text", "content": "Утверждение [2]."}]
    citations = [
        {"type": "citation", "n": 1, "title": "РПД"},
        {"type": "citation", "n": 2, "title": "Вакансия"},
        {"type": "citation", "n": 3, "title": "КРМ"},
    ]
    merged = llm_chat._with_citations(blocks, citations)
    assert [block.get("n") for block in merged if block["type"] == "citation"] == [2]


def test_with_citations_falls_back_to_first_three():
    blocks = [{"type": "text", "content": "Ответ без ссылок."}]
    citations = [{"type": "citation", "n": n, "title": f"src{n}"} for n in (1, 2, 3, 4)]
    merged = llm_chat._with_citations(blocks, citations)
    assert [block.get("n") for block in merged if block["type"] == "citation"] == [1, 2, 3]


def test_with_citations_no_citations_is_noop():
    blocks = [{"type": "text", "content": "текст"}]
    assert llm_chat._with_citations(blocks, None) == blocks
    assert llm_chat._with_citations(blocks, []) == blocks


def test_text_prompt_requires_references_only_with_evidence():
    with_evidence = llm_chat._text_prompt("<evidence>[1] факт</evidence>", "вопрос")
    without = llm_chat._text_prompt('{"metrics": {}}', "вопрос")
    assert "kvadratnyh skobkah" in with_evidence
    assert "kvadratnyh skobkah" not in without


def test_blocks_prompt_mentions_citations_only_with_evidence():
    with_evidence = llm_chat._blocks_prompt("<evidence>[1] факт</evidence>", "сводка")
    without = llm_chat._blocks_prompt('{"metrics": {}}', "сводка")
    assert '"citations"' in with_evidence
    assert '"citations"' not in without
