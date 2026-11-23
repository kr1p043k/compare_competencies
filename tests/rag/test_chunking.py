"""Тесты чанкеров RAG: упаковка, boilerplate, РПД-разделы, вакансии."""

from __future__ import annotations

from src.rag.chunking.base import (
    normalize_text,
    pack_blocks,
    split_paragraphs,
    strip_boilerplate,
)
from src.rag.chunking.rpd import _split_sections, chunk_rpd_document
from src.rag.chunking.vacancy import chunk_vacancy
from src.rag.types import estimate_tokens


def test_normalize_text_collapses_whitespace():
    assert normalize_text("a\t\tb   c\n\n\n\nd") == "a b c\n\nd"


def test_strip_boilerplate_drops_contacts_pages_and_toc():
    raw = (
        "Иванов И.И., доцент, ivanov@sfedu.ru\n"
        "стр. 12\n"
        "Содержание ........ 15\n"
        "Компетенция УК-3 формирует способность работать в команде.\n"
    )
    cleaned = strip_boilerplate(raw)
    assert "ivanov@sfedu.ru" not in cleaned
    assert "стр. 12" not in cleaned
    assert "Компетенция УК-3" in cleaned


def test_split_paragraphs_splits_long_block():
    long_block = " ".join(f"Предложение номер {i} про навыки." for i in range(200))
    parts = split_paragraphs(long_block)
    assert len(parts) > 1
    assert all(estimate_tokens(part) <= 250 for part in parts)


def test_pack_blocks_respects_max_tokens():
    blocks = [" ".join(["слово"] * 60) for _ in range(10)]
    chunks = pack_blocks(blocks, max_tokens=100, overlap_tokens=0, min_tokens=10)
    assert chunks
    assert all(estimate_tokens(chunk) <= 130 for chunk in chunks)


def test_pack_blocks_overlap_repeats_tail():
    blocks = ["первый абзац", "второй абзац", "третий абзац"]
    chunks = pack_blocks(blocks, max_tokens=6, overlap_tokens=3, min_tokens=1)
    assert len(chunks) >= 2


def test_split_sections_recognizes_rpd_headers():
    text = (
        "1. Цели освоения дисциплины\nЦель — научить.\n"
        "2. Задачи дисциплины\nЗадачи — сформировать.\n"
        "3. Компетенции\nУК-3, ОПК-2.\n"
        "5. Содержание дисциплины\nРаздел 1.\n"
    )
    sections = dict(_split_sections(text))
    assert "Цели" in sections
    assert "Задачи" in sections
    assert "Компетенции" in sections
    assert "Содержание" in sections


def test_chunk_rpd_document_returns_none_for_short_text():
    assert chunk_rpd_document({"text": "мало текста", "name": "X"}) is None


def test_chunk_rpd_document_builds_entities_and_sections():
    body = "Компетенция УК-3. " * 40 + "Содержание: " + "работа в команде. " * 40
    doc = chunk_rpd_document(
        {"text": body, "name": "Математический анализ", "file": "a.pdf", "competencies": ["УК-3"]},
        direction="09.03.02",
        year=2025,
    )
    assert doc is not None
    assert doc.source == "rpd"
    assert doc.direction == "09.03.02"
    assert doc.chunks
    entity_types = {entity[0] for chunk in doc.chunks for entity in chunk.entities}
    assert "discipline" in entity_types
    assert "competency" in entity_types


def test_chunk_vacancy_marks_partial_when_no_description():
    doc = chunk_vacancy(
        {
            "hh_id": 1,
            "name": "Data Scientist",
            "snippet_requirement": "Опыт Python, SQL, машинное обучение и статистика.",
            "key_skills": ["Python", "SQL"],
        }
    )
    assert doc is not None
    assert doc.meta["partial"] is True
    assert doc.external_id == "1"


def test_chunk_vacancy_skips_empty():
    assert chunk_vacancy({"hh_id": 2, "name": "X"}) is None


def test_chunk_vacancy_reads_hh_api_shape():
    """hh search отдаёт snippet.requirement и key_skills как [{"name": ...}]."""
    doc = chunk_vacancy(
        {
            "id": "139019168",
            "name": "Data Scientist",
            "snippet": {
                "requirement": "Опыт Python, SQL, машинное обучение и статистика.",
                "responsibility": "Строить модели и анализировать данные.",
            },
            "key_skills": [{"name": "Python"}, {"name": "SQL"}],
            "employer": {"name": "ООО Ромашка"},
            "area": {"name": "Москва"},
        }
    )
    assert doc is not None
    assert doc.external_id == "139019168"
    assert doc.meta["partial"] is True
    assert doc.meta["skills"] == ["Python", "SQL"]
    skill_entities = {
        entity[1] for chunk in doc.chunks for entity in chunk.entities if entity[0] == "skill"
    }
    assert {"python", "sql"} <= skill_entities
    text = " ".join(chunk.text for chunk in doc.chunks)
    assert "машинное обучение" in text
