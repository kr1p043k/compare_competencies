"""Чанкеры источников RAG."""

from src.rag.chunking.base import (
    build_chunks,
    normalize_text,
    pack_blocks,
    split_paragraphs,
    split_sentences,
    strip_boilerplate,
)
from src.rag.chunking.card import (
    competency_card_text,
    discipline_card_text,
    iter_krm_cards,
    iter_skill_cards,
    skill_card_text,
)
from src.rag.chunking.rpd import chunk_rpd_document, iter_rpd_documents
from src.rag.chunking.vacancy import chunk_vacancy

__all__ = [
    "build_chunks",
    "normalize_text",
    "pack_blocks",
    "split_paragraphs",
    "split_sentences",
    "strip_boilerplate",
    "competency_card_text",
    "discipline_card_text",
    "iter_krm_cards",
    "iter_skill_cards",
    "skill_card_text",
    "chunk_rpd_document",
    "iter_rpd_documents",
    "chunk_vacancy",
]
