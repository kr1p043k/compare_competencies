"""Тесты анализа запроса: intent, сущности, фильтры, леммы."""

from __future__ import annotations

from src.rag.query import (
    analyze,
    detect_intent,
    entity_pairs,
    extract_competency_codes,
    extract_directions,
    extract_skills,
    lemmatize,
    normalize_query,
    sparse_query_text,
)
from src.rag.settings import AclScope, reset_settings_cache


def setup_function():
    reset_settings_cache()


def test_normalize_query_collapses_spaces():
    assert normalize_query("  почему   УК-3  не покрыт ") == "почему УК-3 не покрыт"


def test_detect_intent_rules():
    assert detect_intent("Почему компетенция УК-3 не покрыта?") == "rpd_gap_explain"
    assert detect_intent("Какие требования рынка по Python?") == "market_evidence"
    assert detect_intent("Что добавить в рабочую программу?") == "curriculum_design"
    assert detect_intent("Что входит в компетенцию ОПК-2?") == "competency_lookup"
    assert detect_intent("Привет, как дела?") == "chat_qa"


def test_extract_competency_codes_and_directions():
    text = "Компетенции УК-3, ОПК-2.1 и ПК 4 по направлению 09.03.02"
    codes = extract_competency_codes(text)
    assert "ук-3" in codes
    assert any(code.startswith("опк-2.1") for code in codes)
    assert extract_directions(text) == ["09.03.02"]


def test_extract_skills_prefers_longest_match():
    vocab = {"машинное обучение", "обучение", "python"}
    found = extract_skills("нужно машинное обучение и python", vocab)
    assert "машинное обучение" in found
    assert "python" in found


def test_lemmatize_drops_stopwords():
    lemmas = lemmatize("Почему компетенции не покрыты в программе")
    assert "компетенция" in lemmas or "компетенции" in lemmas
    assert "не" not in lemmas


def test_analyze_sets_profile_intent_and_filters():
    result = analyze(
        "Почему УК-3 не покрыт по направлению 09.03.02?",
        skill_vocab={"python"},
        year=2025,
    )
    assert result.intent == "rpd_gap_explain"
    assert result.profile == "rpd_gap_explain"
    assert result.filters["direction"] == "09.03.02"
    assert result.filters["year"] == 2025
    assert result.entities["competency"]


def test_entity_pairs_skips_direction():
    result = analyze("Python и направление 09.03.02", skill_vocab={"python"})
    pairs = entity_pairs(result)
    assert ("skill", "python") in pairs
    assert all(entity_type != "direction" for entity_type, _ in pairs)


def test_sparse_query_text_contains_lemmas_and_codes():
    result = analyze("Почему УК-3 не покрыт?", skill_vocab=set())
    text = sparse_query_text(result)
    assert "ук-3" in text


def test_acl_scope_hierarchy():
    assert "teacher" in AclScope.for_role("admin").allowed
    assert "teacher" in AclScope.for_role("teacher").allowed
    assert "teacher" not in AclScope.for_role("student").allowed
    assert AclScope.for_role(None).allowed == ("public",)
