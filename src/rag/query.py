"""Анализ запроса: нормализация, леммы, intent, сущности, фильтры.

Intent определяется правилами по ключевым словам, а не LLM: дешевле,
детерминированно и тестируется без сети. Ошибка классификации не критична —
профиль влияет только на набор источников и веса веток.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from src.rag.settings import get_settings
from src.rag.types import RagQuery, normalize_entity_key

_WORD_RE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9+#._\-]*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
#: Коды компетенций: УК-3, ОПК-2.1, ПК-1.2.3, ПСК-4.
_COMPETENCY_RE = re.compile(r"\b(?:УК|ОПК|ПК|ПСК|УК|ОК|ВК)[-\s]?\d+(?:\.\d+)*\b", re.IGNORECASE)
#: Коды направлений: 09.03.02, 01.03.01.
_DIRECTION_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{2}\b")

_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "rpd_gap_explain",
        ("почему", "не покрыт", "непокрыт", "пробел", "gap", "дефицит", "не хватает", "отсутству"),
    ),
    (
        "market_evidence",
        ("рынок", "рынка", "ваканси", "работодател", "hh", "требовани", "зарплат", "спрос"),
    ),
    (
        "curriculum_design",
        ("добавить", "включить", "пересмотр", "учебный план", "рабочую программу", "рпд", "дисциплин"),
    ),
    (
        "competency_lookup",
        ("что входит", "какие навыки", "какая компетенц", "состав компетенц", "расшифруй"),
    ),
)

_STOPWORDS = frozenset(
    {
        "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все",
        "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по",
        "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из",
        "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
        "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
        "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
        "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "is", "are",
    }
)


def normalize_query(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


@lru_cache(maxsize=1)
def _morph():
    try:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


def lemmatize(text: str, *, limit: int = 64) -> list[str]:
    """Леммы значимых слов (pymorphy3), без стоп-слов. Без pymorphy — lower-токены."""
    words = _WORD_RE.findall((text or "").lower())
    morph = _morph()
    out: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        lemma = word
        if morph is not None:
            try:
                parsed = morph.parse(word)
                if parsed:
                    lemma = parsed[0].normal_form
            except Exception:
                lemma = word
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append(lemma)
        if len(out) >= limit:
            break
    return out


def detect_intent(text: str) -> str:
    lowered = (text or "").lower()
    for intent, markers in _INTENT_RULES:
        if any(marker in lowered for marker in markers):
            return intent
    return "chat_qa"


def extract_competency_codes(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _COMPETENCY_RE.findall(text or ""):
        code = normalize_entity_key(match.replace(" ", "-"))
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def extract_directions(text: str) -> list[str]:
    out: list[str] = []
    for match in _DIRECTION_RE.findall(text or ""):
        if match not in out:
            out.append(match)
    return out


def extract_skills(text: str, vocab: Iterable[str], *, limit: int = 12) -> list[str]:
    """Найти в тексте названия навыков из словаря (по границам слов).

    Длинные названия проверяются первыми: «машинное обучение» должно победить
    «обучение», иначе сущность определяется неверно.
    """
    haystack = " " + normalize_entity_key(text) + " "
    found: list[str] = []
    seen: set[str] = set()
    for name in sorted((n for n in vocab if n), key=len, reverse=True):
        key = normalize_entity_key(name)
        if not key or key in seen:
            continue
        if len(key) < 2:
            continue
        if f" {key} " in haystack or f" {key}," in haystack or f" {key}." in haystack:
            seen.add(key)
            found.append(key)
            if len(found) >= limit:
                break
    return found


def analyze(
    text: str,
    *,
    profile: str | None = None,
    skill_vocab: Iterable[str] | None = None,
    direction: str | None = None,
    year: int | None = None,
) -> RagQuery:
    settings = get_settings()
    normalized = normalize_query(text)
    intent = detect_intent(normalized)
    entities: dict[str, list[str]] = {}

    codes = extract_competency_codes(normalized)
    if codes:
        entities["competency"] = codes
    skills = extract_skills(normalized, skill_vocab or ())
    if skills:
        entities["skill"] = skills
    dirs = extract_directions(normalized)
    if dirs:
        entities["direction"] = dirs

    filters: dict[str, object] = {}
    effective_direction = direction or (dirs[0] if dirs else None)
    if effective_direction:
        filters["direction"] = effective_direction
    if year is not None:
        filters["year"] = int(year)

    effective_profile = profile or intent
    if effective_profile not in settings.profile_names:
        effective_profile = settings.profile_default

    return RagQuery(
        raw=text or "",
        normalized=normalized,
        lemmas=lemmatize(normalized),
        intent=intent,
        profile=effective_profile,
        entities=entities,
        filters=filters,
    )


def entity_pairs(query: RagQuery) -> list[tuple[str, str]]:
    """Плоский список (тип, ключ) для entity-ветки поиска."""
    pairs: list[tuple[str, str]] = []
    for entity_type, keys in query.entities.items():
        if entity_type == "direction":
            continue
        for key in keys:
            pairs.append((entity_type, key))
    return pairs


def sparse_query_text(query: RagQuery) -> str:
    """Текст для plainto_tsquery: леммы + коды компетенций как есть."""
    parts = list(query.lemmas)
    parts.extend(query.entities.get("competency", []))
    return " ".join(dict.fromkeys(parts))
