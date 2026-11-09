"""Guardrails для LLM: маскирование PII и детект prompt-injection.

Чистые функции без побочек - вызываются до отправки текста в модель. PII
маскируется всегда; признак инъекции отдаём наружу, чтобы вызывающий код решал,
блокировать/логировать/помечать сообщение.
"""
from __future__ import annotations

import re

#: email, телефон РФ, длинные цифровые последовательности (карты/счета).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
_LONG_DIGITS_RE = re.compile(r"(?<!\d)\d{12,19}(?!\d)")

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_EMAIL_RE, "[email]"),
    (_PHONE_RE, "[phone]"),
    (_LONG_DIGITS_RE, "[number]"),
)

#: Эвристики классических prompt-injection. Не претендуют на полноту - это
#: дешёвый сигнал для логирования/эскалации, а не замена модерации.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"disregard\s+.*(instructions|prompt|rules)", re.I),
    re.compile(r"(забудь|игнорируй)\s+.*(инструкц|правил|промпт)", re.I),
    re.compile(r"\bsystem\s*:", re.I),
    re.compile(r"you\s+are\s+now\b", re.I),
    re.compile(r"reveal\s+.*(prompt|instructions|system)", re.I),
    re.compile(r"(повтори|покажи)\s+.*(системн|промпт)", re.I),
)


def redact_pii(text: str) -> str:
    """Маскирует email / телефон / длинные числа. Пустую строку не трогает."""
    if not text:
        return text
    out = text
    for pattern, replacement in _PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def looks_like_injection(text: str) -> bool:
    """True, если текст похож на попытку перехвата инструкций модели."""
    if not text:
        return False
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def guard(text: str) -> tuple[str, bool]:
    """Возвращает (текст без PII, признак возможной инъекции)."""
    return redact_pii(text), looks_like_injection(text)
