"""Базовый чанкер: нормализация текста, абзацы, упаковка с перекрытием.

Почему не готовый сплиттер: корпус русскоязычный, с PDF-артефактами и
таблицами. Нужны предсказуемые правила (минимум/максимум токенов,
перекрытие, отбрасывание boilerplate), которые легко протестировать.
"""

from __future__ import annotations

import re

from src.rag.types import Chunk, estimate_tokens

#: Колонтитулы, контакты, служебные пометки — выкидываем до чанкования.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)")
_PAGE_NO_RE = re.compile(r"^\s*(?:стр\.?|page)\s*\d+\s*(?:из\s*\d+)?\s*$", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_MULTI_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")

#: Минимальная длина строки, ниже — мусор (номера, обрывки таблиц).
_MIN_LINE_LEN = 12


def normalize_text(text: str) -> str:
    """Схлопнуть пробелы, убрать пустые строки, привести к единому виду."""
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _MULTI_WS_RE.sub(" ", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    return _MULTI_NL_RE.sub("\n\n", out).strip()


def strip_boilerplate(text: str) -> str:
    """Убрать email/телефоны/номера страниц/оглавления — они шумят в retrieval."""
    if not text:
        return ""
    out = _EMAIL_RE.sub(" ", text)
    out = _PHONE_RE.sub(" ", out)
    lines: list[str] = []
    for line in out.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if _PAGE_NO_RE.match(stripped):
            continue
        if _TOC_LINE_RE.search(stripped):
            continue
        if len(stripped) < _MIN_LINE_LEN and not stripped.endswith((".", ":", "?")):
            continue
        lines.append(stripped)
    return normalize_text("\n".join(lines))


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Абзацы по пустой строке; одиночные длинные строки дробим по предложениям."""
    if not text:
        return []
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n", text):
        block = raw.strip()
        if not block:
            continue
        if estimate_tokens(block) > 400:
            buffer: list[str] = []
            for sentence in split_sentences(block):
                buffer.append(sentence)
                if estimate_tokens(" ".join(buffer)) >= 200:
                    blocks.append(" ".join(buffer))
                    buffer = []
            if buffer:
                blocks.append(" ".join(buffer))
        else:
            blocks.append(block)
    return blocks


def pack_blocks(
    blocks: list[str],
    *,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
    min_tokens: int = 40,
) -> list[str]:
    """Упаковать абзацы в чанки ≤ max_tokens с перекрытием хвоста.

    Перекрытие берётся из последних абзацев предыдущего чанка — так разрез
    не рвёт мысль, а дубли в выдаче потом снимает дедуп по content_hash.
    """
    if not blocks:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(current).strip()
        if estimate_tokens(text) >= min_tokens:
            chunks.append(text)
        if overlap_tokens > 0:
            tail: list[str] = []
            tail_tokens = 0
            for block in reversed(current):
                block_tokens = estimate_tokens(block)
                if tail_tokens + block_tokens > overlap_tokens and tail:
                    break
                tail.insert(0, block)
                tail_tokens += block_tokens
            current = tail
            current_tokens = tail_tokens
        else:
            current = []
            current_tokens = 0

    for block in blocks:
        block_tokens = estimate_tokens(block)
        if block_tokens > max_tokens:
            flush()
            current, current_tokens = [], 0
            for sentence in split_sentences(block):
                sentence_tokens = estimate_tokens(sentence)
                # PDF без пунктуации даёт «предложения» на целую страницу:
                # такие режем по словам, иначе чанк вылезет за max_tokens.
                if sentence_tokens > max_tokens:
                    if current:
                        text = " ".join(current).strip()
                        if estimate_tokens(text) >= min_tokens:
                            chunks.append(text)
                        current, current_tokens = [], 0
                    words = sentence.split()
                    buffer: list[str] = []
                    buffer_tokens = 0
                    for word in words:
                        word_tokens = estimate_tokens(word) + 1
                        if buffer_tokens + word_tokens > max_tokens and buffer:
                            text = " ".join(buffer).strip()
                            if estimate_tokens(text) >= min_tokens:
                                chunks.append(text)
                            buffer, buffer_tokens = [], 0
                        buffer.append(word)
                        buffer_tokens += word_tokens
                    current = buffer
                    current_tokens = buffer_tokens
                    continue
                if current_tokens + sentence_tokens > max_tokens and current:
                    text = " ".join(current).strip()
                    if estimate_tokens(text) >= min_tokens:
                        chunks.append(text)
                    current, current_tokens = [], 0
                current.append(sentence)
                current_tokens += sentence_tokens
            continue
        if current_tokens + block_tokens > max_tokens and current:
            flush()
        current.append(block)
        current_tokens += block_tokens

    if current:
        text = "\n\n".join(current).strip()
        if estimate_tokens(text) >= min_tokens and text not in chunks:
            chunks.append(text)
    return chunks


def build_chunks(
    sections: list[tuple[str, str]],
    *,
    max_tokens: int = 700,
    min_tokens: int = 40,
    overlap_tokens: int = 80,
    entities: list[tuple[str, str]] | None = None,
    meta: dict | None = None,
) -> list[Chunk]:
    """Собрать чанки из пар (section_path, text).

    ``entities`` применяются ко всем чанкам документа: точный поиск «всё про
    сущность» не должен зависеть от того, попала ли она в конкретный чанк.
    """
    chunks: list[Chunk] = []
    ord_value = 0
    for section_path, text in sections:
        clean = strip_boilerplate(text)
        if not clean:
            continue
        blocks = split_paragraphs(clean)
        for piece in pack_blocks(
            blocks, max_tokens=max_tokens, overlap_tokens=overlap_tokens, min_tokens=min_tokens
        ):
            chunks.append(
                Chunk(
                    text=piece,
                    ord=ord_value,
                    section_path=section_path,
                    entities=list(entities or []),
                    meta=dict(meta or {}),
                )
            )
            ord_value += 1
    return chunks
