"""Гейт целостности документации: относительные ссылки в docs/*.md ведут к файлам.

Ловит переименования и удаления, из-за которых ссылки в доках гниют. Проверяем
только локальные markdown-ссылки вида `[текст](путь)`: http(s), mailto, якоря
`#...` и пути с плейсхолдерами `<...>` пропускаем. Ссылка ищется относительно
самого файла и относительно корня репозитория (в доках встречается и то и то).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _is_checkable(target: str) -> bool:
    target = target.strip()
    if not target:
        return False
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if "<" in target or ">" in target:
        return False
    return not target.startswith("data:")


def _resolves(doc: Path, target: str) -> bool:
    target = target.split("#", 1)[0].strip()
    if not target:
        return True  # чисто якорь - не проверяем
    return (doc.parent / target).exists() or (REPO / target).exists()


def test_relative_doc_links_resolve() -> None:
    broken: list[str] = []
    for doc in sorted(DOCS.rglob("*.md")):
        text = doc.read_text(encoding="utf-8")
        for raw in LINK_RE.findall(text):
            if _is_checkable(raw) and not _resolves(doc, raw):
                broken.append(f"{doc.relative_to(REPO)} -> {raw.strip()}")
    assert not broken, "Битые относительные ссылки в документации:\n" + "\n".join(broken)
