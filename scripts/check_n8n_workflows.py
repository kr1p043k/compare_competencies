"""Валидатор n8n-воркфлоу (BACKLOG 6.7.5).

Проверяет, что каждый `src/n8n/workflows/*.json`:
- корректный JSON (BOM допускаем, но сообщаем как warning - по RFC 8259 BOM в JSON
  недопустим, а n8n/сторонние парсеры могут на нём спотыкаться);
- содержит непустой список `nodes` и словарь `connections`;
- имена узлов уникальны и непусты;
- у узлов есть строковый `type`;
- в тексте нет похожих на секреты литералов (токены/пароли/ключи).

Запуск: `python scripts/check_n8n_workflows.py` (exit 1 при проблемах).
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

WORKFLOWS_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "n8n" / "workflows"

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),               # OpenAI-style
    re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{30,}\b"),          # Telegram bot token
    re.compile(r"(?i)\b(password|secret|api_?key|token)\b\s*[:=]\s*[\"'][^\"'{][^\"']{5,}[\"']"),
)


def validate_file(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Вернуть (problems, warnings)."""
    problems: list[str] = []
    warnings: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        warnings.append("UTF-8 BOM в начале файла (JSON по RFC 8259 без BOM)")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:  # noqa: BLE001 - валидатор должен сообщать, а не падать
        return ([f"некорректный JSON: {exc}"], warnings)

    if not isinstance(data, dict):
        problems.append("корень должен быть объектом")
        return (problems, warnings)

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        problems.append("отсутствует/пуст список 'nodes'")
    else:
        names = [n.get("name") for n in nodes if isinstance(n, dict)]
        if any(not isinstance(n, str) or not n.strip() for n in names):
            problems.append("у части узлов пустое/нестроковое 'name'")
        dupes = sorted({n for n in names if isinstance(n, str) and names.count(n) > 1})
        if dupes:
            problems.append(f"дубли имён узлов: {dupes}")
        if any(not isinstance(n.get("type"), str) for n in nodes if isinstance(n, dict)):
            problems.append("у части узлов нестроковый 'type'")

    if not isinstance(data.get("connections"), dict):
        problems.append("отсутствует словарь 'connections'")

    text = raw.decode("utf-8-sig", errors="ignore")
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            problems.append(f"похоже на секрет в тексте: /{pat.pattern}/")
    return (problems, warnings)


def check_all(directory: pathlib.Path = WORKFLOWS_DIR) -> int:
    files = sorted(directory.glob("*.json"))
    if not files:
        print(f"нет воркфлоу в {directory}")
        return 0
    total_problems = 0
    for path in files:
        problems, warnings = validate_file(path)
        for w in warnings:
            print(f"WARN  {path.name}: {w}")
        for p in problems:
            print(f"ERROR {path.name}: {p}")
        total_problems += len(problems)
    print(f"проверено {len(files)} воркфлоу, проблем: {total_problems}")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(check_all())
