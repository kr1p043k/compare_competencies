"""Гейт: в src/ не осталось deprecated asyncio.get_event_loop().

В Python 3.12+ get_event_loop() вне работающего цикла помечен deprecated, а в
будущих версиях перестанет создавать цикл. Нужны get_running_loop() (в async)
или явный new_event_loop()/asyncio.run() (в sync-мостах).

Проверка через AST: ловим реальные вызовы `asyncio.get_event_loop(...)`, а не
упоминания в докстрингах/комментариях.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _uses_get_event_loop(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get_event_loop"
            and isinstance(func.value, ast.Name)
            and func.value.id == "asyncio"
        ):
            return True
    return False


def test_no_get_event_loop_in_src() -> None:
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        if _uses_get_event_loop(tree):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "Deprecated asyncio.get_event_loop() найден в:\n" + "\n".join(offenders)
    )
