"""Поиск «тихих» исключений, которые глотают ошибку без следа.

Именно так в проекте терялись данные: битый JSON вакансий молча пропускался,
а ошибка записи связей компетенций не попадала в лог. Снаружи всё выглядело
как «данных нет», а не «данные потерялись».

Скрипт ищет три шаблона:
  * `except ...: pass` — ошибка исчезает полностью;
  * `except ...: return None/[]/{}` — исчезает, но хотя бы с выходом;
  * `except ...: continue` — цикл молча пропускает элемент.

Не всё из этого плохо (парсинг необязательных полей вправе пропускать), поэтому
скрипт только отчитывается и делит находки на «в критичных модулях» и
«остальные». Критичные — там, где теряются данные: pipeline, parsing,
api_pkg/routers, analyzers.

Использование:
    python scripts/audit_silent_excepts.py
    python scripts/audit_silent_excepts.py --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CRITICAL_PARTS = ("pipeline", "parsing", "routers", "analyzers", "infrastructure")
HANDLER_BODY_KINDS = (ast.Pass, ast.Continue)


def _is_silent(handler: ast.ExceptHandler) -> str | None:
    """Какой именно шаблон замалчивания в обработчике."""
    body = handler.body
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, HANDLER_BODY_KINDS):
        return "pass" if isinstance(stmt, ast.Pass) else "continue"
    if isinstance(stmt, ast.Return):
        value = stmt.value
        # `return None` в AST — это Constant(None), а не отсутствие значения.
        if isinstance(value, ast.Constant) and value.value is None:
            return "return None"
        if value is None:
            return "return None"
        if isinstance(value, (ast.List, ast.Dict, ast.Tuple)) and not (
            getattr(value, "elts", None) or getattr(value, "keys", None)
        ):
            return "return empty"
        if isinstance(value, ast.Constant) and value.value in (None, False, "", 0):
            return "return falsy"
    return None


def scan(path: Path) -> list[dict]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        kind = _is_silent(node)
        if kind is None:
            continue
        found.append({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "line": node.lineno,
            "kind": kind,
            "exception": (node.type.id if isinstance(node.type, ast.Name) else
                          getattr(node.type, "attr", "?") if node.type else "bare"),
        })
    return found


def is_critical(rel_path: str) -> bool:
    parent = str(Path(rel_path).parent).replace("\\", "/")
    return any(part in parent.split("/") for part in CRITICAL_PARTS)


def run(json_path: str | None) -> int:
    findings: list[dict] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        findings.extend(scan(path))

    critical = [f for f in findings if is_critical(f["file"])]
    other = [f for f in findings if not is_critical(f["file"])]

    print(f"тихих обработчиков всего: {len(findings)}")
    print(f"  в критичных модулях: {len(critical)}")
    print(f"  прочих: {len(other)}")
    print()
    if critical:
        print("Критичные (здесь теряются данные):")
        for item in critical[:25]:
            print(f"  {item['file']}:{item['line']}  except {item['exception']}: {item['kind']}")
        if len(critical) > 25:
            print(f"  ... и ещё {len(critical) - 25}")

    if json_path:
        Path(json_path).write_text(
            json.dumps({"total": len(findings), "critical": critical, "other": other},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON: {json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", dest="json_path", default=None)
    args = p.parse_args()
    return run(args.json_path)


if __name__ == "__main__":
    raise SystemExit(main())
