"""Зонд Course Trends API: показывает реальную структуру ответов.

Ответы сервиса не описаны моделями в OpenAPI (`"schema": {}`), поэтому поля
можно узнать только живым запросом. Скрипт печатает дерево ключей и первые
значения, чтобы по ним спроектировать разбор и UI.

Токен - тот же hub-JWT, что выдаёт hub.sfedu.ru после входа. Его можно
передать аргументом, переменной окружения HUB_JWT или взять из БД: скрипт
берёт самый свежий непустой sso_token из таблицы sessions.

Использование:
    python scripts/probe_course_trends.py
    python scripts/probe_course_trends.py --token <jwt>
    python scripts/probe_course_trends.py --pid 09.03.02_2025 --depth 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.infrastructure.course_trends import CourseTrendsClient, CourseTrendsError  # noqa: E402
try:  # запуск как файла
    from _dsn import resolve_dsn  # noqa: E402
except ImportError:  # импорт из тестов (пакета scripts в sys.path нет)
    from scripts._dsn import resolve_dsn  # noqa: E402

DEFAULT_DSN = ""  # значение берётся из --dsn / DATABASE_URL / .env


def token_from_db(dsn: str) -> str:
    """Самый свежий hub-JWT из сохранённых сессий (если пользователь входил)."""
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=10) as conn:
            row = conn.execute(
                "SELECT sso_token FROM sessions"
                " WHERE sso_token IS NOT NULL AND logged_out_at IS NULL"
                " ORDER BY last_activity DESC NULLS LAST LIMIT 1"
            ).fetchone()
            return (row[0] if row else "") or ""
    except Exception:
        return ""


def describe(value: Any, depth: int, max_depth: int, indent: int = 0) -> list[str]:
    """Дерево ключей: имя, тип, размер, пример значения."""
    pad = "  " * indent
    out: list[str] = []
    if isinstance(value, dict):
        out.append(f"{pad}object({len(value)} keys)")
        if indent >= max_depth:
            return out
        for key, val in list(value.items())[:40]:
            if isinstance(val, (dict, list)):
                out.append(f"{pad}  {key}:")
                out.extend(describe(val, depth + 1, max_depth, indent + 2))
            else:
                shown = str(val)
                if len(shown) > 90:
                    shown = shown[:87] + "..."
                out.append(f"{pad}  {key}: {type(val).__name__} = {shown}")
    elif isinstance(value, list):
        out.append(f"{pad}list({len(value)})")
        if value and indent < max_depth:
            out.append(f"{pad}  [0]:")
            out.extend(describe(value[0], depth + 1, max_depth, indent + 2))
    else:
        out.append(f"{pad}{type(value).__name__} = {value}")
    return out


def dump(label: str, value: Any, depth: int) -> None:
    print(f"\n=== {label} ===")
    for line in describe(value, 0, depth):
        print(line)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token", default=os.environ.get("HUB_JWT", ""),
                   help="hub-JWT; иначе HUB_JWT, иначе самый свежий из sessions")
    p.add_argument("--pid", default=None, help="программа для подробного дампа")
    p.add_argument("--dsn", default="")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--raw", default=None, help="куда сохранить сырой ответ (JSON)")
    args = p.parse_args()
    args.dsn = resolve_dsn(getattr(args, 'dsn', '') or None)

    token = args.token or token_from_db(args.dsn)
    if not token:
        print("Нет токена: передайте --token <jwt>, задайте HUB_JWT или войдите"
              " через hub.sfedu.ru (тогда токен появится в таблице sessions).")
        return 2

    client = CourseTrendsClient(token)
    print(f"base_url: {client.base_url}")

    raw: dict[str, Any] = {}
    try:
        programs = client.list_programs()
        raw["programs"] = programs
        dump("GET /programs", programs, args.depth)
    except CourseTrendsError as exc:
        print(f"/programs: {exc}")
        return 1

    pid = args.pid
    if not pid:
        items = programs if isinstance(programs, list) else (
            programs.get("programs") if isinstance(programs, dict) else []
        )
        if items and isinstance(items[0], dict):
            for key in ("pid", "id", "code", "program_id"):
                if key in items[0]:
                    pid = str(items[0][key])
                    break
        print(f"\npid не задан, взял из первой программы: {pid!r}")

    if pid:
        for name, call in (
            ("GET /programs/{pid}/summary", client.program_summary),
            ("GET /programs/{pid}/course-trends", client.course_trends),
            ("GET /programs/{pid}/course-coverage", client.course_coverage),
            ("GET /programs/{pid}/recommendations", client.recommendations),
        ):
            try:
                value = call(pid)
                raw[name] = value
                dump(name.replace("{pid}", pid), value, args.depth)
            except CourseTrendsError as exc:
                print(f"\n{name}: {exc}")

    try:
        raw["glossary"] = client.glossary()
        dump("GET /meta/glossary", raw["glossary"], 2)
    except CourseTrendsError as exc:
        print(f"\n/meta/glossary: {exc}")

    if args.raw:
        with open(args.raw, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, ensure_ascii=False, indent=2)
        print(f"\nсырой ответ сохранён: {args.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
