"""Проверка, что установленные версии совпадают с пинами requirements.

Реальный инцидент: `requirements.txt` требовал `aiohttp==3.12.14`, в окружении
стояла 3.14.4 (её подняла другая сессия), и 15 тестов упали на несовместимости
`aioresponses`. Ошибка выглядела как баг в коде парсера, а причина была в
расхождении версий.

Проверка сравнивает жёсткие пины (`==`) из requirements-файлов с тем, что
реально установлено. Диапазоны (`>=`, `~=`) пропускаются: они не обещают
конкретную версию.

Использование:
    python scripts/check_dependency_drift.py
    python scripts/check_dependency_drift.py --strict   # exit 1 при расхождении
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = ("requirements.txt", "requirements-dev.txt")

# Имя пакета и версия: "aiohttp==3.12.14", "torch==2.5.0 ; python_version..."
PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+)\s*==\s*(?P<version>[A-Za-z0-9_.+\-]+)\s*(?:;.*)?$"
)


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(paths: list[Path]) -> dict[str, str]:
    """{каноническое имя: версия} по жёстким пинам из файлов."""
    pins: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            m = PIN_RE.match(line)
            if m:
                pins[_canonical(m.group("name"))] = m.group("version")
    return pins


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check(files: list[Path] | None = None) -> list[dict]:
    paths = files if files is not None else [ROOT / name for name in DEFAULT_FILES]
    pins = parse_pins(paths)
    problems: list[dict] = []
    for name, expected in sorted(pins.items()):
        actual = installed_version(name)
        if actual is None:
            problems.append({"package": name, "expected": expected,
                             "actual": None, "kind": "не установлен"})
        elif actual != expected:
            problems.append({"package": name, "expected": expected,
                             "actual": actual, "kind": "версия отличается"})
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true",
                   help="вернуть код 1, если найдены расхождения")
    args = p.parse_args()

    problems = check()
    if not problems:
        print("расхождений с requirements нет")
        return 0

    print(f"расхождений: {len(problems)}")
    for item in problems:
        print(f"  {item['package']}: ожидается {item['expected']}, "
              f"установлено {item['actual'] or '—'} ({item['kind']})")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
