"""Удаление «навыков»-предложений из опорного KRM-JSON.

Парсер изредка кладёт в поле skills не навык, а абзац из методички или
служебную строку: «Дисциплина преподается в 6 и 7 семестрах.», «123»,
«2.4», пустую строку или запись вообще без букв. Такие «навыки» не
сопоставляются с рынком, раздувают знаменатель покрытия и искажают
рекомендации.

Критерии мусора (консервативные):
  - пустая строка или короче 2 символов;
  - чистое число/код вида ^\\d+(\\.\\d+)?$;
  - похоже на предложение: заканчивается на ".", либо длиннее 60 символов,
    либо 5+ слов и слово методички («преподается», «семестр», «дисциплина»…);
  - в строке вообще нет букв.

Базу данных скрипт не трогает: работаем только с файлами data/reference.
По умолчанию режим dry-run; флаг --apply переписывает файл на месте и кладёт
рядом бэкап krm_..._backup_prune.json.

Использование:
    python scripts/prune_prose_skills.py
    python scripts/prune_prose_skills.py --apply
    python scripts/prune_prose_skills.py --file data/reference/krm_disciplines_09.03.02_2024.json --apply
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"

DEFAULT_FILE_NAMES = (
    "krm_disciplines_09.03.02.json",
    "krm_disciplines_09.03.02_2024.json",
)

MIN_LEN = 2
MAX_LEN = 60
MIN_WORDS = 5
NUMBER_RE = re.compile(r"^\d+(\.\d+)?$")
PROSE_WORDS = ("преподается", "изучается", "формируется", "семестр", "дисциплин", "дисциплина")
EMPTY_MARKER = "останется 0 навыков"


def junk_reason(name: object) -> str | None:
    """Почему строку считаем мусором. None - похоже на настоящий навык."""
    if not isinstance(name, str):
        return "не строка"
    text = name.strip()
    if not text:
        return "пусто"
    if len(text) < MIN_LEN:
        return "короче двух символов"
    if NUMBER_RE.fullmatch(text):
        return "число или код"
    if not any(ch.isalpha() for ch in text):
        return "нет букв"
    if text.endswith("."):
        return "похоже на предложение"
    if len(text) > MAX_LEN:
        return "длинная строка"
    if len(text.split()) >= MIN_WORDS and any(word in text.lower() for word in PROSE_WORDS):
        return "фраза методички"
    return None


def is_junk(name: object) -> bool:
    """True, если строку считаем мусором."""
    return junk_reason(name) is not None


@dataclass
class DisciplinePlan:
    """Итог разбора одной дисциплины: что оставляем, что удаляем."""

    name: str
    kept: dict[str, list[str]]
    removed: dict[str, list[str]]

    @property
    def removed_total(self) -> int:
        return sum(len(items) for items in self.removed.values())

    @property
    def would_be_empty(self) -> bool:
        return self.removed_total > 0 and not any(self.kept.values())


def plan_prune(data: dict[str, Any]) -> list[DisciplinePlan]:
    """Разобрать опорную структуру: для каждой дисциплины kept/removed."""
    if not data:
        return []
    sub = next(iter(data.values()))
    if not isinstance(sub, dict):
        return []
    disciplines = sub.get("disciplines", {})
    if not isinstance(disciplines, dict):
        return []
    plans: list[DisciplinePlan] = []
    for name, body in disciplines.items():
        if not isinstance(body, dict):
            continue
        skills = body.get("skills")
        if not isinstance(skills, dict):
            continue
        kept: dict[str, list[str]] = {}
        removed: dict[str, list[str]] = {}
        for code, items in skills.items():
            if not isinstance(items, list):
                continue
            kept[code] = [s for s in items if not is_junk(s)]
            removed[code] = [s for s in items if is_junk(s)]
        plans.append(DisciplinePlan(name=name, kept=kept, removed=removed))
    return plans


def summarize(plans: list[DisciplinePlan]) -> tuple[int, list[str]]:
    """Суммарное число к удалению и список дисциплин, которые опустели бы."""
    removed_total = 0
    empty_disciplines: list[str] = []
    for plan in plans:
        if plan.would_be_empty:
            empty_disciplines.append(plan.name)
            continue
        removed_total += plan.removed_total
    return removed_total, empty_disciplines


def apply_prune(data: dict[str, Any], plans: list[DisciplinePlan]) -> None:
    """Удалить мусор из data на месте, пропуская дисциплины, которые опустели бы."""
    if not data:
        return
    sub = next(iter(data.values()))
    if not isinstance(sub, dict):
        return
    disciplines = sub.get("disciplines", {})
    if not isinstance(disciplines, dict):
        return
    for plan in plans:
        if plan.would_be_empty:
            continue
        body = disciplines.get(plan.name)
        if not isinstance(body, dict):
            continue
        skills = body.get("skills")
        if not isinstance(skills, dict):
            continue
        for code, kept in plan.kept.items():
            skills[code] = kept


def default_files() -> list[Path]:
    """Два опорных файла 09.03.02, если они есть на диске."""
    files: list[Path] = []
    for name in DEFAULT_FILE_NAMES:
        path = REFERENCE_DIR / name
        if path.exists():
            files.append(path)
    return files


def resolve_files(file_arg: str | None) -> list[Path]:
    """Файл из --file, либо оба опорных файла по умолчанию."""
    if file_arg:
        path = Path(file_arg)
        if not path.is_file():
            raise FileNotFoundError(f"файл не найден: {path}")
        return [path]
    files = default_files()
    if not files:
        raise FileNotFoundError("не найдено опорных KRM-файлов 09.03.02 в data/reference")
    return files


def backup_path(path: Path) -> Path:
    """Путь бэкапа рядом с исходным файлом: krm_..._backup_prune.json."""
    return path.with_name(path.stem + "_backup_prune.json")


def process_file(path: Path, apply: bool) -> int:
    """Разобрать один файл, показать находки и (при --apply) переписать его."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    plans = plan_prune(data)
    removed_total, empty_disciplines = summarize(plans)

    print(f"\n{path.name}: дисциплин {len(plans)}, строк к удалению {removed_total}")
    for plan in plans:
        if plan.removed_total == 0:
            continue
        if plan.would_be_empty:
            print(f"  [!] «{plan.name}»: {EMPTY_MARKER} - не трогаем ({plan.removed_total})")
            continue
        print(f"  «{plan.name}»: {plan.removed_total}")
        for items in plan.removed.values():
            for item in items:
                print(f"      [{junk_reason(item)}] {repr(item)[:70]}")

    if not apply:
        return removed_total

    backup_path(path).write_text(raw, encoding="utf-8")
    apply_prune(data, plans)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if "\r\n" in raw:
        text = text.replace("\n", "\r\n")
    path.write_text(text, encoding="utf-8")
    for name in empty_disciplines:
        print(f"  пропущено ({EMPTY_MARKER}): «{name}»")
    print(f"  применено, бэкап: {backup_path(path).name}")
    return removed_total


def main(argv: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="конкретный KRM-файл (по умолчанию оба 09.03.02)")
    parser.add_argument("--apply", action="store_true", help="переписать файл (иначе dry-run)")
    args = parser.parse_args(argv)

    files = resolve_files(args.file)
    total = 0
    for path in files:
        total += process_file(path, apply=args.apply)
    suffix = "" if args.apply else " (dry-run; добавьте --apply)"
    print(f"\nитого строк к удалению: {total}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
