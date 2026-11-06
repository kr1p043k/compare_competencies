#!/usr/bin/env python3
"""Проверка версий критичных пакетов против пинов из requirements*.txt.

Скрипт только читает окружение, ничего не устанавливает и не меняет.
Работает на stdlib + importlib.metadata (без packaging и других зависимостей).

Завершается с кодом 1, если хотя бы один пакет отклонился от пина,
и с кодом 0, если всё совпадает.
"""

from __future__ import annotations

import re
import sys
from contextlib import suppress
from importlib import metadata
from pathlib import Path
from typing import NamedTuple

# Пакеты, чья версия критична для локального тестового прогона
# (обоснование - в docs/TEST-ENV.md).
CRITICAL_PACKAGES = (
    "fastapi",
    "starlette",
    "aiohttp",
    "pydantic",
    "sqlalchemy",
    "httpx",
    "pytest",
)

# Порядок чтения: requirements.txt первым, затем requirements-dev.txt
# (dev-файл делает `-r requirements.txt` и добавляет pytest).
REQ_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
)

# Фазы жизни версии (PEP 440, упрощённо): dev < pre < final < post.
PHASE_DEV = 0
PHASE_PRE = 1
PHASE_FIN = 2
PHASE_POST = 3

# Ранг предрелизной метки внутри фазы PRE: a < b < rc.
PRE_RANK = {
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "rc": 3,
    "c": 3,
    "pre": 3,
    "preview": 3,
}

_VERSION_RE = re.compile(
    r"^v?"
    r"(?P<epoch>\d+!)?"
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre_l>rc|alpha|beta|preview|pre|a|b|c))?"
    r"(?:[-_.]?(?P<pre_n>\d+))?"
    r"(?:[-_.]?(?P<post_l>post|rev|r))?"
    r"(?:[-_.]?(?P<post_n>\d+))?"
    r"(?:[-_.]?(?P<dev_l>dev))?"
    r"(?:[-_.]?(?P<dev_n>\d+))?"
    r"(?:\+[0-9a-zA-Z.]+)?"
    r"$",
    re.IGNORECASE,
)


class Version(NamedTuple):
    """Разобранная версия в виде сортируемого кортежа."""

    epoch: int
    release: tuple[int, ...]
    phase: int
    pre_rank: int
    pre_num: int
    post_num: int
    dev_num: int


def _parse_version(text: str) -> Version:
    match = _VERSION_RE.match(text.strip())
    if match is None:
        # Не распознали формат - считаем нулевой версией (не совпадёт ни с чем).
        return Version(0, (0,), PHASE_FIN, 4, 0, 0, 0)

    epoch = int(match.group("epoch")[:-1]) if match.group("epoch") else 0
    release = tuple(int(part) for part in match.group("release").split("."))

    pre_label = (match.group("pre_l") or "").lower()
    post_label = (match.group("post_l") or "").lower()
    dev_label = (match.group("dev_l") or "").lower()

    if dev_label:
        phase = PHASE_DEV
    elif pre_label:
        phase = PHASE_PRE
    elif post_label:
        phase = PHASE_POST
    else:
        phase = PHASE_FIN

    pre_rank = PRE_RANK.get(pre_label, 4)
    pre_num = int(match.group("pre_n")) if match.group("pre_n") else 0
    post_num = int(match.group("post_n")) if match.group("post_n") else 0
    dev_num = int(match.group("dev_n")) if match.group("dev_n") else 0

    return Version(epoch, release, phase, pre_rank, pre_num, post_num, dev_num)


def _cmp(left: Version, right: Version) -> int:
    """Сравнение версий по PEP 440 (упрощённо). -1/0/1."""
    if left.epoch != right.epoch:
        return -1 if left.epoch < right.epoch else 1

    count = max(len(left.release), len(right.release))
    for index in range(count):
        lv = left.release[index] if index < len(left.release) else 0
        rv = right.release[index] if index < len(right.release) else 0
        if lv != rv:
            return -1 if lv < rv else 1

    for lv, rv in (
        (left.phase, right.phase),
        (left.pre_rank, right.pre_rank),
        (left.pre_num, right.pre_num),
        (left.post_num, right.post_num),
        (left.dev_num, right.dev_num),
    ):
        if lv != rv:
            return -1 if lv < rv else 1
    return 0


def _satisfies(installed: str, specifier: str) -> bool:
    """Проверяет, что установленная версия соответствует спецификатору (==,>=,<,...)."""
    specifier = specifier.strip()
    if not specifier or specifier == "*":
        return True

    installed_version = _parse_version(installed)

    for clause in specifier.split(","):
        clause = clause.strip()
        match = re.match(r"^(==|!=|<=|>=|<|>|~=)?\s*(.*)$", clause)
        if match is None:
            return False

        operator = match.group(1) or "=="
        raw_version = match.group(2).strip()
        wanted = _parse_version(raw_version)
        order = _cmp(installed_version, wanted)

        if operator == "==" and order != 0:
            return False
        if operator == "!=" and order == 0:
            return False
        if operator == ">=" and order < 0:
            return False
        if operator == "<=" and order > 0:
            return False
        if operator == ">" and order <= 0:
            return False
        if operator == "<" and order >= 0:
            return False
        if operator == "~=":
            # ~=X.Y означает >=X.Y и совпадение по всем сегментам, кроме последнего.
            if order < 0:
                return False
            wanted_release = wanted.release
            if len(wanted_release) >= 2:
                prefix = wanted_release[:-1]
                if installed_version.release[: len(prefix)] != prefix:
                    return False

    return True


def _parse_requirement(line: str) -> tuple[str | None, str | None]:
    """Возвращает (каноничное имя, спецификатор) для строки требований или (None, None)."""
    text = line.strip()
    if not text or text.startswith("#") or text.startswith("-"):
        return None, None

    # Отрезаем маркеры окружения: name==1.0; python_version < "3.12"
    if ";" in text:
        text = text.split(";", 1)[0].strip()

    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*(\[[^\]]*\])?)\s*(.*)$", text)
    if match is None:
        return None, None

    raw_name = match.group(1)
    specifier = match.group(3).strip()
    # Канонизация имени по правилам pip: регистр не важен, [-_.] эквивалентны.
    name = re.sub(r"[-_.]+", "-", raw_name.split("[", 1)[0]).lower()
    return name, specifier


def _collect_pins(root: Path) -> dict[str, str]:
    """Собирает карту имя -> спецификатор из requirements*.txt (с учётом -r)."""
    pins: dict[str, str] = {}
    seen: set[Path] = set()

    def load(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            return
        seen.add(resolved)

        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("-r "):
                include = line[3:].strip().split(" ")[0]
                load(path.parent / include)
                continue
            name, specifier = _parse_requirement(line)
            if name is not None:
                pins[name] = specifier or "*"

    for filename in REQ_FILES:
        load(root / filename)

    return pins


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _setup_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


def main() -> int:
    _setup_encoding()

    root = Path(__file__).resolve().parent.parent
    pins = _collect_pins(root)

    print("Проверка окружения тестов (только чтение, ничего не устанавливается).")
    print(f"Пины прочитаны из: {', '.join(REQ_FILES)} (корень: {root})")
    print()

    header = f"{'ПАКЕТ':<12}  {'ПИН (requirements)':<26}  {'УСТАНОВЛЕНО':<15}  СТАТУС"
    separator = "-" * len(header)
    print(header)
    print(separator)

    mismatches = 0
    rows: list[tuple[str, str, str, str, bool]] = []

    for name in CRITICAL_PACKAGES:
        specifier = pins.get(name)
        installed = _installed_version(name)

        if specifier is None:
            status, ok = "нет пина", True
        elif installed is None:
            status, ok = "НЕ УСТАНОВЛЕН", False
        elif _satisfies(installed, specifier):
            status, ok = "OK", True
        else:
            status, ok = "ДРИФТ", False

        if not ok:
            mismatches += 1

        pin_text = specifier if specifier is not None else "-"
        installed_text = installed if installed is not None else "-"
        rows.append((name, pin_text, installed_text, status, ok))

    for name, pin_text, installed_text, status, _ok in rows:
        print(f"{name:<12}  {pin_text:<26}  {installed_text:<15}  {status}")

    print(separator)
    total = len(CRITICAL_PACKAGES)
    if mismatches:
        print(f"Итог: расхождение у {mismatches} из {total} критичных пакетов.")
        return 1
    print(f"Итог: все {total} критичных пакетов соответствуют пинам.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
