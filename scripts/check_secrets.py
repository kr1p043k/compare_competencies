"""Скан секретов в файлах под контролем версий (CI fail-closed).

Ловим только высоконадёжные шаблоны реальных ключей (приватные ключи, AWS,
GitHub, Slack, Google, Stripe, токен Telegram-бота). Никаких «похоже на
секрет по энтропии»: ложные срабатывания в фикстурах и доках недопустимы.
Сканируются файлы из `git ls-files`, поэтому .env/.secrets (gitignore) не
попадают. Строку можно явно исключить комментарием `secret-scan: allow`.

Запуск: python scripts/check_secrets.py [paths...]
Код возврата: 0 - чисто, 1 - найдены секреты.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARKER = "secret-scan: allow"

#: (имя правила, скомпилированный шаблон). Только высоконадёжные.
RULES: list[tuple[str, re.Pattern[str]]] = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("stripe-key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{10,}\b")),
    ("telegram-bot-token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
]

#: Файлы/каталоги, где шаблоны могут встречаться осознанно (пример, тест логики сканера).
ALLOW_PATHS = (
    "scripts/check_secrets.py",
    "tests/test_check_secrets.py",
)

MAX_BYTES = 2 * 1024 * 1024


def scan_text(text: str) -> list[tuple[int, str]]:
    """Вернуть [(номер строки, правило)] без учёта allow-комментариев."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for name, pat in RULES:
            if pat.search(line):
                hits.append((lineno, name))
                break
    return hits


def _tracked_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).is_file()]
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.splitlines() if p]


def scan_file(path: Path) -> list[tuple[int, str]]:
    if path.as_posix() in ALLOW_PATHS:
        return []
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text)


def main(argv: list[str]) -> int:
    findings: list[str] = []
    for path in _tracked_files(argv):
        for lineno, rule in scan_file(path):
            findings.append(f"{path}:{lineno}: {rule}")
    if findings:
        print("Обнаружены возможные секреты:")
        for f in findings:
            print(f"  {f}")
        print(
            "Если это осознанно (пример/фикстура) - добавьте в строку "
            f"комментарий '{ALLOW_MARKER}'."
        )
        return 1
    print("Секреты не найдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
