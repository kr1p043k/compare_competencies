"""Барьер: `.env.example` синхронен с кодом (BACKLOG 7.4.3).

Проверяем, что каждый ключ из `Settings` (pydantic) и каждый ключ, читаемый
через `os.getenv`/`os.environ`, присутствует в `.env.example`. Так расхождение
видно сразу, а не при развёртывании.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"

# Переменные окружения, которые не документируем в примере (шум ОС/CI).
IGNORED_ENV = {
    "PATH", "HOME", "USERPROFILE", "TEMP", "TMP", "APPDATA", "LOCALAPPDATA",
    "PYTHONPATH", "PYTHONHASHSEED", "COMPUTERNAME", "USERNAME", "PWD",
    "CI", "GITHUB_ACTIONS", "PYTEST_CURRENT_TEST", "LANG", "LC_ALL", "SHELL",
    "TERM", "PROGRAMFILES", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
    "SECRET_KEY",  # задаётся окружением/секрет-стором, в примере есть
}

KEY_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=", re.M)
GETENV_RE = re.compile(r"os\.(?:getenv|environ\.get)\(\s*\"([A-Z][A-Z0-9_]*)\"")
ENVIRON_RE = re.compile(r"os\.environ\[\s*\"([A-Z][A-Z0-9_]*)\"")


def _example_keys() -> set[str]:
    return set(KEY_RE.findall(ENV_EXAMPLE.read_text(encoding="utf-8", errors="ignore")))


def _settings_keys() -> set[str]:
    os.environ.setdefault("SECRET_KEY", "x" * 64)
    from src.config import settings

    return set(type(settings).model_fields.keys())


def test_settings_keys_documented():
    example = _example_keys()
    missing = sorted(_settings_keys() - example - IGNORED_ENV)
    assert not missing, (
        "В .env.example нет ключей из src/config.py: " + ", ".join(missing)
    )


def test_getenv_keys_documented():
    example = _example_keys()
    used: set[str] = set()
    for p in (ROOT / "src").rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        used |= set(GETENV_RE.findall(text))
        used |= set(ENVIRON_RE.findall(text))
    missing = sorted(used - example - IGNORED_ENV - _settings_keys())
    assert not missing, (
        "В .env.example нет ключей, читаемых через os.getenv/os.environ: " + ", ".join(missing)
    )


def test_no_duplicate_keys_in_example():
    keys = KEY_RE.findall(ENV_EXAMPLE.read_text(encoding="utf-8", errors="ignore"))
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    assert not duplicates, "Дубликаты ключей в .env.example: " + ", ".join(duplicates)


@pytest.mark.parametrize("name", ["DATABASE_URL", "SECRET_KEY", "ALLOWED_ORIGINS"])
def test_critical_keys_present(name: str):
    assert name in _example_keys()
