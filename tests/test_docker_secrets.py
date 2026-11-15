"""Барьер: секреты не попадают в Docker-образ (BACKLOG 7.4.4).

Проверяем статически: .dockerignore исключает .env/секреты/кэши, а Dockerfile
копирует только явно перечисленные пути (не `.env`, не `.secrets`).
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKERFILE = ROOT / "Dockerfile"


def _ignore_lines() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_dockerignore_excludes_env_and_secrets():
    patterns = set(_ignore_lines())
    for required in (".env", ".env*", ".secrets/", "*.pem", "*.key"):
        assert required in patterns, f".dockerignore не исключает {required!r}"


def test_dockerignore_excludes_caches():
    patterns = set(_ignore_lines())
    for required in (".mypy_cache/", ".ruff_cache/", ".pytest_cache/", "htmlcov/", "*.sqlite"):
        assert required in patterns, f".dockerignore не исключает {required!r}"


def test_dockerfile_copies_only_explicit_paths():
    text = DOCKERFILE.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stmt = line.strip()
        if not stmt.upper().startswith(("COPY ", "ADD ")):
            continue
        # COPY src/ ./src/ -> источник не должен быть .env/.secrets/корнем-с-секретами
        src = re.split(r"\s+", stmt)[1]
        assert ".env" not in src, f"Dockerfile копирует env: {stmt}"
        assert ".secrets" not in src, f"Dockerfile копирует секреты: {stmt}"
