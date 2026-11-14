"""Синхронность .env.example и настроек приложения (Зона D, 7.4.3).

Две проверки:
1. Обратная: каждый ключ из .env.example либо объявлен в Settings, либо явно
   разрешён (переменные compose/сервисов/тестов). Так ловим опечатки и
   переименованные переменные, которые иначе тихо игнорируются.
2. Прямая: критичные для запуска/безопасности переменные присутствуют в примере
   (чтобы новый разработчик не стартовал «на пустом»).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"

#: Ключи, которые сознательно НЕ поля Settings: compose, сервисы, тесты, рантайм.
EXTRA_OK = {
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "N8N_WEBHOOK_URL",
    "N8N_DB_PASSWORD",
    "N8N_ENCRYPTION_KEY",
    "WEBUI_SECRET_KEY",
    "LOG_LEVEL",
    "APP_ENV",
    "CUDA_VISIBLE_DEVICES",
    "FORCE_COLOR",
    "HF_ENDPOINT",
    "HF_HUB_OFFLINE",
    "LLM_PRICING",
    "SEED_USERS_FILE",
    "TEST_DATABASE_URL",
}

#: Критичные переменные, которые обязаны быть (без комментария) в примере.
REQUIRED_PRESENT = {
    "SECRET_KEY",
    "DATABASE_URL",
    "ALLOWED_ORIGINS",
    "RATELIMIT_ENABLED",
    "TOKEN_TTL_DAYS",
    "COOKIE_SECURE",
    "N8N_API_KEY",
    "N8N_WEBHOOK_SECRET",
    "LLM_CHAT_URL",
    "LLM_CHAT_MODEL",
    "GRAFANA_URL",
    "GRAFANA_USER",
    "GRAFANA_PASSWORD",
    "ACADEMIC_API_BASE",
    "HUB_API_BASE",
    "OLLAMA_URL",
}

KEY_RE = re.compile(r"(?m)^([A-Z_][A-Z0-9_]*)=")


def _example_keys() -> set[str]:
    return set(KEY_RE.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))


def test_example_has_no_unknown_keys() -> None:
    fields = set(Settings.model_fields)
    unknown = sorted(_example_keys() - fields - EXTRA_OK)
    assert not unknown, (
        ".env.example содержит ключи, которых нет ни в Settings, ни в "
        "разрешённом списке (опечатка/переименование?):\n" + "\n".join(unknown)
    )


def test_example_covers_required_keys() -> None:
    present = _example_keys()
    missing = sorted(REQUIRED_PRESENT - present)
    assert not missing, (
        "В .env.example нет критичных переменных:\n" + "\n".join(missing)
    )
