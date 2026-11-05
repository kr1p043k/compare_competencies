"""Глобальные настройки безопасности, которые админ меняет из панели.

Сейчас это один рубильник: включена ли двухфакторная аутентификация (2FA).

Почему отдельный файл, а не env: переключатель должен применяться мгновенно,
без рестарта API, и жить рядом с остальными настройками админки
(`data/settings/scheduler.json`). Файл читается на каждый вызов, но с кэшем по
mtime: правка из панели подхватывается сразу, а лишних чтений с диска нет.

По умолчанию 2FA ВЫКЛЮЧЕНА: отсутствующий или битый файл не должен внезапно
запирать админов без кода.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

SECURITY_DIRNAME = "settings"
SECURITY_FILENAME = "security.json"

DEFAULTS: dict[str, bool] = {
    "two_factor_enabled": False,
}

# Кэш по (path, mtime, size): файл меняется редко, а читается на каждый логин.
_cache_key: tuple[str, int, int] | None = None
_cache_value: dict[str, bool] = dict(DEFAULTS)


def _settings_path() -> Path:
    return Path(config.DATA_DIR) / SECURITY_DIRNAME / SECURITY_FILENAME


def load_security_settings() -> dict[str, bool]:
    """Прочитать настройки с диска. Битый файл -> дефолты (fail-open для входа)."""
    global _cache_key, _cache_value

    path = _settings_path()
    try:
        stat = path.stat()
    except OSError:
        _cache_key = None
        _cache_value = dict(DEFAULTS)
        return dict(_cache_value)

    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cache_key == key:
        return dict(_cache_value)

    merged = dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Не валим логин из-за опечатки в файле: работаем по дефолтам и кричим в лог.
        logger.error("security_settings_read_failed error=%s path=%s", str(exc), str(path))
        data = {}

    if isinstance(data, dict):
        for key_name in DEFAULTS:
            value = data.get(key_name)
            if isinstance(value, bool):
                merged[key_name] = value

    _cache_key = key
    _cache_value = merged
    return dict(merged)


def two_factor_enabled() -> bool:
    """Включена ли 2FA глобально. Читается на каждом логине."""
    return load_security_settings()["two_factor_enabled"]


def save_security_settings(patch: dict) -> dict[str, bool]:
    """Слить частичный патч, записать атомарно, вернуть полный набор."""
    current = load_security_settings()
    for key_name in DEFAULTS:
        if key_name in patch and isinstance(patch[key_name], bool):
            current[key_name] = patch[key_name]

    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Атомарная замена: админ может получить 500 на середине записи, но не
    # полузаписанный JSON, из которого потом не читается флаг.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".security-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(current, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    global _cache_key, _cache_value
    _cache_key = None
    _cache_value = dict(current)
    return dict(current)


def reset_cache_for_tests() -> None:
    """Сбросить модульный кэш (нужно тестам с monkeypatch на DATA_DIR)."""
    global _cache_key, _cache_value
    _cache_key = None
    _cache_value = dict(DEFAULTS)
