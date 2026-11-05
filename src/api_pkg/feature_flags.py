"""Runtime-переключатели функций (feature flags).

Флаг живёт в `data/settings/features.json`, чтобы его можно было изменить из
админки без перезапуска сервиса. Значение из окружения (`config`) задаёт
начальное состояние; файл его переопределяет, когда в нём есть явное значение.

Файл читается на каждый вызов: переключатель должен срабатывать сразу, а не
после рестарта. Чтение дешёвое (единицы килобайт), поэтому кэш не нужен и
риска «залипшего» значения нет.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

from src import config

logger = structlog.get_logger(__name__)

#: Ключ флага двухфакторной аутентификации.
FLAG_TOTP_2FA = "totp_2fa_enabled"

_FEATURES_PATH = Path(config.DATA_DIR) / "settings" / "features.json"
_DEFAULTS: dict[str, bool] = {FLAG_TOTP_2FA: False}


def _env_default(flag: str) -> bool:
    """Значение по умолчанию из настроек процесса."""
    if flag == FLAG_TOTP_2FA:
        return bool(getattr(config, "TOTP_2FA_ENABLED", False))
    return _DEFAULTS.get(flag, False)


def _load_file() -> dict[str, Any]:
    try:
        raw = _FEATURES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # Битый файл не должен ронять приложение: работаем на значениях по
        # умолчанию и оставляем след в логе.
        logger.warning("features_file_unreadable", error=str(exc), path=str(_FEATURES_PATH))
        return {}


def is_enabled(flag: str) -> bool:
    """Включён ли флаг: файл переопределяет значение по умолчанию."""
    stored = _load_file().get(flag)
    if isinstance(stored, bool):
        return stored
    return _env_default(flag)


def _save(values: dict[str, Any]) -> None:
    _FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True)
    # Атомарная запись: temporary в том же каталоге + os.replace.
    fd, tmp = tempfile.mkstemp(dir=str(_FEATURES_PATH.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, _FEATURES_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def set_flag(flag: str, enabled: bool) -> dict[str, Any]:
    """Записать значение флага и вернуть полное состояние."""
    if flag not in _DEFAULTS:
        raise KeyError(flag)
    current = _load_file()
    current[flag] = bool(enabled)
    _save(current)
    logger.info("feature_flag_changed", flag=flag, enabled=bool(enabled))
    return all_flags()


def all_flags() -> dict[str, Any]:
    """Состояние всех известных флагов для админки."""
    return {
        name: {
            "enabled": is_enabled(name),
            "default": _env_default(name),
            "source": "file" if isinstance(_load_file().get(name), bool) else "env",
        }
        for name in _DEFAULTS
    }
