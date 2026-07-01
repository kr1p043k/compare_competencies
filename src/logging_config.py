"""Конфигурация structlog с маскированием чувствительных данных и защитой от дублирования handler'ов."""

import logging
import os
import re
from logging.handlers import RotatingFileHandler

import structlog

from src.config import LOG_FILE


class SecretsMasker:
    """Заменяет API-ключи, токены и пароли на *** при логировании."""

    _patterns = [
        (re.compile(r"(api[_-]?key[=:]\s*)[^\s,}]+", re.IGNORECASE), r"\1***"),
        (re.compile(r"(token[=:]\s*)[^\s,}]+", re.IGNORECASE), r"\1***"),
        (re.compile(r"(secret[=:]\s*)[^\s,}]+", re.IGNORECASE), r"\1***"),
        (re.compile(r"(password[=:]\s*)[^\s,}]+", re.IGNORECASE), r"\1***"),
        (re.compile(r"(?:Ключ|Пароль|Токен|Секрет)[=:]\s*[^\s,}]+", re.IGNORECASE), "***"),
    ]

    @classmethod
    def mask(cls, _, __, event_dict):
        for pattern, replacement in cls._patterns:
            for key, value in event_dict.items():
                if isinstance(value, str):
                    event_dict[key] = pattern.sub(replacement, value)
        return event_dict


def setup_structlog(console_level: int = None):
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    if console_level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        console_level = getattr(logging, level_str, logging.INFO)

    file_handler = RotatingFileHandler(LOG_FILE, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=5, delay=True)
    file_handler.setLevel(logging.DEBUG)
    _orig_rollover = file_handler.doRollover

    def _safe_rollover():
        try:
            _orig_rollover()
        except PermissionError:
            pass

    file_handler.doRollover = _safe_rollover

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    use_colors = (
        os.getenv("FORCE_COLOR", "").lower() in ("1", "true")
        or hasattr(console_handler.stream, "isatty")
        and console_handler.stream.isatty()
    )

    console_renderer = structlog.dev.ConsoleRenderer() if use_colors else structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            SecretsMasker.mask,  # <-- маскирование секретов
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            console_renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy.cmdstan").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("pystan").setLevel(logging.WARNING)
