"""Конфигурация structlog с маскированием чувствительных данных и защитой от дублирования handler'ов."""

import contextlib
import logging
import os
import re
import sys
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


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler, который не может уронить вызывающий код.

    Логирование кириллицы в консоль с cp1251 выбрасывало UnicodeEncodeError.
    Он улетал из logger.* наружу, попадал в собственный `except Exception`
    обработчика и превращал нормальный ответ в 503. Поэтому здесь глотаем любые
    ошибки записи: лог важен, но ответ пользователю важнее лога.
    """

    def emit(self, record):
        try:
            super().emit(record)
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 — защитный слой логирования
            pass

    def handleError(self, record):  # noqa: N802 — имя задано API logging.Handler
        # Никогда не печатаем в stderr: он и есть источник проблемы с кодировкой.
        pass


def _harden_stdio_streams() -> None:
    """Переводит stdout/stderr в utf-8 с backslashreplace, если это возможно."""
    for stream in (_stream for _stream in (sys.stdout, sys.stderr)):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def setup_structlog(console_level: int = None):
    root_logger = logging.getLogger()
    _harden_stdio_streams()

    def _has_file_handler():
        return any(
            isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("backend.log")
            for h in root_logger.handlers
        )

    if _has_file_handler():
        return

    if console_level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        console_level = getattr(logging, level_str, logging.INFO)

    file_handler = RotatingFileHandler(LOG_FILE, encoding="utf-8", maxBytes=10 * 1024 * 1024, backupCount=5, delay=True)
    file_handler.setLevel(logging.DEBUG)
    _orig_rollover = file_handler.doRollover

    def _safe_rollover():
        with contextlib.suppress(PermissionError):
            _orig_rollover()

    file_handler.doRollover = _safe_rollover

    console_handler = SafeStreamHandler()
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
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    if not has_console:
        root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy.cmdstan").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("pystan").setLevel(logging.WARNING)
