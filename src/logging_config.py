"""Конфигурация structlog с защитой от дублирования handler'ов."""

import logging
import os

import structlog

from src.config import LOG_FILE


def setup_structlog(console_level: int = None):
    """
    Настраивает structlog: JSON в файл, цветной вывод в консоль.

    Args:
        console_level: Уровень логирования для консоли (по умолчанию из LOG_LEVEL или INFO)
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    if console_level is None:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        console_level = getattr(logging, level_str, logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

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
