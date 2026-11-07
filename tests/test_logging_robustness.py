"""Логирование не имеет права ронять процесс из-за кодировки потока.

Prod-defect: на cp1251-консоли запись кириллицы/эмодзи в `sys.stderr` (structlog
`ConsoleRenderer` через `logging.StreamHandler`) бросала `UnicodeEncodeError`.
Стандартный `handleError` ловит только `OSError`, поэтому исключение улетало в
`except Exception` обработчика и превращало нормальный ответ в 503.

Тесты симулируют поток, падающий на не-ASCII записях, и доказывают:
  * log-вызов с кириллицей/эмодзи не поднимает исключение;
  * обычная ASCII-строка по-прежнему доходит до консольного обработчика
    (логирование не выключено молча);
  * консольный обработчик - именно SafeStreamHandler.
"""

from __future__ import annotations

import contextlib
import logging
import sys

import structlog

import src.logging_config as lc
from src.logging_config import SafeStreamHandler


class _AsciiOnlyStream:
    """Поток, который бросает UnicodeEncodeError на любой не-ASCII записи."""

    encoding = "ascii"

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, s: str) -> int:
        s.encode("ascii")  # падает на кириллице и эмодзи
        self.written.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _clean_logging(monkeypatch, tmp_path):
    """Снимает root-обработчики, уводит лог-файл в tmp и ставит ASCII-поток."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()

    monkeypatch.setattr(lc, "LOG_FILE", tmp_path / "robustness.log")
    stream = _AsciiOnlyStream()
    monkeypatch.setattr(sys, "stderr", stream, raising=False)
    lc.setup_structlog()
    try:
        yield stream
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()


def test_cyrillic_message_does_not_raise_and_ascii_still_logged(monkeypatch, tmp_path):
    with _clean_logging(monkeypatch, tmp_path) as stream:
        logger = structlog.get_logger("test.logging_robustness")

        # Кириллица не должна поднять исключение наружу.
        logger.info("Привет, мир")

        # ASCII обязан дойти до консольного обработчика.
        logger.info("ascii-probe")

        assert any("ascii-probe" in chunk for chunk in stream.written), stream.written


def test_exception_with_cyrillic_and_emoji_does_not_raise(monkeypatch, tmp_path):
    with _clean_logging(monkeypatch, tmp_path):
        logger = structlog.get_logger("test.logging_robustness")

        try:
            raise ValueError("Ошибка 💥 данные недоступны")
        except ValueError:
            logger.exception("processing_failed")  # не должно поднять исключение


def test_console_handler_is_safe_stream_handler(monkeypatch, tmp_path):
    with _clean_logging(monkeypatch, tmp_path):
        root = logging.getLogger()
        # RotatingFileHandler тоже StreamHandler, поэтому файловые исключаем.
        ours = [
            h for h in root.handlers
            if isinstance(h, SafeStreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert ours, f"нет SafeStreamHandler: {[type(h).__name__ for h in root.handlers]}"


def test_setup_reconfigures_stderr_to_backslashreplace(monkeypatch, tmp_path):
    """Филиал с reconfigure: поток переводится на UTF-8 + backslashreplace."""
    calls: list[dict] = []

    class _RecordingStream(_AsciiOnlyStream):
        def reconfigure(self, **kwargs) -> None:
            calls.append(kwargs)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()

    monkeypatch.setattr(lc, "LOG_FILE", tmp_path / "robustness.log")
    stream = _RecordingStream()
    monkeypatch.setattr(sys, "stderr", stream, raising=False)
    lc.setup_structlog()

    assert any(
        c.get("errors") == "backslashreplace" and c.get("encoding") == "utf-8"
        for c in calls
    ), f"reconfigure не вызван с нужными параметрами: {calls}"
