# tests/test_logging_config.py
import logging
import os
from unittest.mock import patch, MagicMock

import pytest

from src.logging_config import SecretsMasker, setup_structlog


class TestSecretsMasker:
    def test_mask_api_key(self):
        event = {"message": "api_key=abcdef12345"}
        result = SecretsMasker.mask(None, None, event)
        assert "abcdef" not in result["message"]
        assert "***" in result["message"]

    def test_mask_token(self):
        event = {"headers": "Authorization: token=deadbeef"}
        result = SecretsMasker.mask(None, None, event)
        assert "deadbeef" not in result["headers"]
        assert "***" in result["headers"]

    def test_mask_password(self):
        event = {"body": "password=supersecret"}   # ожидаемый формат
        result = SecretsMasker.mask(None, None, event)
        assert "supersecret" not in result["body"]
        assert "***" in result["body"]

    def test_no_sensitive_data(self):
        event = {"info": "just a normal message"}
        result = SecretsMasker.mask(None, None, event)
        assert result == event


class TestSetupStructlog:
    def test_setup_without_existing_handlers(self, tmp_path, monkeypatch):
        # Убираем все хендлеры root логгера
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        # Подменяем LOG_FILE на временный
        monkeypatch.setattr("src.logging_config.LOG_FILE", tmp_path / "test.log")
        setup_structlog()
        # Проверяем, что хендлеры добавились
        assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)

    def test_setup_with_existing_handlers_does_nothing(self, tmp_path, monkeypatch):
        root = logging.getLogger()
        # Добавляем временный хендлер
        root.addHandler(logging.StreamHandler())
        # Патчим LOG_FILE, чтобы убедиться, что setup не меняет хендлеры
        monkeypatch.setattr("src.logging_config.LOG_FILE", tmp_path / "test.log")
        handler_count_before = len(root.handlers)
        setup_structlog()
        assert len(root.handlers) == handler_count_before

    def test_setup_with_env_log_level(self, tmp_path, monkeypatch):
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        monkeypatch.setattr("src.logging_config.LOG_FILE", tmp_path / "test.log")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        setup_structlog()
        assert any(h.level == logging.DEBUG for h in root.handlers if isinstance(h, logging.StreamHandler))

    def test_setup_with_invalid_log_level(self, tmp_path, monkeypatch):
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        monkeypatch.setattr("src.logging_config.LOG_FILE", tmp_path / "test.log")
        monkeypatch.setenv("LOG_LEVEL", "INVALID")
        # Должен использоваться INFO по умолчанию
        setup_structlog()
        assert any(h.level == logging.INFO for h in root.handlers if isinstance(h, logging.StreamHandler))
