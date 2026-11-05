"""Тесты маскирования email в логах.

Аудит хранит полный адрес (нужен для атрибуции), но структурированные логи
уходят в файлы и сборщики — там персональные данные не нужны.
"""

from __future__ import annotations

import pytest

from src.api_pkg.request_logger import mask_email


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ivanov@sfedu.ru", "i***@sfedu.ru"),
        ("a@b.c", "a***@b.c"),
        ("  petrov@example.org  ", "p***@example.org"),
    ],
)
def test_masks_local_part_keeps_domain(raw, expected):
    assert mask_email(raw) == expected


def test_empty_and_none_become_anonymous():
    assert mask_email(None) == "anonymous"
    assert mask_email("") == "anonymous"


def test_non_email_is_fully_masked():
    assert mask_email("не-email") == "***"


def test_local_part_is_not_recoverable():
    masked = mask_email("verylongusername@sfedu.ru")
    assert "verylongusername" not in masked
    assert masked.endswith("@sfedu.ru")
