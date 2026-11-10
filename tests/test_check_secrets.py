"""Тесты сканера секретов: ловим реальные ключи, не трогаем обычный текст.

Образцы собираются конкатенацией: в исходнике нет цельных секретных строк,
поэтому pre-commit-хук репозитория не принимает этот тест за утечку.
"""
from __future__ import annotations

from scripts.check_secrets import scan_text

AWS = "AKIA" + "IOSFODNN7EXAMPLE"
GITHUB = "ghp_" + "a" * 36
SLACK = "xoxb-" + "1234567890-abcdefghij"
GOOGLE = "AIza" + "b" * 35
STRIPE = "sk_live_" + "c" * 24
TELEGRAM = "123456789:" + "d" * 35


def test_detects_private_key_header() -> None:
    text = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIE...\n"
    assert scan_text(text) == [(1, "private-key")]


def test_detects_aws_github_slack_google_stripe() -> None:
    text = "\n".join(
        [
            f"aws = {AWS}",
            f"gh = {GITHUB}",
            f"sl = {SLACK}",
            f"go = {GOOGLE}",
            f"st = {STRIPE}",
        ]
    )
    rules = {rule for _ln, rule in scan_text(text)}
    assert rules == {
        "aws-access-key",
        "github-token",
        "slack-token",
        "google-api-key",
        "stripe-key",
    }


def test_detects_telegram_bot_token() -> None:
    assert scan_text(f"token={TELEGRAM}") == [(1, "telegram-bot-token")]


def test_plain_text_is_clean() -> None:
    text = (
        "SECRET_KEY = os.getenv('SECRET_KEY')\n"
        "password = input()\n"
        "# example key: not-a-real-key\n"
    )
    assert scan_text(text) == []


def test_allow_marker_skips_line() -> None:
    line = f"aws = {AWS}  # secret-scan: allow (документация)"
    assert scan_text(line) == []


def test_short_and_low_entropy_values_are_ignored() -> None:
    text = "api_key = 'short'\nkey=AKIA1234\n"
    assert scan_text(text) == []
