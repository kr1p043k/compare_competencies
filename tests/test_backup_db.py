"""backup_db: восстановление выбирает свежайший дамп, пустой каталог - выход с ошибкой.

pg_dump/pg_restore не запускаем: subprocess.run замокан, проверяем выбор и CLI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import src.cli.backup_db as bk


def _fake_run(calls):
    def _run(args, *a, **k):
        calls.append(args)
        return type("R", (), {"returncode": 0})()
    return _run


def test_restore_latest_selects_newest(tmp_path, monkeypatch):
    (tmp_path / "backup_20260101_000000.sql").write_bytes(b"old")
    (tmp_path / "backup_20260201_000000.sql").write_bytes(b"new")
    monkeypatch.setattr(bk, "BACKUP_DIR", tmp_path)
    calls: list = []
    monkeypatch.setattr(bk.subprocess, "run", _fake_run(calls))

    bk.main(restore="latest")

    assert calls, "pg_restore не вызван"
    assert str(calls[0][-1]).endswith("backup_20260201_000000.sql")


def test_restore_named_path(tmp_path, monkeypatch):
    named = tmp_path / "backup_20260301_000000.sql"
    named.write_bytes(b"x")
    monkeypatch.setattr(bk, "BACKUP_DIR", tmp_path)
    calls: list = []
    monkeypatch.setattr(bk.subprocess, "run", _fake_run(calls))

    bk.main(restore=str(named))

    assert Path(calls[0][-1]) == named


def test_restore_latest_without_backups_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(bk, "BACKUP_DIR", tmp_path)
    monkeypatch.setattr(bk.subprocess, "run", _fake_run([]))
    with pytest.raises(SystemExit):
        bk.main(restore="latest")
