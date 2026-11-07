"""Покрытие src/api_pkg/feature_flags.py: файл флагов, env-default, set_flag.

Это НЕ src/feature_flags.py (старый env-модуль). Здесь флаги живут в
data/settings/features.json, а переменные окружения задают лишь значение по
умолчанию. Тесты подменяют путь на tmp, чтобы не трогать общий файл.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.api_pkg.feature_flags as ff


@pytest.fixture()
def flags_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "features.json"
    monkeypatch.setattr(ff, "_FEATURES_PATH", path)
    monkeypatch.delenv("FEATURES_FILE", raising=False)
    return path


# ---------- _features_path ----------


def test_features_path_uses_module_default(tmp_path, monkeypatch):
    target = tmp_path / "f.json"
    monkeypatch.setattr(ff, "_FEATURES_PATH", target)
    monkeypatch.delenv("FEATURES_FILE", raising=False)
    assert ff._features_path() == target


def test_features_path_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom.json"
    monkeypatch.setenv("FEATURES_FILE", str(override))
    assert ff._features_path() == override


# ---------- is_enabled / defaults ----------


def test_flag_defaults_off(flags_file: Path):
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is False


def test_env_default_enables_flag(flags_file: Path, monkeypatch):
    monkeypatch.setattr(ff.config, "TOTP_2FA_ENABLED", True, raising=False)
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is True


def test_file_value_overrides_env_default(flags_file: Path, monkeypatch):
    monkeypatch.setattr(ff.config, "TOTP_2FA_ENABLED", True, raising=False)
    flags_file.write_text(json.dumps({ff.FLAG_TOTP_2FA: False}), encoding="utf-8")
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is False


def test_unknown_flag_uses_default_false(flags_file: Path):
    assert ff.is_enabled("неизвестный_флаг") is False


def test_load_file_tolerates_broken_json(flags_file: Path):
    flags_file.write_text("{ это не json", encoding="utf-8")
    # Битый файл не должен ронять: работаем на значении по умолчанию.
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is False


def test_load_file_ignores_non_dict_json(flags_file: Path):
    flags_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is False


def test_load_file_missing_returns_empty(flags_file: Path):
    assert ff._load_file() == {}


# ---------- set_flag / all_flags ----------


def test_set_flag_persists_and_returns_state(flags_file: Path):
    state = ff.set_flag(ff.FLAG_TOTP_2FA, True)
    assert state[ff.FLAG_TOTP_2FA]["enabled"] is True
    assert state[ff.FLAG_TOTP_2FA]["source"] == "file"
    assert json.loads(flags_file.read_text(encoding="utf-8"))[ff.FLAG_TOTP_2FA] is True


def test_set_flag_toggle_back(flags_file: Path):
    ff.set_flag(ff.FLAG_TOTP_2FA, True)
    ff.set_flag(ff.FLAG_TOTP_2FA, False)
    assert ff.is_enabled(ff.FLAG_TOTP_2FA) is False


def test_set_flag_unknown_raises(flags_file: Path):
    with pytest.raises(KeyError):
        ff.set_flag("нет_такого_флага", True)


def test_all_flags_reports_env_source_when_no_file(flags_file: Path):
    state = ff.all_flags()
    assert ff.FLAG_TOTP_2FA in state
    assert state[ff.FLAG_TOTP_2FA]["source"] == "env"
    assert state[ff.FLAG_TOTP_2FA]["default"] is False


def test_all_flags_coerces_non_bool_to_bool(flags_file: Path):
    # В файле может оказаться не bool: set_flag приводит к bool.
    ff.set_flag(ff.FLAG_TOTP_2FA, 1)  # type: ignore[arg-type]
    assert json.loads(flags_file.read_text(encoding="utf-8"))[ff.FLAG_TOTP_2FA] is True


def test_save_is_atomic_no_tmp_left(flags_file: Path):
    ff.set_flag(ff.FLAG_TOTP_2FA, True)
    leftovers = list(flags_file.parent.glob("*.tmp"))
    assert leftovers == []
