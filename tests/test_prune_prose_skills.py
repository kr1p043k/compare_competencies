"""Тесты для scripts/prune_prose_skills.py.

Проверяем только работу с JSON на диске (без БД): правила детекции мусора,
dry-run без изменения файла, --apply с удалением только мусора и сохранением
остальной структуры, а также пропуск дисциплин, которые опустели бы.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "prune_prose_skills", ROOT / "scripts" / "prune_prose_skills.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _krm(disciplines):
    return {
        "09.03.02": {
            "direction_name": "Информационные системы и технологии",
            "profile": "профиль",
            "disciplines": disciplines,
        }
    }


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_real_skills_kept(mod):
    for name in ["python", "sql", "работа в команде", "C#", "1С"]:
        assert mod.is_junk(name) is False


def test_junk_rejected(mod):
    junk = [
        "",
        "   ",
        "a",
        "123",
        "12.34",
        "Дисциплина преподается в 6 и 7 семестрах.",
        "эта дисциплина изучается в пятом семестре глубоко",
        "!!!",
        "а" * 70,
    ]
    for name in junk:
        assert mod.is_junk(name) is True


def test_dry_run_does_not_modify(mod, tmp_path):
    data = _krm(
        {
            "Математика": {
                "competencies": {"к-1": "компетенция"},
                "skills": {"код-1": ["python", "123", ""]},
                "ksa": {},
            },
        }
    )
    path = tmp_path / "krm.json"
    _write(path, data)
    before = path.read_text(encoding="utf-8")

    removed = mod.process_file(path, apply=False)

    assert path.read_text(encoding="utf-8") == before
    assert removed == 2
    assert not (tmp_path / "krm_backup_prune.json").exists()


def test_apply_removes_only_junk(mod, tmp_path):
    data = _krm(
        {
            "Математика": {
                "competencies": {"к-1": {"name": "компетенция"}},
                "skills": {
                    "код-1": ["python", "sql", "123", "Дисциплина преподается в 6 и 7 семестрах.", ""],
                    "код-2": ["работа в команде"],
                },
                "ksa": {"описание": "не трогать"},
            },
        }
    )
    path = tmp_path / "krm.json"
    _write(path, data)

    removed = mod.process_file(path, apply=True)

    assert removed == 3
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    disc = reloaded["09.03.02"]["disciplines"]["Математика"]
    assert disc["skills"]["код-1"] == ["python", "sql"]
    assert disc["skills"]["код-2"] == ["работа в команде"]
    assert disc["competencies"] == {"к-1": {"name": "компетенция"}}
    assert disc["ksa"] == {"описание": "не трогать"}
    backup = tmp_path / "krm_backup_prune.json"
    assert backup.exists()
    assert json.loads(backup.read_text(encoding="utf-8")) == data


def test_empty_discipline_left_untouched(mod, tmp_path, capsys):
    data = _krm(
        {
            "Пустая": {
                "competencies": {},
                "skills": {"код-1": ["123", ""]},
                "ksa": {},
            },
            "Смешанная": {
                "competencies": {},
                "skills": {"код-1": ["python", "123"]},
                "ksa": {},
            },
        }
    )
    path = tmp_path / "krm.json"
    _write(path, data)

    removed = mod.process_file(path, apply=True)

    assert removed == 1
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["09.03.02"]["disciplines"]["Пустая"]["skills"]["код-1"] == ["123", ""]
    assert reloaded["09.03.02"]["disciplines"]["Смешанная"]["skills"]["код-1"] == ["python"]
    out = capsys.readouterr().out
    assert "0 навыков" in out


def test_main_apply_flag(mod, tmp_path):
    data = _krm(
        {
            "Математика": {
                "competencies": {},
                "skills": {"код-1": ["python", "123"]},
                "ksa": {},
            },
        }
    )
    path = tmp_path / "krm.json"
    _write(path, data)

    code = mod.main(["--file", str(path), "--apply"])

    assert code == 0
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["09.03.02"]["disciplines"]["Математика"]["skills"]["код-1"] == ["python"]
