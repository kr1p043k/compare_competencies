# tests/test_utils.py (дополнительные тесты)
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src import utils


class TestGetLogger:
    def test_get_logger_creates_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.LOG_FILE", tmp_path / "test.log")
        logger = utils.get_logger("test")
        assert len(logger.handlers) == 2

    def test_get_logger_returns_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.LOG_FILE", tmp_path / "test.log")
        logger1 = utils.get_logger("test2")
        logger2 = utils.get_logger("test2")
        assert logger1 is logger2


class TestLoadCompetencyMapping:
    def test_load_success(self, tmp_path, monkeypatch):
        data = {"C1": ["skill1"]}
        path = tmp_path / "mapping.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr("src.utils.COMPETENCY_MAPPING_FILE", path)
        assert utils.load_competency_mapping() == data


class TestAtomicWriteRead:
    def test_atomic_write_and_read(self, tmp_path):
        filepath = tmp_path / "test.json"
        data = {"key": "value"}
        utils.atomic_write_json(data, filepath)
        assert filepath.exists()
        result = utils.atomic_read_json(filepath)
        assert result == data

    def test_atomic_write_exception_cleans_temp(self, tmp_path):
        filepath = tmp_path / "test.json"
        with patch("src.utils.json.dump", side_effect=Exception("write error")):
            with pytest.raises(Exception):
                utils.atomic_write_json({"a": 1}, filepath)
        # Временный файл должен быть удалён
        assert len(list(filepath.parent.glob("*.tmp"))) == 0

    def test_atomic_read_json_broken(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{invalid")
        assert utils.atomic_read_json(path) is None


class TestSafeReadJSON:
    def test_valid(self, tmp_path):
        path = tmp_path / "valid.json"
        path.write_text('[{"id": 1}]')
        assert utils.safe_read_json(path) == [{"id": 1}]

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        assert utils.safe_read_json(path) is None

    def test_not_a_list(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text('{"key": "value"}')
        assert utils.safe_read_json(path) is None


class TestSafeReadCompetencyJSON:
    def test_valid(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"компетенции": ["C1", "C2"]}))
        assert utils.safe_read_competency_json(path) == ["C1", "C2"]

    def test_uses_skills_key(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"навыки": ["S1"]}))
        assert utils.safe_read_competency_json(path) == ["S1"]

    def test_invalid_structure(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"компетенции": "not a list"}))
        assert utils.safe_read_competency_json(path) == []

    def test_file_missing(self, tmp_path):
        assert utils.safe_read_competency_json(tmp_path / "nonexistent.json") == []


class TestValidateSafePath:
    def test_inside(self, tmp_path):
        base = tmp_path
        user = "subdir/file.txt"
        resolved = utils.validate_safe_path(user, base_dir=base)
        assert str(resolved).startswith(str(base.resolve()))

    def test_outside_raises(self, tmp_path):
        base = tmp_path
        user = "../outside.txt"
        with pytest.raises(ValueError, match="выходит за пределы"):
            utils.validate_safe_path(user, base_dir=base)


class TestSafeLoadPickle:
    def test_success(self, tmp_path):
        import pickle
        data = {"key": "value"}
        filepath = tmp_path / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        result = utils.safe_load_pickle(filepath, allowed_dirs=[tmp_path])
        assert result == data

    def test_outside_dir_returns_none(self, tmp_path):
        filepath = tmp_path / "test.pkl"
        filepath.touch()
        result = utils.safe_load_pickle(filepath, allowed_dirs=[tmp_path / "other"])
        assert result is None

    def test_corrupted_file_returns_none(self, tmp_path):
        filepath = tmp_path / "bad.pkl"
        filepath.write_text("not a pickle")
        result = utils.safe_load_pickle(filepath, allowed_dirs=[tmp_path])
        assert result is None
