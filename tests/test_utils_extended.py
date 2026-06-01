import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src import Ok, Err, DomainError
from src import utils


class TestLoadCompetencyMappingResult:
    def test_success(self, tmp_path, monkeypatch):
        data = {"C1": ["skill1"]}
        path = tmp_path / "mapping.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr("src.utils.COMPETENCY_MAPPING_FILE", path)
        result = utils.load_competency_mapping_result()
        assert result.is_ok()
        assert result.unwrap() == data

    def test_file_not_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.COMPETENCY_MAPPING_FILE", tmp_path / "nonexistent.json")
        result = utils.load_competency_mapping_result()
        assert result.is_err()
        assert "not found" in result.err().message

    def test_json_decode_error(self, tmp_path, monkeypatch):
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        monkeypatch.setattr("src.utils.COMPETENCY_MAPPING_FILE", path)
        result = utils.load_competency_mapping_result()
        assert result.is_err()
        assert "Failed to load" in result.err().message


class TestSafeReadJsonResult:
    def test_success(self, tmp_path):
        path = tmp_path / "valid.json"
        path.write_text('[{"id": 1}]')
        result = utils.safe_read_json_result(path)
        assert result.is_ok()
        assert result.unwrap() == [{"id": 1}]

    def test_file_not_found(self, tmp_path):
        result = utils.safe_read_json_result(tmp_path / "nonexistent.json")
        assert result.is_err()
        assert "not found" in result.err().message

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        result = utils.safe_read_json_result(path)
        assert result.is_err()
        assert "Empty" in result.err().message

    def test_not_a_list(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text('{"key": "value"}')
        result = utils.safe_read_json_result(path)
        assert result.is_err()
        assert "expected list" in result.err().message

    def test_json_decode_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        result = utils.safe_read_json_result(path)
        assert result.is_err()
        assert "read error" in result.err().message

    def test_unicode_decode_error(self, tmp_path):
        path = tmp_path / "bad_enc.json"
        path.write_bytes(b"\xff\xfe\x00\x01")
        result = utils.safe_read_json_result(path)
        assert result.is_err()
        assert "read error" in result.err().message

    def test_unexpected_exception(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text('["ok"]')
        with patch("json.load", side_effect=Exception("something broke")):
            result = utils.safe_read_json_result(path)
            assert result.is_err()
            assert "Unexpected" in result.err().message


class TestSafeReadCompetencyJsonResult:
    def test_success_competencies_key(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"компетенции": ["C1", "C2"]}))
        result = utils.safe_read_competency_json_result(path)
        assert result.is_ok()
        assert result.unwrap() == ["C1", "C2"]

    def test_success_skills_key(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"навыки": ["S1", "S2"]}))
        result = utils.safe_read_competency_json_result(path)
        assert result.is_ok()
        assert result.unwrap() == ["S1", "S2"]

    def test_success_codes_key(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"codes": ["CODE1", "CODE2"]}))
        result = utils.safe_read_competency_json_result(path)
        assert result.is_ok()
        assert result.unwrap() == ["CODE1", "CODE2"]

    def test_file_not_found(self, tmp_path):
        result = utils.safe_read_competency_json_result(tmp_path / "nonexistent.json")
        assert result.is_err()
        assert "not found" in result.err().message

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("")
        result = utils.safe_read_competency_json_result(path)
        assert result.is_err()
        assert "Empty" in result.err().message

    def test_invalid_structure(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"компетенции": "not a list"}))
        result = utils.safe_read_competency_json_result(path)
        assert result.is_err()
        assert "Invalid competency structure" in result.err().message

    def test_json_decode_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid")
        result = utils.safe_read_competency_json_result(path)
        assert result.is_err()
        assert "read error" in result.err().message

    def test_unicode_decode_error(self, tmp_path):
        path = tmp_path / "bad_enc.json"
        path.write_bytes(b"\xff\xfe\x00\x01")
        result = utils.safe_read_competency_json_result(path)
        assert result.is_err()
        assert "read error" in result.err().message

    def test_unexpected_exception(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"компетенции": ["C1"]}))
        with patch("json.load", side_effect=Exception("unexpected")):
            result = utils.safe_read_competency_json_result(path)
            assert result.is_err()
            assert "Unexpected" in result.err().message


class TestValidateSafePathResult:
    def test_inside(self, tmp_path):
        base = tmp_path
        user = "subdir/file.txt"
        result = utils.validate_safe_path_result(user, base_dir=base)
        assert result.is_ok()
        assert str(result.unwrap()).startswith(str(base.resolve()))

    def test_outside_returns_err(self, tmp_path):
        base = tmp_path
        user = "../outside.txt"
        result = utils.validate_safe_path_result(user, base_dir=base)
        assert result.is_err()
        assert "outside" in result.err().message

    def test_uses_default_base_dir(self, monkeypatch):
        fake_base = Path("/fake/base")
        monkeypatch.setattr("src.utils.BASE_DIR", fake_base)
        result = utils.validate_safe_path_result("some/file.txt")
        assert result.is_ok()

    def test_exception_returns_err(self, tmp_path):
        base = tmp_path
        with patch.object(Path, "resolve", side_effect=Exception("boom")):
            result = utils.validate_safe_path_result("whatever", base_dir=base)
            assert result.is_err()
            assert "Path validation failed" in result.err().message


class TestSafeReadCompetencyJsonCodesKey:
    def test_codes_key(self, tmp_path):
        path = tmp_path / "comp.json"
        path.write_text(json.dumps({"codes": ["C1", "C2"]}))
        assert utils.safe_read_competency_json(path) == ["C1", "C2"]


class TestAtomicWriteJsonReplaceFailure:
    def test_replace_failure_cleans_temp(self, tmp_path):
        filepath = tmp_path / "test.json"
        with patch("os.replace", side_effect=Exception("replace failed")):
            with pytest.raises(Exception, match="replace failed"):
                utils.atomic_write_json({"a": 1}, filepath)
        assert len(list(filepath.parent.glob("*.tmp"))) == 0
