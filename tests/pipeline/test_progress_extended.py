import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.pipeline.progress import write, write_result, MAX_LOGS, PROGRESS_FILE


class TestProgressWrite:
    def test_write_creates_dir(self, tmp_path):
        f = tmp_path / "sub" / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(10, "start")
        assert f.exists()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["pct"] == 10
        assert data["message"] == "start"

    def test_write_read_existing_fails_continues(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        f.parent.mkdir(parents=True)
        f.write_text("invalid json", encoding="utf-8")
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(50, "recovered")
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["pct"] == 50
        assert len(data["logs"]) == 1

    def test_write_existing_file_with_logs(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"pct": 10, "message": "old", "logs": ["[00:00:00] old"]}, ensure_ascii=False),
                     encoding="utf-8")
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(20, "new")
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 2
        assert "new" in data["logs"][-1]

    def test_write_provided_logs_replaces(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(10, "first", logs=["existing"])
            write(20, "second", logs=["replacement"])
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["logs"] == ["replacement"]

    def test_write_duplicate_logs_not_appended(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(10, "msg")
            write(20, "msg")
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 1

    def test_write_logs_truncated(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            with patch("src.pipeline.progress.MAX_LOGS", 2):
                write(10, "a")
                write(20, "b")
                write(30, "c")
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 2
        assert "b" in data["logs"][0]
        assert "c" in data["logs"][1]

    def test_write_exception_does_not_raise(self):
        with patch("src.pipeline.progress.PROGRESS_FILE", Path("/invalid/path/file.json")):
            write(50, "safe_fail")

    def test_write_logs_is_none_without_existing(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write(10, "first")
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 1


class TestProgressWriteResult:
    def test_write_result_ok(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            result = write_result(100, "done")
        assert result.is_ok()

    def test_write_result_creates_dir(self, tmp_path):
        f = tmp_path / "a" / "b" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            result = write_result(50, "test")
        assert result.is_ok()
        assert f.exists()

    def test_write_result_reads_existing(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"pct": 10, "message": "old", "logs": ["[00:00:00] old"]}, ensure_ascii=False),
                     encoding="utf-8")
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            result = write_result(50, "updated")
        assert result.is_ok()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 2

    def test_write_result_read_existing_fails_continues(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        f.parent.mkdir(parents=True)
        f.write_text("not-json", encoding="utf-8")
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            result = write_result(30, "recovered")
        assert result.is_ok()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["pct"] == 30

    def test_write_result_logs_provided(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            result = write_result(80, "almost", logs=["custom"])
        assert result.is_ok()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["logs"] == ["custom"]

    def test_write_result_logs_duplicate_skipped(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            write_result(10, "same")
            result = write_result(20, "same")
        assert result.is_ok()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 1

    def test_write_result_truncates(self, tmp_path):
        f = tmp_path / "cache" / "pipeline_progress.json"
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            with patch("src.pipeline.progress.MAX_LOGS", 2):
                write_result(10, "a")
                write_result(20, "b")
                result = write_result(30, "c")
        assert result.is_ok()
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["logs"]) == 2

    def test_write_result_err_on_failure(self):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = write_result(50, "fail")
        assert result.is_err()

    def test_write_result_err_on_json_dump_fail(self):
        mock_open = MagicMock()
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_open.return_value = mock_file
        mock_file.write.side_effect = OSError("write fail")
        with patch("builtins.open", mock_open):
            result = write_result(50, "fail")
        assert result.is_err()
