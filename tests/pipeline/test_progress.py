import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.progress import write, write_result


def test_writes_pct_and_message(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.PROGRESS_FILE", f):
        write(50, "halfway")
    assert f.exists()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["pct"] == 50
    assert data["message"] == "halfway"
    assert "logs" in data


def test_writes_appends_logs(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.PROGRESS_FILE", f):
        write(10, "start")
        write(20, "next")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert len(data["logs"]) == 2
    assert "next" in data["logs"][-1]


def test_writes_uses_provided_logs(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.PROGRESS_FILE", f):
        write(50, "with logs", logs=["custom"])
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["logs"] == ["custom"]


def test_writes_skips_duplicate_logs(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.PROGRESS_FILE", f):
        write(10, "same")
        write(20, "same")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert len(data["logs"]) == 1


def test_writes_no_error_on_write_failure(tmp_path):
    with patch("src.pipeline.progress.PROGRESS_FILE", Path("/nonexistent/path/to/file.json")):
        write(50, "should not crash")


def test_write_result_returns_ok(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.PROGRESS_FILE", f):
        result = write_result(100, "done")
    assert result.is_ok()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["pct"] == 100


def test_write_result_returns_err_on_failure():
    with patch("src.pipeline.progress.PROGRESS_FILE", Path("/nonexistent/path/to/file.json")):
        result = write_result(50, "fail")
    assert result.is_err()


def test_logs_truncated_to_max(tmp_path):
    f = tmp_path / "cache" / "pipeline_progress.json"
    with patch("src.pipeline.progress.MAX_LOGS", 3):
        with patch("src.pipeline.progress.PROGRESS_FILE", f):
            for i in range(10):
                write(i * 10, f"step {i}")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert len(data["logs"]) == 3
