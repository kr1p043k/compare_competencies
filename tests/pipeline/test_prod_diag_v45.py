"""v45 prod-diagnosis tests: error surfacing, cache fallback, profiles-first order."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from src import Ok
from src.api_pkg.routers.pipeline import _format_task_error
from src.errors import PipelineError
from src.pipeline.data_source import HhDataSource
from src.result import Err


def _args():
    a = MagicMock()
    a.skip_collection = True
    return a


class TestFormatTaskError:
    def test_detail_appended(self):
        err = PipelineError(message="Pipeline failed at stage data_collection",
                            stage="data_collection", detail="boom-root-cause")
        msg = _format_task_error("GAP-test", err)
        assert "Pipeline failed at stage data_collection" in msg
        assert "boom-root-cause" in msg

    def test_no_detail_no_dash(self):
        err = PipelineError(message="Pipeline failed at stage X", stage="X")
        msg = _format_task_error("Prefix", err)
        assert msg == "Prefix: Pipeline failed at stage X"

    def test_prefix_colon_stripped_once(self):
        err = PipelineError(message="m", stage="s")
        assert _format_task_error("Prefix: ", err) == "Prefix: m"

    def test_plain_exception_ok(self):
        assert _format_task_error("P", ValueError("x")) == "P: x"


class TestLoadFromCacheV45:
    def test_fallback_to_basic_when_detailed_unreadable(self, tmp_path, monkeypatch):
        ds = HhDataSource(_args())
        bad = tmp_path / "hh_vacancies_detailed.json"
        bad.write_text("{not json", encoding="utf-8")
        good = tmp_path / "hh_vacancies_basic.json"
        good.write_text(json.dumps([{"id": "1", "name": "t"}]), encoding="utf-8")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        mock_vac = MagicMock()
        with patch("src.models.vacancy.Vacancy.from_api", return_value=mock_vac):
            match ds.get_vacancies():
                case Ok((vacancies, _parser)):
                    assert len(vacancies) == 1
                case Err(e):
                    pytest.fail(f"fallback failed: {e.message}")

    def test_bad_records_skipped_not_fatal(self, tmp_path, monkeypatch):
        ds = HhDataSource(_args())
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text(json.dumps([{"id": "1"}, {"id": "bad"}, {"id": "2"}]), encoding="utf-8")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)

        def _from_api(v):
            if v.get("id") == "bad":
                raise ValueError("bad record")
            return MagicMock()

        with patch("src.models.vacancy.Vacancy.from_api", side_effect=_from_api):
            match ds.get_vacancies():
                case Ok((vacancies, _parser)):
                    assert len(vacancies) == 2
                case Err(e):
                    pytest.fail(f"should skip bad record: {e.message}")

    def test_all_bad_records_is_err(self, tmp_path, monkeypatch):
        ds = HhDataSource(_args())
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text(json.dumps([{"id": "bad"}]), encoding="utf-8")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        with patch("src.models.vacancy.Vacancy.from_api", side_effect=ValueError("nope")):
            assert ds.get_vacancies().is_err()

    def test_missing_files_error_names_paths(self, tmp_path, monkeypatch):
        ds = HhDataSource(_args())
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        res = ds._load_from_cache()
        assert res.is_err()
        assert "hh_vacancies_basic.json" in res.err().message


class TestProfilesFirstOrder:
    def test_profiles_block_precedes_heavy_ml(self):
        src = Path("src/api_pkg/startup.py").read_text(encoding="utf-8")
        assert src.index("deps.student_profiles[pname]") < src.index("def _run_skill_extraction")


class TestVacancyFileStatus:
    def test_missing_files_reported(self, tmp_path, monkeypatch):
        from src.pipeline.helpers import vacancy_file_status
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_RAW_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_DIR", tmp_path)
        st = vacancy_file_status()
        assert st["raw_exists"] is False
        assert st["detailed_exists"] is False
        assert st["raw_size"] is None
        assert st["students_files"] == []
        assert "cwd" in st

    def test_present_files_reported(self, tmp_path, monkeypatch):
        from src.pipeline.helpers import vacancy_file_status
        raw = tmp_path / "hh_vacancies_basic.json"
        raw.write_text("[1,2,3]", encoding="utf-8")
        sdir = tmp_path / "students"
        sdir.mkdir()
        (sdir / "base_competency.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_RAW_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_DIR", tmp_path)
        st = vacancy_file_status()
        assert st["raw_exists"] is True
        assert st["raw_size"] == 7
        assert st["students_files"] == ["base_competency.json"]

    def test_never_raises(self, tmp_path, monkeypatch):
        import pytest as _pt
        from src.pipeline.helpers import vacancy_file_status
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_PROCESSED_DIR", tmp_path / "nope")
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_RAW_DIR", tmp_path / "nope")
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_DIR", tmp_path / "nope")
        st = vacancy_file_status()  # must not raise
        assert st["raw_exists"] is False
