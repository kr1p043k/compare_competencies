from unittest.mock import MagicMock, patch

import pytest

from src import Err, Ok
from src.errors import DataSourceError, DomainError
from src.infrastructure.file_provider import FileDataProvider, StudentFileProvider


class TestFileDataProvider:
    @patch("src.infrastructure.file_provider.config")
    def test_get_vacancies_no_files(self, mock_config):
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value = MagicMock()
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value.exists.return_value = False
        mock_config.DATA_RAW_DIR.joinpath.return_value = MagicMock()
        mock_config.DATA_RAW_DIR.joinpath.return_value.exists.return_value = False
        p = FileDataProvider()
        result = p.get_vacancies(["test"], max_pages=5)
        assert result.is_err()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_json")
    def test_get_vacancies_success(self, mock_read, mock_config):
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value = MagicMock()
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value.exists.return_value = True
        mock_read.return_value = [{"id": 1}]
        p = FileDataProvider()
        result = p.get_vacancies(["dev"], max_pages=5)
        assert result.is_ok()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_json")
    def test_get_vacancies_empty_data(self, mock_read, mock_config):
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value = MagicMock()
        mock_config.DATA_PROCESSED_DIR.joinpath.return_value.exists.return_value = True
        mock_read.return_value = None
        p = FileDataProvider()
        result = p.get_vacancies(["dev"], max_pages=5)
        assert result.is_err()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_competency_json")
    def test_get_student_profiles(self, mock_read, mock_config):
        mock_config.STUDENTS_DIR.joinpath.side_effect = lambda *a: MagicMock(exists=lambda: True)
        mock_read.return_value = ["code1"]
        p = FileDataProvider()
        result = p.get_student_profiles()
        assert result.is_ok()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_competency_json")
    def test_get_student_profiles_empty(self, mock_read, mock_config):
        mock_config.STUDENTS_DIR.joinpath.side_effect = lambda *a: MagicMock(exists=lambda: True)
        mock_read.return_value = None
        p = FileDataProvider()
        result = p.get_student_profiles()
        assert result.is_ok()
        assert result.ok() == {}

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_json")
    def test_get_reference_data_unknown(self, mock_read, mock_config):
        p = FileDataProvider()
        result = p.get_reference_data("nonexistent_ref")
        assert result.is_err()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_json")
    def test_get_reference_data_success(self, mock_read, mock_config):
        mock_config.COMPETENCY_MAPPING_FILE = MagicMock()
        mock_config.COMPETENCY_MAPPING_FILE.exists.return_value = True
        mock_read.return_value = {"key": "val"}
        p = FileDataProvider()
        result = p.get_reference_data("competency_mapping")
        assert result.is_ok()
        assert result.ok()["key"] == "val"


class TestStudentFileProvider:
    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_competency_json")
    def test_load_profile(self, mock_read, mock_config):
        mock_config.STUDENTS_DIR.joinpath.return_value = MagicMock()
        mock_config.STUDENTS_DIR.joinpath.return_value.exists.return_value = True
        mock_read.return_value = ["code"]
        p = StudentFileProvider()
        result = p.load_profile("base")
        assert result.is_ok()

    @patch("src.infrastructure.file_provider.config")
    @patch("src.infrastructure.file_provider.safe_read_competency_json")
    def test_load_profile_not_found(self, mock_read, mock_config):
        mock_config.STUDENTS_DIR.joinpath.return_value = MagicMock()
        mock_config.STUDENTS_DIR.joinpath.return_value.exists.return_value = False
        mock_read.return_value = None
        p = StudentFileProvider()
        result = p.load_profile("missing")
        assert result.is_err()

    @patch("src.infrastructure.file_provider.StudentFileProvider.load_profile")
    def test_load_all(self, mock_load):
        mock_load.side_effect = lambda n: Ok([n]) if n in ("base", "dc") else Err(DomainError("not found"))
        p = StudentFileProvider()
        result = p.load_all()
        assert result.is_ok()
        assert "base" in result.ok()
        assert "top_dc" not in result.ok()

    @patch("src.infrastructure.file_provider.StudentFileProvider.load_profile")
    def test_load_all_cached(self, mock_load):
        p = StudentFileProvider()
        p._cache = {"cached": ["skill"]}
        result = p.load_all()
        assert result.is_ok()
        assert "cached" in result.ok()
        mock_load.assert_not_called()
