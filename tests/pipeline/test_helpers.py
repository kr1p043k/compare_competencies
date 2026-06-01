from unittest.mock import MagicMock, patch, ANY

import pytest

from src import Ok, Err


class TestGetLoadMode:
    def test_async_disabled_by_user(self):
        args = MagicMock()
        args.use_async = False
        result = __import__("src.pipeline.helpers", fromlist=["get_load_mode"]).get_load_mode(10, args, MagicMock())
        assert result == (False, 0, "async_disabled_by_user")

    def test_sync_mode_large_volume(self):
        args = MagicMock()
        args.use_async = True
        args.async_threshold = 5
        log = MagicMock()
        result = __import__("src.pipeline.helpers", fromlist=["get_load_mode"]).get_load_mode(10, args, log)
        assert result == (False, 0, "sync_mode_large_volume")
        log.warning.assert_called_once()

    def test_async_mode(self):
        args = MagicMock()
        args.use_async = True
        args.async_threshold = 20
        args.async_workers = 5
        result = __import__("src.pipeline.helpers", fromlist=["get_load_mode"]).get_load_mode(10, args, MagicMock())
        assert result == (True, 5, "async_mode")


class TestSaveDetailedVacancies:
    def test_save_ok(self, tmp_path):
        from src.pipeline.helpers import save_detailed_vacancies
        v = {"id": 1}
        with patch("src.pipeline.helpers.config") as cfg:
            cfg.DATA_PROCESSED_DIR = tmp_path
            result = save_detailed_vacancies([v], MagicMock())
        assert result.is_ok()

    def test_save_exception(self):
        from src.pipeline.helpers import save_detailed_vacancies
        with patch("src.pipeline.helpers.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = MagicMock()
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value.parent.mkdir.side_effect = PermissionError("denied")
            result = save_detailed_vacancies([MagicMock()], MagicMock())
        assert result.is_err()


class TestConsole:
    def test_console_info(self, capsys):
        from src.pipeline.helpers import console_info
        console_info("test message")
        out = capsys.readouterr().out
        assert "test message" in out

    def test_console_header(self, capsys):
        from src.pipeline.helpers import console_header
        console_header("HEADER")
        out = capsys.readouterr().out
        assert "HEADER" in out


class TestLoadVacanciesDetails:
    def test_sync_load_empty(self):
        from src.pipeline.helpers import load_vacancies_details
        hh_api = MagicMock()
        with patch("src.pipeline.helpers.config") as cfg:
            cfg.REQUEST_DELAY = 0.001
            cfg.PYDANTIC_VALIDATION_ENABLED = False
            with patch("src.pipeline.helpers.tqdm"):
                result = load_vacancies_details([], hh_api, False, 0, MagicMock(), MagicMock())
        assert result.is_ok()

    def test_sync_load_with_validated(self):
        from src.pipeline.helpers import load_vacancies_details
        hh_api = MagicMock()
        mock_vac = MagicMock()
        mock_vac.get.return_value = "123"
        hh_api.get_vacancy_details_validated.return_value = Ok(MagicMock())
        with patch("src.pipeline.helpers.config") as cfg:
            cfg.REQUEST_DELAY = 0.001
            cfg.PYDANTIC_VALIDATION_ENABLED = True
            with patch("src.pipeline.helpers.tqdm"):
                with patch("src.pipeline.helpers.Vacancy.from_api"):
                    result = load_vacancies_details([mock_vac], hh_api, False, 0, MagicMock(), MagicMock())
        assert result.is_ok()

    def test_sync_load_validated_fails(self):
        from src.pipeline.helpers import load_vacancies_details
        hh_api = MagicMock()
        mock_vac = MagicMock()
        mock_vac.get.return_value = "123"
        hh_api.get_vacancy_details_validated.return_value = Err("validation error")
        with patch("src.pipeline.helpers.config") as cfg:
            cfg.REQUEST_DELAY = 0.001
            cfg.PYDANTIC_VALIDATION_ENABLED = True
            with patch("src.pipeline.helpers.tqdm"):
                result = load_vacancies_details([mock_vac], hh_api, False, 0, MagicMock(), MagicMock())
        assert result.is_ok()
