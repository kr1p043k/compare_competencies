from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.errors import DataSourceError
from src.infrastructure.hh_provider import HHVacancyProvider


class TestHHVacancyProvider:
    def test_init(self):
        p = HHVacancyProvider()
        assert p._api is not None
        assert p._areas_cache is None

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_search_success(self, MockAPI):
        mock_api = MagicMock()
        mock_api.search_vacancies.return_value = [{"id": 1}]
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.search("python", area=1, period=30, pages=5)
        assert result.is_ok()
        assert len(result.ok()) == 1

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_search_no_results(self, MockAPI):
        mock_api = MagicMock()
        mock_api.search_vacancies.return_value = []
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.search("nonexistent", area=1, period=30, pages=5)
        assert result.is_err()

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_search_exception(self, MockAPI):
        mock_api = MagicMock()
        mock_api.search_vacancies.side_effect = Exception("API error")
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.search("python", area=1, period=30, pages=5)
        assert result.is_err()

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_get_details_success(self, MockAPI):
        mock_api = MagicMock()
        mock_api.get_vacancy.return_value = {"id": "123"}
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.get_details("123")
        assert result.is_ok()

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_get_details_not_found(self, MockAPI):
        mock_api = MagicMock()
        mock_api.get_vacancy.return_value = None
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.get_details("missing")
        assert result.is_err()

    @patch("src.infrastructure.hh_provider.HeadHunterAPI")
    def test_get_details_exception(self, MockAPI):
        mock_api = MagicMock()
        mock_api.get_vacancy.side_effect = Exception("HTTP error")
        MockAPI.return_value = mock_api
        p = HHVacancyProvider()
        p._api = mock_api
        result = p.get_details("123")
        assert result.is_err()

    def test_get_areas_cached(self):
        p = HHVacancyProvider()
        p._areas_cache = [{"id": 1}]
        result = p.get_areas()
        assert result.is_ok()
        assert result.ok() == [{"id": 1}]

    @patch("requests.get")
    def test_get_areas_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": "1", "name": "Russia"}]
        mock_get.return_value = mock_resp
        p = HHVacancyProvider()
        result = p.get_areas()
        assert result.is_ok()
        assert result.ok()[0]["name"] == "Russia"

    @patch("requests.get")
    def test_get_areas_exception(self, mock_get):
        mock_get.side_effect = Exception("connection error")
        p = HHVacancyProvider()
        result = p.get_areas()
        assert result.is_err()
