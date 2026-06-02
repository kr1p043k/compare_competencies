from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from src import Ok, Err


@pytest.fixture
def args():
    mock = MagicMock()
    mock.skip_collection = False
    mock.interactive = False
    mock.queries_file = None
    mock.regions = None
    mock.industry = None
    mock.it_sector = False
    mock.query = "Python Developer"
    mock.area_id = 1
    mock.queries = None
    mock.area_ids = None
    mock.period = 30
    mock.max_pages = 5
    mock.max_vacancies_per_query = 100
    mock.skip_details = False
    mock.show_vacancies = False
    mock.excel = False
    mock.no_filter = False
    mock.async_threshold = 100
    mock.use_async = False
    mock.async_workers = 5
    return mock


class TestHhDataSourceInit:
    def test_init(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        assert ds.args is args


class TestHhDataSourceGetVacancies:
    def test_skip_collection_loads_from_cache(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.skip_collection = True
        with patch.object(ds, "_find_file", return_value=Path("test.json")):
            with patch("src.pipeline.data_source.safe_read_json", return_value=[{"id": "1", "name": "Dev"}]):
                with patch("src.pipeline.data_source.VacancyParser"):
                    with patch("src.models.vacancy.Vacancy.from_api") as mock_from_api:
                        mock_from_api.return_value = MagicMock()
                        result = ds.get_vacancies()
        assert result.is_ok()
        vacancies, parser = result.unwrap()
        assert len(vacancies) == 1

    def test_skip_collection_no_file(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.skip_collection = True
        with patch.object(ds, "_find_file", return_value=None):
            result = ds.get_vacancies()
        assert result.is_err()

    def test_skip_collection_empty_data(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.skip_collection = True
        with patch.object(ds, "_find_file", return_value=Path("test.json")):
            with patch("src.pipeline.data_source.safe_read_json", return_value=None):
                result = ds.get_vacancies()
        assert result.is_err()

    def test_collect_single_query_ok(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        raw_vacs = [{"id": "1", "name": "Dev"}]

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance
            api_instance.search_vacancies.return_value = Ok(raw_vacs)

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.get_load_mode", return_value=(False, 0, "sync")):
                    with patch("src.pipeline.data_source.load_vacancies_details",
                              return_value=Ok([MagicMock()])):
                        with patch("src.pipeline.data_source.save_detailed_vacancies",
                                   return_value=Ok(None)):
                            result = ds.get_vacancies()
        assert result.is_ok()

    def test_collect_single_query_no_vacancies(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance
            api_instance.search_vacancies.return_value = Ok([])

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance
                result = ds.get_vacancies()
        assert result.is_err()

    def test_collect_single_query_api_error(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance
            api_instance.search_vacancies.return_value = Err("api error")

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance
                result = ds.get_vacancies()
        assert result.is_err()

    def test_collect_single_query_skip_details(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.skip_details = True
        raw_vacs = [{"id": "1", "name": "Dev"}]

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance
            api_instance.search_vacancies.return_value = Ok(raw_vacs)

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.get_load_mode", return_value=(False, 0, "sync")):
                    result = ds.get_vacancies()
        assert result.is_ok()

    def test_collect_single_query_details_err(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        raw_vacs = [{"id": "1", "name": "Dev"}]

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance
            api_instance.search_vacancies.return_value = Ok(raw_vacs)

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.get_load_mode", return_value=(False, 0, "sync")):
                    with patch("src.pipeline.data_source.load_vacancies_details",
                              return_value=Err("details fail")):
                        result = ds.get_vacancies()
        assert result.is_err()

    def test_collect_with_it_sector(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.it_sector = True

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.collect_vacancies_multiple",
                          return_value=[{"id": "1"}]):
                    with patch("src.pipeline.data_source.get_load_mode",
                              return_value=(False, 0, "sync")):
                        with patch("src.pipeline.data_source.load_vacancies_details",
                                  return_value=Ok([MagicMock()])):
                            with patch("src.pipeline.data_source.save_detailed_vacancies",
                                      return_value=Ok(None)):
                                result = ds.get_vacancies()
        assert result.is_ok()
        assert isinstance(ds.args.queries, list)
        assert len(ds.args.queries) > 5
        assert ds.args.industry == 7

    def test_collect_with_queries_file(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.queries_file = "queries.txt"

        with patch("src.pipeline.data_source.validate_safe_path", return_value=Path("queries.txt")):
            with patch("src.pipeline.data_source.load_queries_from_file", return_value=["Dev"]):
                with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
                    api_instance = MagicMock()
                    MockAPI.return_value = api_instance

                    with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                        parser_instance = MagicMock()
                        MockParser.return_value = parser_instance

                        with patch("src.pipeline.data_source.collect_vacancies_multiple",
                                  return_value=[{"id": "1"}]):
                            with patch("src.pipeline.data_source.get_load_mode",
                                      return_value=(False, 0, "sync")):
                                with patch("src.pipeline.data_source.load_vacancies_details",
                                          return_value=Ok([MagicMock()])):
                                    with patch("src.pipeline.data_source.save_detailed_vacancies",
                                              return_value=Ok(None)):
                                        result = ds.get_vacancies()
        assert result.is_ok()

    def test_collect_multiple_no_vacancies(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.it_sector = True

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.collect_vacancies_multiple",
                          return_value=[]):
                    result = ds.get_vacancies()
        assert result.is_err()

    def test_collect_interactive(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.interactive = True

        with patch("src.pipeline.data_source.interactive_config") as mock_ic:
            mock_ic.return_value = {
                "query": "Python",
                "queries": ["Python"],
                "area_ids": [1],
                "industry": None,
                "period": 30,
                "max_pages": 5,
                "skip_details": False,
                "show_vacancies": False,
                "excel": False,
                "no_filter": False,
                "max_vacancies_per_query": 100,
            }
            with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
                api_instance = MagicMock()
                MockAPI.return_value = api_instance

                with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                    parser_instance = MagicMock()
                    MockParser.return_value = parser_instance

                    with patch("src.pipeline.data_source.collect_vacancies_multiple",
                              return_value=[{"id": "1"}]):
                        with patch("src.pipeline.data_source.get_load_mode",
                                  return_value=(False, 0, "sync")):
                            with patch("src.pipeline.data_source.load_vacancies_details",
                                      return_value=Ok([MagicMock()])):
                                with patch("src.pipeline.data_source.save_detailed_vacancies",
                                          return_value=Ok(None)):
                                    result = ds.get_vacancies()
        assert result.is_ok()

    def test_collect_with_regions(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        ds.args.regions = "1,2,3"
        ds.args.it_sector = True

        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI:
            api_instance = MagicMock()
            MockAPI.return_value = api_instance

            with patch("src.pipeline.data_source.VacancyParser") as MockParser:
                parser_instance = MagicMock()
                MockParser.return_value = parser_instance

                with patch("src.pipeline.data_source.collect_vacancies_multiple",
                          return_value=[{"id": "1"}]):
                    with patch("src.pipeline.data_source.get_load_mode",
                              return_value=(False, 0, "sync")):
                        with patch("src.pipeline.data_source.load_vacancies_details",
                                  return_value=Ok([MagicMock()])):
                            with patch("src.pipeline.data_source.save_detailed_vacancies",
                                      return_value=Ok(None)):
                                result = ds.get_vacancies()
        assert result.is_ok()
        assert ds.args.area_ids == [1, 2, 3]

    def test_collect_exception_bubbles_up(self, args):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)

        with patch.object(ds, "_collect_from_hh", side_effect=ValueError("unexpected")):
            result = ds.get_vacancies()
        assert result.is_err()


class TestHhDataSourceFindFile:
    def test_find_file_detailed(self, args, tmp_path):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        detailed = tmp_path / "hh_vacancies_detailed.json"
        detailed.write_text("[]", encoding="utf-8")
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text("[]", encoding="utf-8")

        config = MagicMock()
        config.DATA_PROCESSED_DIR = tmp_path
        config.DATA_RAW_DIR = tmp_path

        with patch("src.pipeline.data_source.config", config):
            result = ds._find_file()
        assert result == detailed

    def test_find_file_basic(self, args, tmp_path):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text("[]", encoding="utf-8")

        config = MagicMock()
        config.DATA_PROCESSED_DIR = tmp_path
        config.DATA_RAW_DIR = tmp_path

        with patch("src.pipeline.data_source.config", config):
            result = ds._find_file()
        assert result == basic

    def test_find_file_none(self, args, tmp_path):
        from src.pipeline.data_source import HhDataSource
        ds = HhDataSource(args)

        config = MagicMock()
        config.DATA_PROCESSED_DIR = tmp_path
        config.DATA_RAW_DIR = tmp_path

        with patch("src.pipeline.data_source.config", config):
            result = ds._find_file()
        assert result is None


class TestHhDataSourceConsoleInfo:
    def test_console_info(self, args, capsys):
        from src.pipeline.data_source import HhDataSource
        HhDataSource._console_info("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out
