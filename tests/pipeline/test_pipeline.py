# tests/pipeline/test_pipeline.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules['shap'] = MagicMock()
sys.modules['cv2'] = MagicMock()

from src.pipeline.data_source import HhDataSource
from src.pipeline.helpers import get_load_mode, load_vacancies_details, save_detailed_vacancies
from src.pipeline.level_builder import LevelBuilder
from src.pipeline.metric_computer import MetricComputer
from src.result import Err, Ok
from src.pipeline.recommendation_runner import RecommendationRunner
from src.pipeline.skill_extractor import SkillExtractor
from src.pipeline.weight_cleaner import WeightCleaner
from src.errors import RecommendationError
from src.models.vacancy import Vacancy
from src.pipeline.gap_runner import GapRunner
from src.predictors.models import RecommendationResult, RecommendationSummary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_args():
    """Возвращает объект args с основными полями."""
    args = MagicMock()
    args.skip_collection = False
    args.interactive = False
    args.queries_file = None
    args.regions = None
    args.industry = None
    args.it_sector = False
    args.query = "python"
    args.area_id = 1
    args.period = 30
    args.max_pages = 5
    args.skip_details = False
    args.show_vacancies = False
    args.excel = False
    args.no_filter = False
    args.max_vacancies_per_query = 10
    args.use_async = False
    args.async_threshold = 200
    args.async_workers = 4
    args.use_llm = False
    return args


# ---------------------------------------------------------------------------
# HhDataSource
# ---------------------------------------------------------------------------
class TestHhDataSource:
    def test_load_from_cache_success(self, tmp_path, monkeypatch, mock_args):
        mock_args.skip_collection = True
        ds = HhDataSource(mock_args)

        data = [{"id": "1", "name": "test"}]
        cache_file = tmp_path / "hh_vacancies_detailed.json"
        cache_file.write_text(json.dumps(data))
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)

        mock_vac = MagicMock()
        with patch("src.models.vacancy.Vacancy.from_api", return_value=mock_vac):
            match ds.get_vacancies():
                case Ok((vacancies, parser)):
                    assert len(vacancies) == 1
                    assert vacancies[0] is mock_vac
                    assert parser is not None
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")

    def test_load_from_cache_file_not_found(self, mock_args, tmp_path, monkeypatch):
        mock_args.skip_collection = True
        ds = HhDataSource(mock_args)
        fake_file = tmp_path / "nonexistent.json"
        ds._find_file = MagicMock(return_value=fake_file)
        with patch("src.pipeline.data_source.safe_read_json", return_value=None), \
            patch("src.pipeline.data_source.console_info"):
            result = ds._load_from_cache()
            assert result.is_err()

    def test_collect_from_hh_interactive(self, mock_args):
        mock_args.interactive = True
        ds = HhDataSource(mock_args)

        # Мокаем все внешние зависимости
        with patch("src.pipeline.data_source.HeadHunterAPI") as mock_api_class, \
             patch("src.pipeline.data_source.VacancyParser") as mock_parser_class, \
             patch("src.pipeline.data_source.interactive_config") as mock_interact, \
             patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
             patch("src.pipeline.data_source.get_load_mode") as mock_load_mode, \
             patch("src.pipeline.data_source.load_vacancies_details") as mock_load_det, \
             patch("src.pipeline.data_source.save_detailed_vacancies") as mock_save, \
             patch("src.pipeline.data_source.console_info"), \
             patch("src.pipeline.data_source.console_header"):

            mock_interact.return_value = {
                "query": "Python", "queries": ["Python"], "area_ids": [1],
                "industry": None, "period": 30, "max_pages": 5,
                "skip_details": False, "show_vacancies": False,
                "excel": False, "no_filter": False, "max_vacancies_per_query": 10,
            }
            mock_collect.return_value = [{"id": "1"}]
            mock_load_mode.return_value = (False, 0, "sync_mode")
            mock_load_det.return_value = [{"id": "1", "details": True}]

            match ds.get_vacancies():
                case Ok((vacancies, _)):
                    assert len(vacancies) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")
            mock_save.assert_called_once()

    def test_collect_from_hh_it_sector(self, mock_args):
        mock_args.it_sector = True
        ds = HhDataSource(mock_args)

        with patch("src.pipeline.data_source.HeadHunterAPI"), \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
             patch("src.pipeline.data_source.get_load_mode") as mock_load_mode, \
             patch("src.pipeline.data_source.load_vacancies_details") as mock_load_det, \
             patch("src.pipeline.data_source.save_detailed_vacancies"), \
             patch("src.pipeline.data_source.console_info"), \
             patch("src.pipeline.data_source.console_header"):

            mock_collect.return_value = [{"id": "1"}]
            mock_load_mode.return_value = (False, 0, "sync_mode")
            mock_load_det.return_value = [{"id": "1", "details": True}]

            match ds.get_vacancies():
                case Ok((vacancies, _)):
                    assert len(vacancies) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")
            call_args = mock_collect.call_args[1]
            assert call_args["industry"] == 7

    def test_collect_from_hh_skip_details(self, mock_args):
        mock_args.skip_details = True
        ds = HhDataSource(mock_args)

        with patch("src.pipeline.data_source.HeadHunterAPI") as mock_api_class, \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
             patch("src.pipeline.data_source.console_info"), \
             patch("src.pipeline.data_source.console_header"):

            mock_api_class.return_value.search_vacancies.return_value = [{"id": "1"}]
            mock_collect.return_value = [{"id": "1"}]

            match ds.get_vacancies():
                case Ok((v, _)):
                    assert len(v) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")

    def test_find_file_detailed_exists(self, tmp_path, monkeypatch, mock_args):
        detailed = tmp_path / "hh_vacancies_detailed.json"
        detailed.write_text("[]")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = HhDataSource(mock_args)
        f = ds._find_file()
        assert f == detailed

    def test_find_file_only_basic_exists(self, tmp_path, monkeypatch, mock_args):
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text("[]")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = HhDataSource(mock_args)
        f = ds._find_file()
        assert f == basic

    def test_find_file_none_exists(self, tmp_path, monkeypatch, mock_args):
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = HhDataSource(mock_args)
        f = ds._find_file()
        assert f is None
    def test_collect_from_hh_single_query(self, mock_args):
        mock_args.use_multiple = False  # будет заходить в else
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.get_load_mode") as mock_mode, \
            patch("src.pipeline.data_source.load_vacancies_details") as mock_load, \
            patch("src.pipeline.data_source.save_detailed_vacancies"), \
            patch("src.pipeline.data_source.console_info"):
            mock_api_instance = MockAPI.return_value
            mock_api_instance.search_vacancies.return_value = [{"id": "1"}]
            mock_mode.return_value = (True, 4, "async")
            mock_load.return_value = [{"id": "1", "details": True}]
            match ds.get_vacancies():
                case Ok((v, _)):
                    assert len(v) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")
            mock_api_instance.search_vacancies.assert_called_once()

    def test_collect_from_hh_queries_file(self, mock_args):
        mock_args.queries_file = "queries.txt"
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI"), \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.validate_safe_path", return_value=Path("queries.txt")), \
            patch("src.pipeline.data_source.load_queries_from_file", return_value=["QA"]), \
            patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
            patch("src.pipeline.data_source.get_load_mode", return_value=(False,0,"sync")), \
            patch("src.pipeline.data_source.load_vacancies_details", return_value=[{"id":"1"}]), \
            patch("src.pipeline.data_source.save_detailed_vacancies"), \
            patch("src.pipeline.data_source.console_info"):
            mock_collect.return_value = [{"id":"1"}]
            result = ds.get_vacancies()
            assert result.is_ok()
            args, kwargs = mock_collect.call_args
            assert "QA" in kwargs["queries"]

    def test_collect_from_hh_single_query_skip_details(self, mock_args):
        """Покрытие одиночного поиска с --skip-details (строки 146-147)."""
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = True
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.console_info"):
            api = MockAPI.return_value
            api.search_vacancies.return_value = [{"id": "1"}]
            match ds.get_vacancies():
                case Ok((v, _)):
                    assert len(v) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")

    def test_collect_from_hh_single_query_async(self, mock_args):
        """Покрытие одиночного запроса с асинхронной загрузкой (строки 153-154)."""
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = False
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.get_load_mode") as mock_mode, \
             patch("src.pipeline.data_source.load_vacancies_details") as mock_load, \
             patch("src.pipeline.data_source.save_detailed_vacancies"), \
             patch("src.pipeline.data_source.console_info"):
            api = MockAPI.return_value
            api.search_vacancies.return_value = [{"id": "1"}]
            mock_mode.return_value = (True, 4, "async")
            mock_load.return_value = [{"id": "1", "details": True}]
            match ds.get_vacancies():
                case Ok((v, _)):
                    assert len(v) == 1
                case Err(e):
                    pytest.fail(f"get_vacancies failed: {e.message}")
            mock_load.assert_called_once()

    def test_collect_from_hh_no_vacancies_returns_err(self, mock_args):
        mock_args.it_sector = True
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI"), \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
            patch("src.pipeline.data_source.console_info"):
            mock_collect.return_value = []
            result = ds._collect_from_hh()
            assert result.is_err()

    def test_collect_from_hh_single_query_async_real(self, mock_args, monkeypatch):
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = False
        ds = HhDataSource(mock_args)
        # Мокаем hh_api.search_vacancies
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.get_load_mode") as mock_mode, \
            patch("src.pipeline.data_source.save_detailed_vacancies"), \
            patch("src.pipeline.data_source.console_info"):
            api = MockAPI.return_value
            api.search_vacancies.return_value = [{"id": "1"}]
            mock_mode.return_value = (True, 4, "async")
            # Не патчим load_vacancies_details, но мокаем HeadHunterAPIAsync внутри
            with patch("src.parsing.api.hh_api_async.HeadHunterAPIAsync") as MockAsync:
                async_instance = MockAsync.return_value
                async_instance.get_vacancies_details_sync_validated.return_value = []
                match ds.get_vacancies():
                    case Ok((vacancies, _)):
                        assert len(vacancies) >= 0
                    case Err(e):
                        pytest.fail(f"get_vacancies failed: {e.message}")
                async_instance.get_vacancies_details_sync_validated.assert_called()

    def test_collect_from_hh_single_query_no_vacancies_returns_err(self, mock_args):
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.console_info"):
            api = MockAPI.return_value
            api.search_vacancies.return_value = []
            result = ds._collect_from_hh()
            assert result.is_err()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_get_load_mode_disabled(self, mock_args):
        mock_args.use_async = False
        res = get_load_mode(10, mock_args, MagicMock())
        assert res == (False, 0, "async_disabled_by_user")

    def test_get_load_mode_over_threshold(self, mock_args):
        mock_args.use_async = True
        mock_args.async_threshold = 100
        res = get_load_mode(150, mock_args, MagicMock())
        assert res == (False, 0, "sync_mode_large_volume")

    def test_get_load_mode_async(self, mock_args):
        mock_args.use_async = True
        mock_args.async_threshold = 500
        mock_args.async_workers = 8
        res = get_load_mode(100, mock_args, MagicMock())
        assert res == (True, 8, "async_mode")

    def test_save_detailed_vacancies(self, tmp_path, monkeypatch):
        vac = MagicMock()
        vac.raw_data = {"id": "1"}
        monkeypatch.setattr("src.pipeline.helpers.config.DATA_PROCESSED_DIR", tmp_path)
        # Заставляем isinstance(v, Vacancy) вернуть True, чтобы использовался raw_data
        with patch("src.pipeline.helpers.isinstance", return_value=True):
            save_detailed_vacancies([vac], MagicMock())
        saved_file = tmp_path / "hh_vacancies_detailed.json"
        assert saved_file.exists()
        with open(saved_file, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["id"] == "1"

    def test_run_no_data(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {
            "hybrid_weights": {},
            "skill_freq": {},
            "vacancies_skills": [],
            "level_vacancies_data": [],
            "trend_analyzer": None
        }
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        result = runner.run()
        assert result.is_err()  # GapRunner returns Err when required data is missing


# ---------------------------------------------------------------------------
# LevelBuilder
# ---------------------------------------------------------------------------
class TestLevelBuilder:
    def test_build_with_dicts(self):
        builder = LevelBuilder()
        vacancies = [
            {"key_skills": [{"name": "python"}], "experience": {"id": "no_experience"}, "description": ""},
            {"key_skills": [{"name": "java"}], "experience": {"id": "between1And3"}, "description": ""},
        ]
        match builder.build(vacancies, MagicMock()):
            case Ok((levels, skills)):
                assert len(levels) == 2
                assert levels[0]["experience"] == "junior"
                assert levels[1]["experience"] == "middle"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_vacancy_objects(self):
        builder = LevelBuilder()
        vac1 = MagicMock(spec=Vacancy)
        vac1.id = "1"
        vac1.name = "Python Developer"
        vac1.key_skills = [MagicMock(name="python")]
        vac1.experience = MagicMock(id="no_experience")
        vac1.description = ""
        vac1.snippet = None
        vac2 = MagicMock()
        vac2.id = "2"
        vac2.name = "Java Developer"
        vac2.key_skills = [MagicMock(name="java")]
        vac2.experience = MagicMock(id="between1And3")
        vac2.description = ""
        vac2.snippet = None

        match builder.build([vac1, vac2], MagicMock()):
            case Ok((levels, _)):
                assert len(levels) == 2
                assert levels[0]["experience"] == "junior"
                assert levels[1]["experience"] == "middle"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_junior_name(self):
        builder = LevelBuilder()
        vac = MagicMock(spec=Vacancy)
        vac.name = "Junior Python Developer"
        vac.key_skills = [MagicMock(name="python")]
        vac.experience = MagicMock(id="between1And3")
        vac.description = ""
        vac.snippet = None

        match builder.build([vac], MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "junior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_senior_name(self):
        builder = LevelBuilder()
        vac = MagicMock(spec=Vacancy)
        vac.name = "Senior Python Developer"
        vac.key_skills = [MagicMock(name="python")]
        vac.experience = MagicMock(id="between1And3")
        vac.description = ""
        vac.snippet = None

        match builder.build([vac], MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "senior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_load_vacancies_sync(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.pipeline.helpers.config.PYDANTIC_VALIDATION_ENABLED", False)
        monkeypatch.setattr("src.pipeline.helpers.config.REQUEST_DELAY", 0)
        basic = [{"id": "1"}, {"id": "2"}]
        hh_api = MagicMock()
        hh_api.get_vacancy_details_as_object.return_value = MagicMock()
        # tqdm должен пропускать итератор, а не подменять список
        with patch("src.pipeline.helpers.tqdm", lambda iterable, *args, **kwargs: iterable):
            result = load_vacancies_details(basic, hh_api, use_async=False, async_workers=0,
                                            parser=MagicMock(), log=MagicMock())
            assert len(result) == 2

    def test_load_vacancies_async_validated(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.pipeline.helpers.config.PYDANTIC_VALIDATION_ENABLED", True)
        basic = [{"id": "1"}]
        hh_api = MagicMock()
        from src.models.hh_responses import VacancyDetailResponse
        fake_response = MagicMock(spec=VacancyDetailResponse)
        fake_response.model_dump.return_value = {"id": "1", "name": "test"}
        with patch("src.parsing.api.hh_api_async.HeadHunterAPIAsync") as MockAsync:
            async_instance = MockAsync.return_value
            async_instance.get_vacancies_details_sync_validated.return_value = [fake_response]
            with patch("src.pipeline.helpers.Vacancy.from_api", return_value=MagicMock()):
                result = load_vacancies_details(basic, hh_api, use_async=True, async_workers=2,
                                                parser=MagicMock(), log=MagicMock())
                assert len(result) == 1

    def test_build_with_vacancy_objects_extracted_skills(self):
        builder = LevelBuilder()
        vac = MagicMock(spec=Vacancy)
        vac.id = "1"
        vac.name = "Python Developer"
        vac.key_skills = []
        vac.extracted_skills = ["python", "django"]
        vac.experience = MagicMock(id="no_experience")
        vac.description = ""
        vac.snippet = None

        match builder.build([vac], MagicMock()):
            case Ok((levels, _)):
                assert len(levels) == 1
                assert levels[0]["skills"] == ["python", "django"]
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_junior_name_in_dict(self):
        builder = LevelBuilder()
        vac = {"key_skills": [{"name": "python"}], "name": "младший разработчик", "experience": {"id": "between1And3"}, "description": ""}
        match builder.build([vac], MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "junior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_senior_name_in_dict(self):
        builder = LevelBuilder()
        vac = {"key_skills": [{"name": "python"}], "name": "старший разработчик", "experience": {"id": "between1And3"}, "description": ""}
        match builder.build([vac], MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "senior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")


# ---------------------------------------------------------------------------
# MetricComputer
# ---------------------------------------------------------------------------
class TestMetricComputer:
    def test_prepare_and_compute(self):
        comp = MetricComputer({}, [], [], {})
        assert comp.evaluator is None
        match comp.prepare({}):
            case Ok(None):
                pass
            case Err(e):
                pytest.fail(f"prepare failed: {e}")
        assert comp.evaluator is not None

        comp.evaluator.evaluate_profile = MagicMock(return_value=Ok({"readiness": 80}))
        profiles = {"stud1": MagicMock()}
        match comp.compute(profiles):
            case Ok(evals):
                assert "stud1" in evals
                assert evals["stud1"] == {"readiness": 80}
            case Err(e):
                pytest.fail(f"compute failed: {e}")

    def test_compute_without_prepare_returns_err(self):
        comp = MetricComputer({}, [], [], {})
        result = comp.compute({})
        assert result.is_err()


# ---------------------------------------------------------------------------
# RecommendationRunner
# ---------------------------------------------------------------------------
class TestRecommendationRunner:
    def test_initialize_and_run(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)

        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine, \
             patch("src.pipeline.recommendation_runner.CompetencyComparator") as MockComp:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.return_value = Ok(
                RecommendationResult(summary=RecommendationSummary(), recommendations=[])
            )

            runner.initialize_engine(mock_evaluator)
            assert runner.engine is not None

            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                  "domain_coverage_score": 60, "domain_coverage": {}}}
            match runner.run(evaluations):
                case Ok(recs):
                    assert "p1" in recs
                case Err(e):
                    pytest.fail(f"run failed: {e.message}")

    def test_run_without_initialize_returns_err(self, mock_args):
        runner = RecommendationRunner({}, {}, mock_args)
        result = runner.run({})
        assert result.is_err()

    def test_run_with_exception(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.return_value = Err(
                RecommendationError(message="generation failed", profile="p1")
            )
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                "domain_coverage_score": 60, "domain_coverage": {}}}
            match runner.run(evaluations):
                case Ok(recs):
                    assert recs == {}
                case Err(e):
                    pytest.fail(f"run failed: {e.message}")

    def test_build_with_string_experience_junior(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [{"name": "python"}], "experience": "нет опыта", "description": ""}]
        match builder.build(vacancies, MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "junior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_string_experience_senior(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [{"name": "python"}], "experience": "senior", "description": ""}]
        match builder.build(vacancies, MagicMock()):
            case Ok((levels, _)):
                assert levels[0]["experience"] == "senior"
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_build_with_empty_skills(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [], "experience": {"id": "noExperience"}, "description": ""}]
        match builder.build(vacancies, MagicMock()):
            case Ok((levels, _)):
                assert levels == []
            case Err(e):
                pytest.fail(f"build failed: {e.message}")

    def test_initialize_engine_no_data(self, mock_args):
        """Покрытие initialize_engine с пустыми данными (строка 51)."""
        mock_profiles = {}
        mock_data = {"hybrid_weights": {}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine, \
             patch("src.pipeline.recommendation_runner.CompetencyComparator"):
            runner.initialize_engine(mock_evaluator)
            assert runner.engine is not None

    def test_run_recommendation_generation_failed(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.return_value = Err(
                RecommendationError(message="generation failed", profile="p1")
            )
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                  "domain_coverage_score": 60, "domain_coverage": {},
                                  "skill_metrics": {}, "cluster_context": {}}}
            match runner.run(evaluations):
                case Ok(recs):
                    assert recs == {}
                case Err(e):
                    pytest.fail(f"run failed: {e.message}")

# ---------------------------------------------------------------------------
# SkillExtractor
# ---------------------------------------------------------------------------
class TestSkillExtractor:
    def test_extract_from_parser(self, tmp_path, monkeypatch, mock_args):
        cache_path = tmp_path / "cache.joblib"
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        monkeypatch.setattr("src.pipeline.skill_extractor.config.DATA_PROCESSED_DIR", tmp_path)

        extractor = SkillExtractor(mock_args)
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 10},
            "hybrid_weights": {"python": 0.8}
        }

        with patch("src.pipeline.skill_extractor.load_it_skills", return_value=set()), \
            patch("src.pipeline.skill_extractor.TrendAnalyzer"), \
            patch("src.pipeline.skill_extractor.print_top_skills"), \
            patch("src.pipeline.skill_extractor.load_competency_mapping", return_value=None), \
            patch("src.pipeline.skill_extractor.ArtifactManifest"):
            match extractor.extract([], mock_parser):
                case Ok((freq, hw, trend)):
                    assert freq == {"python": 10}
                case Err(e):
                    pytest.fail(f"extract failed: {e.message}")
            assert cache_path.exists()

    def test_extract_with_cache(self, tmp_path, monkeypatch, mock_args):
        cache_path = tmp_path / "cache.joblib"
        cache_path.touch()
        raw_file = tmp_path / "raw.json"
        raw_file.write_text("dummy")

        data = {"source_hash": "abc", "result": {"frequencies": {"java": 5}, "hybrid_weights": {"java": 0.7}}}
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        monkeypatch.setattr("src.pipeline.skill_extractor.config.DATA_PROCESSED_DIR", tmp_path)

        extractor = SkillExtractor(mock_args)
        mock_parser = MagicMock()

        with patch.object(extractor, "_get_file_hash", return_value="abc"), \
             patch.object(extractor, "_check_manifest"), \
             patch("src.cache_manager.CacheManager.load", return_value=Ok(data)), \
             patch("src.pipeline.skill_extractor.TrendAnalyzer"), \
             patch("src.pipeline.skill_extractor.print_top_skills"), \
             patch("src.pipeline.skill_extractor.load_competency_mapping", return_value=None), \
             patch("src.pipeline.skill_extractor.ArtifactManifest"):
            match extractor.extract([], mock_parser, raw_file=raw_file):
                case Ok((freq, hw, trend)):
                    assert freq == {"java": 5}
                case Err(e):
                    pytest.fail(f"extract failed: {e.message}")

    def test_extract_with_competency_mapping(self, tmp_path, monkeypatch, mock_args):
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", tmp_path / "cache.joblib")
        monkeypatch.setattr("src.pipeline.skill_extractor.config.DATA_PROCESSED_DIR", tmp_path)
        extractor = SkillExtractor(mock_args)
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 10}, "hybrid_weights": {"python": 0.8}
        }
        mapping = {"comp1": ["python"]}
        with patch("src.pipeline.skill_extractor.load_it_skills", return_value=set()), \
            patch("src.pipeline.skill_extractor.TrendAnalyzer"), \
            patch("src.pipeline.skill_extractor.print_top_skills"), \
            patch("src.pipeline.skill_extractor.load_competency_mapping", return_value=mapping), \
            patch("src.pipeline.skill_extractor.map_to_competencies") as mock_map, \
            patch("src.pipeline.skill_extractor.ArtifactManifest"), \
            patch("src.analyzers.skills.skill_filter.SkillFilter") as MockFilter, \
            patch("builtins.open", MagicMock()):
            mock_map.return_value = MagicMock()
            mock_map.return_value.most_common.return_value = [("python", 10)]
            MockFilter.return_value.GENERIC_WORDS = set()
            match extractor.extract([], mock_parser):
                case Ok((freq, _, _)):
                    assert freq == {"python": 10}
                case Err(e):
                    pytest.fail(f"extract failed: {e.message}")
            mock_map.assert_called_once()

    def test_run_sets_cluster_context(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.return_value = Ok(
                RecommendationResult(summary=RecommendationSummary(), recommendations=[])
            )
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                "domain_coverage_score": 60, "domain_coverage": {},
                                "skill_metrics": {"docker": {"cluster_relevance": 0.8}},
                                "cluster_context": {"skills": {"docker": 0.9}}}}
            result = runner.run(evaluations)
            assert result.is_ok()
            mock_engine_instance.set_cluster_context.assert_called_once_with({"docker": 0.9})

    def test_check_manifest_deletes_incompatible(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "cache.joblib"
        cache_path.touch()
        manifest_path = cache_path.with_suffix(".manifest.json")
        manifest_path.write_text("{}")
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        extractor = SkillExtractor(MagicMock())
        with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
            mock_inst = MagicMock()
            mock_inst.is_compatible.return_value = False
            MockManifest.load.return_value = Ok(mock_inst)
            extractor._check_manifest(cache_path)
        assert not cache_path.exists()
        assert not manifest_path.exists()

# ---------------------------------------------------------------------------
# WeightCleaner
# ---------------------------------------------------------------------------
class TestWeightCleaner:
    def test_clean(self, tmp_path, monkeypatch):
        cleaner = WeightCleaner()
        comp_file = tmp_path / "competency_frequency.json"
        comp_file.write_text('{"py": 10}')
        monkeypatch.setattr("src.pipeline.weight_cleaner.config.DATA_PROCESSED_DIR", tmp_path)

        mock_filter = MagicMock()
        mock_filter.get_clean_weights.return_value = {"py": 0.9}
        with patch("src.pipeline.weight_cleaner.SkillFilter", return_value=mock_filter):
            match cleaner.clean({"py": 0.8}):
                case Ok(v):
                    assert v == {"py": 0.9}
                case Err(e):
                    pytest.fail(f"clean failed: {e.message}")

    def test_clean_no_comp_file(self, tmp_path, monkeypatch):
        cleaner = WeightCleaner()
        monkeypatch.setattr("src.pipeline.weight_cleaner.config.DATA_PROCESSED_DIR", tmp_path)
        mock_filter = MagicMock()
        mock_filter.get_clean_weights.return_value = {"py": 0.5}
        with patch("src.pipeline.weight_cleaner.SkillFilter", return_value=mock_filter):
            result = cleaner.clean({"py": 0.8})
            # Проверяем, что передан пустой competency_freq
            call_kwargs = mock_filter.get_clean_weights.call_args[1]
            assert call_kwargs["competency_freq"] == {}

    def test_extract_cache_mismatch_recalculates(self, tmp_path, monkeypatch, mock_args):
        cache_path = tmp_path / "cache.joblib"
        raw_file = tmp_path / "raw.json"
        raw_file.write_text("new content")
        old_data = {"source_hash": "old", "result": {"frequencies": {"old": 1}, "hybrid_weights": {}}}

        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        monkeypatch.setattr("src.pipeline.skill_extractor.config.DATA_PROCESSED_DIR", tmp_path)

        extractor = SkillExtractor(mock_args)
        mock_parser = MagicMock()
        mock_parser.extract_skills_from_vacancies.return_value = {
            "frequencies": {"python": 10}, "hybrid_weights": {"python": 0.8}
        }

        with patch("src.pipeline.skill_extractor.TrendAnalyzer"), \
            patch("src.pipeline.skill_extractor.print_top_skills"), \
            patch("src.pipeline.skill_extractor.load_competency_mapping", return_value=None), \
            patch("src.pipeline.skill_extractor.ArtifactManifest"), \
            patch("src.cache_manager.CacheManager.load", return_value=Ok(old_data)):
            match extractor.extract([], mock_parser, raw_file=raw_file):
                case Ok((freq, hw, _)):
                    assert freq == {"python": 10}
                case Err(e):
                    pytest.fail(f"extract failed: {e.message}")

# ---------------------------------------------------------------------------
# GapRunner
# ---------------------------------------------------------------------------

class TestGapRunner:
    def test_run_with_data(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {
            "hybrid_weights": {"py": 0.9},
            "skill_freq": {"py": 10},
            "vacancies_skills": [["py"]],
            "level_vacancies_data": [{"skills": ["py"], "experience": "middle"}],
            "trend_analyzer": None
        }
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        with patch("src.pipeline.gap_runner.ProfileEvaluator") as MockPE, \
             patch("src.pipeline.gap_runner.RecommendationEngine") as MockRE, \
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer") as MockSLA, \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = Ok({
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            })
            MockSLA.return_value.get_weights_for_level.return_value = Ok({"py": 0.9})
            MockRE.return_value.fit.return_value = Ok(MockRE.return_value)
            mock_re_instance = MockRE.return_value
            mock_re_instance.generate_recommendations.return_value = Ok(RecommendationResult(summary=RecommendationSummary(), recommendations=[]))
            mock_re_instance.ltr_engine = MagicMock(is_fitted=True)

            match runner.run():
                case Ok((evals, recs)):
                    assert "p1" in evals
                    assert "p1" in recs
                case Err(e):
                    pytest.fail(f"run failed: {e.message}")

    def test_collect_from_hh_regions(self, mock_args):
        mock_args.regions = "1,2"
        ds = HhDataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI"), \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
            patch("src.pipeline.data_source.get_load_mode") as mock_load_mode, \
            patch("src.pipeline.data_source.load_vacancies_details") as mock_load_det, \
            patch("src.pipeline.data_source.save_detailed_vacancies"), \
            patch("src.pipeline.data_source.console_info"):
            mock_collect.return_value = [{"id": "1"}]
            mock_load_mode.return_value = (False, 0, "sync")
            mock_load_det.return_value = [{"id": "1", "details": True}]
            result = ds.get_vacancies()
            assert result.is_ok()
            args, kwargs = mock_collect.call_args
            assert kwargs["area_ids"] == [1, 2]

    def test_run_with_missing_data(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {}, "skill_freq": {}, "vacancies_skills": [], "level_vacancies_data": []}
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        match runner.run():
            case Ok((evals, recs)):
                assert evals == {}
                assert recs == {}
            case Err(e):
                # GapRunner returns Err when required data is missing
                assert "данных" in str(e) or "data" in str(e).lower()

    def test_run_ltr_not_fitted(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {
            "hybrid_weights": {"py": 0.9},
            "skill_freq": {"py": 10},
            "vacancies_skills": [["py"]],
            "level_vacancies_data": [{"skills": ["py"], "experience": "middle"}],
            "trend_analyzer": None
        }
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        with patch("src.pipeline.gap_runner.ProfileEvaluator") as MockPE, \
             patch("src.pipeline.gap_runner.RecommendationEngine") as MockRE, \
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer") as MockSLA, \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = Ok({
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            })
            MockSLA.return_value.get_weights_for_level.return_value = Ok({"py": 0.9})
            MockRE.return_value.fit.return_value = Ok(MockRE.return_value)
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = None
            mock_re_instance.generate_recommendations.return_value = Ok(RecommendationResult(summary=RecommendationSummary(), recommendations=[]))
            match runner.run():
                case Ok((evals, recs)):
                    assert "p1" in recs
                case Err(e):
                    pytest.fail(f"run failed: {e.message}")

    def test_run_with_ltr_not_fitted_warning(self, mock_args, capsys):
        """Покрытие ветки LTR не обучен (строки 122-125)."""
        mock_profiles = {"p1": MagicMock()}
        mock_data = {
            "hybrid_weights": {"py": 0.9},
            "skill_freq": {"py": 10},
            "vacancies_skills": [["py"]],
            "level_vacancies_data": [{"skills": ["py"], "experience": "middle"}],
            "trend_analyzer": None
        }
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        with patch("src.pipeline.gap_runner.ProfileEvaluator") as MockPE, \
             patch("src.pipeline.gap_runner.RecommendationEngine") as MockRE, \
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer") as MockSLA, \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = Ok({
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            })
            MockSLA.return_value.get_weights_for_level.return_value = Ok({"py": 0.9})
            MockRE.return_value.fit.return_value = Ok(MockRE.return_value)
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = None
            mock_re_instance.generate_recommendations.return_value = Ok(RecommendationResult(summary=RecommendationSummary(), recommendations=[]))
            result = runner.run()
            assert result.is_ok()
            captured = capsys.readouterr()
            assert "LTR-модель не загружена" in captured.out

    def test_run_print_summary(self, mock_args, capsys):
        """Покрытие _print_summary (строки 107, 113-114)."""
        mock_profiles = {"p1": MagicMock()}
        mock_data = {
            "hybrid_weights": {"py": 0.9},
            "skill_freq": {"py": 10},
            "vacancies_skills": [["py"]],
            "level_vacancies_data": [{"skills": ["py"], "experience": "middle"}],
            "trend_analyzer": None
        }
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        with patch("src.pipeline.gap_runner.ProfileEvaluator") as MockPE, \
             patch("src.pipeline.gap_runner.RecommendationEngine") as MockRE, \
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer") as MockSLA, \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = Ok({
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            })
            MockSLA.return_value.get_weights_for_level.return_value = Ok({"py": 0.9})
            MockRE.return_value.fit.return_value = Ok(MockRE.return_value)
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = MagicMock(is_fitted=True)
            mock_re_instance.generate_recommendations.return_value = Ok(RecommendationResult(summary=RecommendationSummary(), recommendations=[]))
            result = runner.run()
            assert result.is_ok()
            captured = capsys.readouterr()
            assert "СВОДКА МЕТРИК ПО ПРОФИЛЯМ" in captured.out
            assert "Готовность к уровню" in captured.out
