# tests/pipeline/test_pipeline.py
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

sys.modules['shap'] = MagicMock()
sys.modules['cv2'] = MagicMock()

from src.pipeline.data_source import DataSource
from src.pipeline.helpers import get_load_mode, load_vacancies_details, save_detailed_vacancies
from src.pipeline.level_builder import LevelBuilder
from src.pipeline.metric_computer import MetricComputer
from src.pipeline.recommendation_runner import RecommendationRunner
from src.pipeline.skill_extractor import SkillExtractor
from src.pipeline.weight_cleaner import WeightCleaner
from src.pipeline.gap_runner import GapRunner

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
# DataSource
# ---------------------------------------------------------------------------
class TestDataSource:
    def test_load_from_cache_success(self, tmp_path, monkeypatch, mock_args):
        mock_args.skip_collection = True
        ds = DataSource(mock_args)

        # Создаём кэш-файл
        data = [{"id": "1", "name": "test"}]
        cache_file = tmp_path / "hh_vacancies_detailed.json"
        cache_file.write_text(json.dumps(data))
        # Переопределяем config
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)

        vacancies, parser = ds.get_vacancies()
        assert vacancies == data
        assert parser is not None

    def test_load_from_cache_file_not_found(self, mock_args, tmp_path, monkeypatch):
        mock_args.skip_collection = True
        ds = DataSource(mock_args)
        # Замокаем _find_file, чтобы он не вызывал sys.exit
        fake_file = tmp_path / "nonexistent.json"
        ds._find_file = MagicMock(return_value=fake_file)
        # safe_read_json вернёт None
        with patch("src.pipeline.data_source.safe_read_json", return_value=None), \
            patch("src.pipeline.data_source.sys.exit") as mock_exit:
            ds._load_from_cache()
            mock_exit.assert_called_once_with(1)

    def test_collect_from_hh_interactive(self, mock_args):
        mock_args.interactive = True
        ds = DataSource(mock_args)

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

            vacancies, parser = ds.get_vacancies()
            assert len(vacancies) == 1
            mock_save.assert_called_once()

    def test_collect_from_hh_it_sector(self, mock_args):
        mock_args.it_sector = True
        ds = DataSource(mock_args)

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

            vacancies, _ = ds.get_vacancies()
            # Проверяем, что industry стал 7
            call_args = mock_collect.call_args[1]
            assert call_args["industry"] == 7

    def test_collect_from_hh_skip_details(self, mock_args):
        mock_args.skip_details = True
        ds = DataSource(mock_args)

        with patch("src.pipeline.data_source.HeadHunterAPI") as mock_api_class, \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
             patch("src.pipeline.data_source.console_info"), \
             patch("src.pipeline.data_source.console_header"):

            mock_api_class.return_value.search_vacancies.return_value = [{"id": "1"}]
            mock_collect.return_value = [{"id": "1"}]

            vacancies, _ = ds.get_vacancies()
            assert len(vacancies) == 1

    def test_find_file_detailed_exists(self, tmp_path, monkeypatch, mock_args):
        detailed = tmp_path / "hh_vacancies_detailed.json"
        detailed.write_text("[]")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = DataSource(mock_args)
        f = ds._find_file()
        assert f == detailed

    def test_find_file_only_basic_exists(self, tmp_path, monkeypatch, mock_args):
        basic = tmp_path / "hh_vacancies_basic.json"
        basic.write_text("[]")
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = DataSource(mock_args)
        f = ds._find_file()
        assert f == basic

    def test_find_file_none_exists(self, tmp_path, monkeypatch, mock_args):
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("src.pipeline.data_source.config.DATA_RAW_DIR", tmp_path)
        ds = DataSource(mock_args)
        with patch("src.pipeline.data_source.sys.exit") as mock_exit:
            ds._find_file()
            mock_exit.assert_called_once_with(1)
    def test_collect_from_hh_single_query(self, mock_args):
        mock_args.use_multiple = False  # будет заходить в else
        ds = DataSource(mock_args)
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
            vacancies, _ = ds.get_vacancies()
            assert len(vacancies) == 1
            mock_api_instance.search_vacancies.assert_called_once()

    def test_collect_from_hh_queries_file(self, mock_args):
        mock_args.queries_file = "queries.txt"
        ds = DataSource(mock_args)
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
            ds.get_vacancies()
            # проверяем, что queries переданы
            args, kwargs = mock_collect.call_args
            assert "QA" in kwargs["queries"]

    def test_collect_from_hh_single_query_skip_details(self, mock_args):
        """Покрытие одиночного поиска с --skip-details (строки 146-147)."""
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = True
        ds = DataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
             patch("src.pipeline.data_source.VacancyParser"), \
             patch("src.pipeline.data_source.console_info"):
            api = MockAPI.return_value
            api.search_vacancies.return_value = [{"id": "1"}]
            vacancies, _ = ds.get_vacancies()
            assert len(vacancies) == 1
            # не должно быть вызова load_vacancies_details

    def test_collect_from_hh_single_query_async(self, mock_args):
        """Покрытие одиночного запроса с асинхронной загрузкой (строки 153-154)."""
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = False
        ds = DataSource(mock_args)
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
            vacancies, _ = ds.get_vacancies()
            assert len(vacancies) == 1
            mock_load.assert_called_once()

    def test_collect_from_hh_no_vacancies_sys_exit(self, mock_args):
        mock_args.it_sector = True  # включает use_multiple
        ds = DataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI"), \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.collect_vacancies_multiple") as mock_collect, \
            patch("src.pipeline.data_source.console_info"), \
            patch("src.pipeline.data_source.sys.exit") as mock_exit:
            mock_collect.return_value = []
            ds._collect_from_hh()
            mock_exit.assert_called_once_with(1)

    def test_collect_from_hh_single_query_async_real(self, mock_args, monkeypatch):
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        mock_args.skip_details = False
        ds = DataSource(mock_args)
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
                # Возвращаем пустой список, но функция не упадет
                vacancies, _ = ds.get_vacancies()
                # Проверяем, что load_vacancies_details была вызвана (можно по вызову async метода)
                async_instance.get_vacancies_details_sync_validated.assert_called()

    def test_collect_from_hh_single_query_no_vacancies_exit(self, mock_args):
        mock_args.it_sector = False
        mock_args.interactive = False
        mock_args.queries_file = None
        mock_args.regions = None
        ds = DataSource(mock_args)
        with patch("src.pipeline.data_source.HeadHunterAPI") as MockAPI, \
            patch("src.pipeline.data_source.VacancyParser"), \
            patch("src.pipeline.data_source.console_info"), \
            patch("src.pipeline.data_source.sys.exit") as mock_exit:
            api = MockAPI.return_value
            api.search_vacancies.return_value = []
            ds._collect_from_hh()
            mock_exit.assert_called_once_with(1)

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
        evals, recs = runner.run()
        assert evals == {}
        assert recs == {}


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
        levels, skills = builder.build(vacancies, MagicMock())
        assert len(levels) == 2
        assert levels[0]["experience"] == "junior"   # теперь совпадает
        assert levels[1]["experience"] == "middle"

    def test_build_with_vacancy_objects(self):
        from src.models.vacancy import Vacancy
        vac1 = MagicMock(spec=Vacancy)
        vac1.key_skills = [MagicMock(name="python")]
        vac1.key_skills[0].name = "python"
        vac1.experience = MagicMock()
        vac1.experience.id = "between3And6"
        vac1.name = "Backend Dev"
        vac1.description = "desc"

        vac2 = MagicMock(spec=Vacancy)
        vac2.key_skills = [MagicMock(name="java")]
        vac2.key_skills[0].name = "java"
        vac2.experience = "senior"
        vac2.name = "Senior Dev"
        vac2.description = "desc"

        builder = LevelBuilder()
        levels, _ = builder.build([vac1, vac2], MagicMock())
        assert levels[0]["experience"] == "middle"
        assert levels[1]["experience"] == "senior"

    def test_build_with_junior_name(self):
        from src.models.vacancy import Vacancy
        vac = MagicMock(spec=Vacancy)
        vac.key_skills = [MagicMock()]
        vac.key_skills[0].name = "python"
        vac.experience = MagicMock()
        vac.experience.id = "between1And3"   # middle по опыту
        vac.name = "Junior Python Developer" # но имя переопределяет на junior
        vac.description = ""
        builder = LevelBuilder()
        levels, _ = builder.build([vac], MagicMock())
        assert levels[0]["experience"] == "junior"

    def test_build_with_senior_name(self):
        from src.models.vacancy import Vacancy
        vac = MagicMock(spec=Vacancy)
        vac.key_skills = [MagicMock()]
        vac.key_skills[0].name = "python"
        vac.experience = "middle"
        vac.name = "Senior Developer"
        vac.description = ""
        builder = LevelBuilder()
        levels, _ = builder.build([vac], MagicMock())
        assert levels[0]["experience"] == "senior"

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
        """Покрытие ветки extracted_skills (строки 21-22)."""
        from src.models.vacancy import Vacancy
        vac = MagicMock(spec=Vacancy)
        vac.key_skills = []               # нет key_skills
        vac.extracted_skills = ["python", "sql"]
        vac.experience = MagicMock()
        vac.experience.id = "between1And3"
        vac.name = "Developer"
        vac.description = ""
        builder = LevelBuilder()
        levels, _ = builder.build([vac], MagicMock())
        assert len(levels) == 1
        assert levels[0]["skills"] == ["python", "sql"]

    def test_build_with_junior_name_in_dict(self):
        """Покрытие переопределения уровня по имени (строки 66-67)."""
        builder = LevelBuilder()
        vac = {
            "key_skills": [{"name": "python"}],
            "experience": {"id": "between1And3"},  # middle по опыту
            "description": "",
            "name": "Junior Python Developer"
        }
        levels, _ = builder.build([vac], MagicMock())
        assert levels[0]["experience"] == "junior"

    def test_build_with_senior_name_in_dict(self):
        """Покрытие переопределения уровня по имени 'senior'."""
        builder = LevelBuilder()
        vac = {
            "key_skills": [{"name": "python"}],
            "experience": {"id": "between1And3"},  # middle
            "description": "",
            "name": "Senior Python Developer"
        }
        levels, _ = builder.build([vac], MagicMock())
        assert levels[0]["experience"] == "senior"


# ---------------------------------------------------------------------------
# MetricComputer
# ---------------------------------------------------------------------------
class TestMetricComputer:
    def test_prepare_and_compute(self):
        comp = MetricComputer({}, [], [], {})
        assert comp.evaluator is None
        comp.prepare({})
        assert comp.evaluator is not None

        # Мокаем evaluator.evaluate_profile
        comp.evaluator.evaluate_profile = MagicMock(return_value={"readiness": 80})
        profiles = {"stud1": MagicMock()}
        evals = comp.compute(profiles)
        assert "stud1" in evals
        assert evals["stud1"] == {"readiness": 80}

    def test_compute_without_prepare_raises(self):
        comp = MetricComputer({}, [], [], {})
        with pytest.raises(RuntimeError, match="Сначала вызовите prepare()"):
            comp.compute({})


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
            mock_engine_instance.generate_recommendations.return_value = {
                "summary": {}, "recommendations": []
            }

            runner.initialize_engine(mock_evaluator)
            assert runner.engine is not None

            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                  "domain_coverage_score": 60, "domain_coverage": {}}}
            recs = runner.run(evaluations)
            assert "p1" in recs

    def test_run_without_initialize_raises(self, mock_args):
        runner = RecommendationRunner({}, {}, mock_args)
        with pytest.raises(RuntimeError, match="Сначала вызовите initialize_engine()"):
            runner.run({})

    def test_run_with_exception(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.side_effect = Exception("fail")
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                "domain_coverage_score": 60, "domain_coverage": {}}}
            recs = runner.run(evaluations)
            assert recs == {}  # пустой результат при ошибке

    def test_build_with_string_experience_junior(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [{"name": "python"}], "experience": "нет опыта", "description": ""}]
        levels, _ = builder.build(vacancies, MagicMock())
        assert levels[0]["experience"] == "junior"

    def test_build_with_string_experience_senior(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [{"name": "python"}], "experience": "senior", "description": ""}]
        levels, _ = builder.build(vacancies, MagicMock())
        assert levels[0]["experience"] == "senior"

    def test_build_with_empty_skills(self):
        builder = LevelBuilder()
        vacancies = [{"key_skills": [], "experience": {"id": "noExperience"}, "description": ""}]
        levels, _ = builder.build(vacancies, MagicMock())
        assert levels == []   # не добавляется, потому что vac_skills пусты

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
        """Покрытие обработки исключения в run (строка 69)."""
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.side_effect = Exception("fail")
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                  "domain_coverage_score": 60, "domain_coverage": {},
                                  "skill_metrics": {}, "cluster_context": {}}}
            recs = runner.run(evaluations)
            assert recs == {}

# ---------------------------------------------------------------------------
# SkillExtractor
# ---------------------------------------------------------------------------
class TestSkillExtractor:
    def test_extract_from_parser(self, tmp_path, monkeypatch, mock_args):
        cache_path = tmp_path / "cache.pkl"
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
            freq, hw, trend = extractor.extract([], mock_parser)
            assert freq == {"python": 10}
            assert cache_path.exists()  # кэш сохранен

    def test_extract_with_cache(self, tmp_path, monkeypatch, mock_args):
        cache_path = tmp_path / "cache.pkl"
        cache_path.touch()                     # <-- файл должен существовать
        raw_file = tmp_path / "raw.json"
        raw_file.write_text("dummy")

        data = {"source_hash": "abc", "result": {"frequencies": {"java": 5}, "hybrid_weights": {"java": 0.7}}}
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        monkeypatch.setattr("src.pipeline.skill_extractor.config.DATA_PROCESSED_DIR", tmp_path)

        extractor = SkillExtractor(mock_args)
        mock_parser = MagicMock()

        with patch.object(extractor, "_get_file_hash", return_value="abc"), \
             patch.object(extractor, "_check_manifest"), \
             patch("src.pipeline.skill_extractor.safe_load_pickle", return_value=data), \
             patch("src.pipeline.skill_extractor.TrendAnalyzer"), \
             patch("src.pipeline.skill_extractor.print_top_skills"), \
             patch("src.pipeline.skill_extractor.load_competency_mapping", return_value=None), \
             patch("src.pipeline.skill_extractor.ArtifactManifest"):
            freq, hw, trend = extractor.extract([], mock_parser, raw_file=raw_file)
            assert freq == {"java": 5}

    def test_extract_with_competency_mapping(self, tmp_path, monkeypatch, mock_args):
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", tmp_path / "cache.pkl")
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
            freq, _, _ = extractor.extract([], mock_parser)
            assert freq == {"python": 10}
            mock_map.assert_called_once()

    def test_run_sets_cluster_context(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {"py": 0.9}, "vacancies_skills": [], "trend_analyzer": None}
        runner = RecommendationRunner(mock_profiles, mock_data, mock_args)
        mock_evaluator = MagicMock()
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
            runner.initialize_engine(mock_evaluator)
            evaluations = {"p1": {"market_coverage_score": 80, "skill_coverage": 70,
                                "domain_coverage_score": 60, "domain_coverage": {},
                                "skill_metrics": {"docker": {"cluster_relevance": 0.8}},
                                "cluster_context": {"skills": {"docker": 0.9}}}}
            runner.run(evaluations)
            mock_engine_instance.set_cluster_context.assert_called_once_with({"docker": 0.9})

    def test_check_manifest_deletes_incompatible(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "cache.pkl"
        cache_path.touch()
        manifest_path = cache_path.with_suffix(".manifest.json")
        manifest_path.write_text("{}")
        monkeypatch.setattr("src.pipeline.skill_extractor.config.PARSED_SKILLS_CACHE_PATH", cache_path)
        extractor = SkillExtractor(MagicMock())
        with patch("src.pipeline.skill_extractor.ArtifactManifest") as MockManifest:
            mock_inst = MockManifest.load.return_value
            mock_inst.is_compatible.return_value = False
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
            result = cleaner.clean({"py": 0.8})
            assert result == {"py": 0.9}

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
        cache_path = tmp_path / "cache.pkl"
        raw_file = tmp_path / "raw.json"
        raw_file.write_text("new content")
        # старый кэш с другим хешем
        old_data = {"source_hash": "old", "result": {"frequencies": {"old": 1}, "hybrid_weights": {}}}
        with open(cache_path, "wb") as f:
            import pickle
            pickle.dump(old_data, f)

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
            patch("src.pipeline.skill_extractor.safe_load_pickle", return_value=old_data):
            # _check_manifest будет вызван, но кэш не подойдёт по хешу
            freq, hw, _ = extractor.extract([], mock_parser, raw_file=raw_file)
            assert freq == {"python": 10}

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
            mock_pe_instance.evaluate_profile.return_value = {
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            }
            mock_re_instance = MockRE.return_value
            mock_re_instance.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
            mock_re_instance.ltr_engine = MagicMock(is_fitted=True)

            evals, recs = runner.run()
            assert "p1" in evals
            assert "p1" in recs

    def test_collect_from_hh_regions(self, mock_args):
        mock_args.regions = "1,2"
        ds = DataSource(mock_args)
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
            ds.get_vacancies()
            # Проверяем, что area_ids установлены
            args, kwargs = mock_collect.call_args
            assert kwargs["area_ids"] == [1, 2]

    def test_run_with_missing_data(self, mock_args):
        mock_profiles = {"p1": MagicMock()}
        mock_data = {"hybrid_weights": {}, "skill_freq": {}, "vacancies_skills": [], "level_vacancies_data": []}
        runner = GapRunner(mock_profiles, mock_data, mock_args)
        evals, recs = runner.run()
        assert evals == {}
        assert recs == {}

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
            mock_pe_instance.evaluate_profile.return_value = {
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            }
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = None              # LTR отсутствует
            mock_re_instance.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
            evals, recs = runner.run()
            assert "p1" in recs

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
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer"), \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = {
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            }
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = None   # LTR отсутствует
            mock_re_instance.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
            runner.run()
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
             patch("src.pipeline.gap_runner.SkillLevelAnalyzer"), \
             patch("src.pipeline.gap_runner.CompetencyComparator"):
            mock_pe_instance = MockPE.return_value
            mock_pe_instance.evaluate_profile.return_value = {
                "market_coverage_score": 70, "skill_coverage": 65, "domain_coverage_score": 60,
                "market_skill_coverage": 50, "readiness_score": 68, "top_recommendations": [],
                "domain_coverage": {}, "skill_metrics": {}
            }
            mock_re_instance = MockRE.return_value
            mock_re_instance.ltr_engine = MagicMock(is_fitted=True)
            mock_re_instance.generate_recommendations.return_value = {"summary": {}, "recommendations": []}
            runner.run()
            captured = capsys.readouterr()
            assert "СВОДКА МЕТРИК ПО ПРОФИЛЯМ" in captured.out
            assert "Готовность к уровню" in captured.out
