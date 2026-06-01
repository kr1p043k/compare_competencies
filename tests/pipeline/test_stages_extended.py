"""Extended pipeline stage tests for uncovered paths (69% → 85%+)."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src import Ok, Err
from src.models.vacancy import Vacancy
from src.pipeline.stages import (
    DataCollectionStage,
    QualityScoringStage,
    SkillExtractionStage,
    ModelTrainingStage,
)


@pytest.fixture
def args():
    return MagicMock()


@pytest.fixture
def mock_vacancy():
    v = MagicMock()
    v.raw_data = {}
    return v


class TestDataCollectionSkipCollection:
    def test_skip_collection_calls_find_file(self, args):
        s = DataCollectionStage(args)
        args.skip_collection = True
        with patch("src.pipeline.stages.HhDataSource") as MockSource:
            mock_source = MockSource.return_value
            mock_source.get_vacancies.return_value = Ok(([], None))
            mock_source._find_file.return_value = "cached.json"
            result = s.run()
        assert result.is_ok()
        data = result.unwrap()
        assert data["raw_file"] == "cached.json"
        mock_source._find_file.assert_called_once()

    def test_skip_collection_false_returns_none_raw_file(self, args):
        s = DataCollectionStage(args)
        args.skip_collection = False
        with patch("src.pipeline.stages.HhDataSource") as MockSource:
            mock_source = MockSource.return_value
            mock_source.get_vacancies.return_value = Ok(([], None))
            result = s.run()
        assert result.is_ok()
        data = result.unwrap()
        assert data["raw_file"] is None


class TestQualityScoringExcel:
    def test_excel_generation(self, args, mock_vacancy):
        s = QualityScoringStage(args)
        s.args.excel = True
        mock_score = MagicMock()
        mock_score.is_spam = False
        mock_parser = MagicMock()
        mock_parser.aggregate_to_dataframe.return_value = MagicMock()

        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value.score.return_value = Ok(mock_score)
            MockScorer.return_value._build_report.return_value = {}
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                with patch("builtins.open"):
                    with patch("src.pipeline.stages.config") as cfg:
                        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
                        result = s.run(vacancies=[mock_vacancy], parser=mock_parser)

        assert result.is_ok()
        mock_parser.aggregate_to_dataframe.assert_called_once()
        mock_parser.save_to_excel.assert_called_once()

    def test_excel_no_vacancies_skipped(self, args):
        s = QualityScoringStage(args)
        s.args.excel = True
        mock_parser = MagicMock()

        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value._build_report.return_value = {}
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                with patch("builtins.open"):
                    with patch("src.pipeline.stages.config") as cfg:
                        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
                        result = s.run(vacancies=[], parser=mock_parser)

        assert result.is_ok()
        mock_parser.aggregate_to_dataframe.assert_not_called()
        mock_parser.save_to_excel.assert_not_called()


class TestQualityScoringCleanPctWarning:
    def test_clean_pct_below_30_triggers_warning(self, args, mock_vacancy):
        s = QualityScoringStage(args)
        s.args.excel = False
        mock_score = MagicMock()
        mock_score.is_spam = True
        mock_score.flags = [MagicMock(reason="spam")]

        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value.score.return_value = Ok(mock_score)
            MockScorer.return_value._build_report.return_value = {}
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                with patch("builtins.open"):
                    with patch("src.pipeline.stages.config") as cfg:
                        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
                        with patch("src.pipeline.stages.logger.warning") as mock_warn:
                            result = s.run(vacancies=[mock_vacancy], parser=MagicMock())

        assert result.is_ok()
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args
        assert call_args[0][0] == "hh_possible_similar_queries"
        assert call_args[1]["clean_pct"] == 0.0
        assert call_args[1]["spam_count"] == 1
        assert call_args[1]["total"] == 1

    def test_clean_pct_above_30_no_warning(self, args, mock_vacancy):
        s = QualityScoringStage(args)
        s.args.excel = False
        mock_score = MagicMock()
        mock_score.is_spam = False

        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value.score.return_value = Ok(mock_score)
            MockScorer.return_value._build_report.return_value = {}
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                with patch("builtins.open"):
                    with patch("src.pipeline.stages.config") as cfg:
                        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
                        with patch("src.pipeline.stages.logger.warning") as mock_warn:
                            result = s.run(vacancies=[mock_vacancy], parser=MagicMock())

        assert result.is_ok()
        mock_warn.assert_not_called()


class TestSkillExtractionRun:
    def test_run_ok_with_vacancy_objects(self, args):
        s = SkillExtractionStage(args)
        mock_vac = MagicMock(spec=Vacancy)
        mock_vac.raw_data = {}
        mock_parser = MagicMock()
        mock_parser.skill_parser.parse_vacancy.return_value = Ok([])

        mock_result = ({"python": 5}, {"python": 0.8}, MagicMock())

        with patch("src.pipeline.stages.SkillExtractor") as MockExtractor:
            MockExtractor.return_value.extract.return_value = Ok(mock_result)
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                result = s.run(vacancies=[mock_vac], parser=mock_parser, raw_file=None)

        assert result.is_ok()
        data = result.unwrap()
        assert data["skill_freq"] == {"python": 5}
        assert data["hybrid_weights_raw"] == {"python": 0.8}
        assert mock_vac.raw_data["extracted_skills"] == []

    def test_run_ok_with_dict_vacancies(self, args):
        s = SkillExtractionStage(args)
        mock_vac = {
            "id": "1",
            "name": "Python Developer",
            "area": {"id": 1, "name": "Moscow"},
            "employer": {"id": "1", "name": "Company"},
            "raw_data": {},
        }
        mock_parser = MagicMock()
        mock_parser.skill_parser.parse_vacancy.return_value = Ok([])

        mock_result = ({"sql": 3}, {"sql": 0.5}, MagicMock())

        with patch("src.pipeline.stages.SkillExtractor") as MockExtractor:
            MockExtractor.return_value.extract.return_value = Ok(mock_result)
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                result = s.run(vacancies=[mock_vac], parser=mock_parser, raw_file=None)

        assert result.is_ok()
        assert mock_vac["extracted_skills"] == []

    def test_run_ok_with_parse_error(self, args):
        s = SkillExtractionStage(args)
        mock_vac = MagicMock(spec=Vacancy)
        mock_vac.raw_data = {}
        mock_parser = MagicMock()
        mock_parser.skill_parser.parse_vacancy.return_value = Err("parse failed")

        mock_result = ({"python": 5}, {"python": 0.8}, MagicMock())

        with patch("src.pipeline.stages.SkillExtractor") as MockExtractor:
            MockExtractor.return_value.extract.return_value = Ok(mock_result)
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                result = s.run(vacancies=[mock_vac], parser=mock_parser, raw_file=None)

        assert result.is_ok()
        assert mock_vac.raw_data["extracted_skills"] == []

    def test_run_extract_err(self, args):
        s = SkillExtractionStage(args)
        mock_parser = MagicMock()

        with patch("src.pipeline.stages.SkillExtractor") as MockExtractor:
            MockExtractor.return_value.extract.return_value = Err("extraction failed")
            result = s.run(vacancies=[], parser=mock_parser, raw_file=None)

        assert result.is_err()


class TestModelTrainingAlreadyUpToDate:
    def test_already_up_to_date(self, args):
        s = ModelTrainingStage()
        mock_engine = MagicMock()
        mock_engine.is_fitted = True

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )

            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = True
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            model_path_mock.stat.return_value = MagicMock(st_mtime=200.0)
            detailed_mock.stat.return_value = MagicMock(st_mtime=100.0)

            with patch("src.pipeline.stages.create_ranking_predictor") as MockCreate:
                MockCreate.return_value = Ok(mock_engine)
                result = s.run()

        assert result.is_ok()
        assert result.unwrap()["reason"] == "already_up_to_date"

    def test_model_exists_but_load_fails_falls_through(self, args):
        s = ModelTrainingStage()
        mock_engine = MagicMock()
        mock_engine.is_fitted = False

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )

            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = True
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.create_ranking_predictor") as MockCreate:
                MockCreate.return_value = Ok(mock_engine)
                with patch("src.pipeline.stages.safe_read_json", return_value=[{"id": 1}]):
                    with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                        mock_instance = MockEngine.return_value
                        mock_instance.fit.return_value = Ok(())
                        mock_instance.is_fitted = True
                        mock_instance.last_metrics = {"r2": 0.8, "mae": 0.5}
                        result = s.run()

        assert result.is_ok()
        assert result.unwrap()["model_trained"] is True


class TestModelTrainingSuccess:
    def test_training_success(self, args):
        s = ModelTrainingStage()
        mock_engine = MagicMock()
        mock_engine.is_fitted = False

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )

            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = False
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.safe_read_json", return_value=[{"id": 1}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                    mock_instance = MockEngine.return_value
                    mock_instance.fit.return_value = Ok(())
                    mock_instance.is_fitted = True
                    mock_instance.last_metrics = {"r2": 0.8, "mae": 0.5}
                    result = s.run()

        assert result.is_ok()
        assert result.unwrap()["model_trained"] is True


class TestModelTrainingEmptyData:
    def test_safe_read_json_returns_empty(self, args):
        s = ModelTrainingStage()

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )

            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = False
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.safe_read_json", return_value=[]):
                result = s.run()

        assert result.is_err()

    def test_safe_read_json_returns_none(self, args):
        s = ModelTrainingStage()

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )

            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = False
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.safe_read_json", return_value=None):
                result = s.run()

        assert result.is_err()


class TestModelTrainingFitErrors:
    def test_fit_returns_error(self, args):
        s = ModelTrainingStage()

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )
            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = False
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.safe_read_json", return_value=[{"id": 1}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                    mock_instance = MockEngine.return_value
                    mock_instance.fit.return_value = Err("training failed")
                    result = s.run()

        assert result.is_err()

    def test_fit_ok_but_not_fitted(self, args):
        s = ModelTrainingStage()
        from unittest.mock import MagicMock

        mock_instance = MagicMock(spec=["fit", "is_fitted"])
        mock_instance.fit.return_value = Ok(())
        mock_instance.is_fitted = False

        with patch("src.pipeline.stages.config") as cfg:
            detailed_mock = MagicMock()
            detailed_mock.exists.return_value = True
            basic_mock = MagicMock()
            basic_mock.exists.return_value = False
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = lambda x: (
                detailed_mock if x == "hh_vacancies_detailed.json" else basic_mock
            )
            model_path_mock = MagicMock()
            model_path_mock.exists.return_value = False
            cfg.MODELS_DIR.__truediv__.return_value = model_path_mock

            with patch("src.pipeline.stages.safe_read_json", return_value=[{"id": 1}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine", return_value=mock_instance):
                    result = s.run()

        assert result.is_err()
