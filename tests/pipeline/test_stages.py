from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.pipeline.stages import (
    ClusterTrainingStage,
    DataCollectionStage,
    GapAnalysisStage,
    LevelBuildingStage,
    ModelTrainingStage,
    QualityScoringStage,
    SkillExtractionStage,
    WeightCleaningStage,
)


@pytest.fixture
def args():
    return MagicMock()


@pytest.fixture
def mock_vacancy():
    v = MagicMock()
    v.raw_data = {}
    return v


class TestSimpleStages:
    def test_data_collection_init(self, args):
        s = DataCollectionStage(args)
        assert s.name == "data_collection"
        assert s.pct_range == (0, 18)

    def test_data_collection_run(self, args):
        s = DataCollectionStage(args)
        with patch("src.pipeline.stages.HhDataSource") as MockSource:
            MockSource.return_value.get_vacancies.return_value = Ok(([], None))
            result = s.run()
        assert result.is_ok()

    def test_data_collection_run_fail(self, args):
        s = DataCollectionStage(args)
        mock_err = MagicMock()
        mock_err.message = "api error"
        with patch("src.pipeline.stages.HhDataSource") as MockSource:
            MockSource.return_value.get_vacancies.return_value = Err(mock_err)
            result = s.run()
        assert result.is_err()

    def test_quality_scoring_init(self, args):
        s = QualityScoringStage(args)
        assert s.name == "quality_scoring"
        assert s.pct_range == (18, 26)

    def test_quality_scoring_run_ok(self, args, mock_vacancy):
        s = QualityScoringStage(args)
        mock_score = MagicMock()
        mock_score.is_spam = False
        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value.score.return_value = Ok(mock_score)
            MockScorer.return_value._build_report.return_value = {}
            with patch("src.pipeline.helpers.save_detailed_vacancies"):
                with patch("builtins.open"):
                    with patch("src.pipeline.stages.config") as cfg:
                        cfg.REPORTS_DIR.__truediv__.return_value = MagicMock()
                        s.args.excel = False
                        result = s.run(vacancies=[mock_vacancy], parser=MagicMock())
        assert result.is_ok()

    def test_quality_scoring_run_spam(self, args, mock_vacancy):
        s = QualityScoringStage(args)
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
                        s.args.excel = False
                        result = s.run(vacancies=[mock_vacancy], parser=MagicMock())
        assert result.is_ok()
        assert mock_vacancy.raw_data.get("is_spam") is True

    def test_quality_scoring_run_fail(self, args, mock_vacancy):
        s = QualityScoringStage(args)
        mock_err = MagicMock()
        mock_err.message = "scoring error"
        with patch("src.pipeline.stages.VacancyQualityScorer") as MockScorer:
            MockScorer.return_value.score.return_value = Err(mock_err)
            result = s.run(vacancies=[mock_vacancy], parser=MagicMock())
        assert result.is_err()

    def test_skill_extraction_init(self, args):
        s = SkillExtractionStage(args)
        assert s.name == "skill_extraction"
        assert s.pct_range == (26, 40)

    def test_weight_cleaning_init(self, args):
        s = WeightCleaningStage()
        assert s.name == "weight_cleaning"
        assert s.pct_range == (40, 50)

    def test_level_building_init(self, args):
        s = LevelBuildingStage()
        assert s.name == "level_building"
        assert s.pct_range == (50, 60)

    def test_level_building_run(self, args):
        s = LevelBuildingStage()
        with patch("src.pipeline.stages.LevelBuilder") as MockBuilder:
            MockBuilder.return_value.build.return_value = Ok(([], []))
            result = s.run(vacancies=[], parser=MagicMock())
        assert result.is_ok()

    def test_cluster_training_init(self, args):
        s = ClusterTrainingStage()
        assert s.name == "cluster_training"
        assert s.pct_range == (60, 65)

    def test_model_training_init(self, args):
        s = ModelTrainingStage()
        assert s.name == "model_training"
        assert s.pct_range == (65, 70)

    def test_cluster_training_run_ok(self, args):
        s = ClusterTrainingStage()
        with patch("scripts.train_clusters.train_clusters", return_value=True):
            result = s.run()
        assert result.is_ok()

    def test_cluster_training_run_fail(self, args):
        s = ClusterTrainingStage()
        with patch("scripts.train_clusters.train_clusters", return_value=False):
            result = s.run()
        assert result.is_err()

    def test_weight_cleaning_run(self, args):
        s = WeightCleaningStage()
        with patch("src.pipeline.stages.WeightCleaner") as MockCleaner:
            MockCleaner.return_value.clean.return_value = Ok({"cleaned": {}})
            result = s.run(hybrid_weights_raw={"a": 1.0})
        assert result.is_ok()

    def test_weight_cleaning_clean_fail(self, args):
        s = WeightCleaningStage()
        with patch("src.pipeline.stages.WeightCleaner") as MockCleaner:
            MockCleaner.return_value.clean.return_value = Err("clean failed")
            result = s.run(hybrid_weights_raw={"a": 1.0})
        assert result.is_err()

    def test_model_training_run_no_vacancy_file(self, args):
        s = ModelTrainingStage()
        with patch("src.pipeline.stages.config") as cfg:
            def mock_truediv(key):
                m = MagicMock()
                m.exists.return_value = False
                return m
            cfg.DATA_PROCESSED_DIR.__truediv__.side_effect = mock_truediv
            cfg.DATA_RAW_DIR.__truediv__.side_effect = mock_truediv
            result = s.run()
        assert result.is_ok()
        assert result.unwrap()["reason"] == "no_vacancy_file"

    def test_gap_analysis_init(self, args):
        s = GapAnalysisStage({}, {}, args)
        assert s.name == "gap_analysis"
        assert s.pct_range == (70, 92)

    def test_gap_analysis_run_ok(self, args):
        s = GapAnalysisStage({"p1": {}}, MagicMock(), args)
        with patch("src.pipeline.stages.GapRunner") as MockRunner:
            MockRunner.return_value.run.return_value = Ok(({}, {}))
            result = s.run()
        assert result.is_ok()

    def test_gap_analysis_run_fail(self, args):
        s = GapAnalysisStage({"p1": {}}, MagicMock(), args)
        with patch("src.pipeline.stages.GapRunner") as MockRunner:
            MockRunner.return_value.run.return_value = Err("gap failed")
            result = s.run()
        assert result.is_err()
