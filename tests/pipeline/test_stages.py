from unittest.mock import MagicMock, patch

import pytest

from src import Ok
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


class TestSimpleStages:
    def test_data_collection_init(self, args):
        s = DataCollectionStage(args)
        assert s.name == "data_collection"
        assert s.pct_range == (0, 18)

    def test_quality_scoring_init(self, args):
        s = QualityScoringStage(args)
        assert s.name == "quality_scoring"
        assert s.pct_range == (18, 26)

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
