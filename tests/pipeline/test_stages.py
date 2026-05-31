from unittest.mock import MagicMock, patch

import pytest

from src import Ok, Err
from src.pipeline.stages import (
    DataCollectionStage,
    QualityScoringStage,
    SkillExtractionStage,
    WeightCleaningStage,
    LevelBuildingStage,
    ClusterTrainingStage,
    ModelTrainingStage,
    GapAnalysisStage,
)


class TestDataCollectionStage:
    def test_init(self):
        s = DataCollectionStage(args=MagicMock())
        assert s.name == "data_collection"
        assert s.pct_range == (0, 18)

    @patch("src.pipeline.stages.HhDataSource")
    def test_run_success(self, MockDS):
        mock_source = MagicMock()
        mock_source.get_vacancies.return_value = Ok((["vac1", "vac2"], MagicMock()))
        MockDS.return_value = mock_source
        s = DataCollectionStage(args=MagicMock(skip_collection=False))
        result = s.run()
        assert result.is_ok()

    @patch("src.pipeline.stages.HhDataSource")
    def test_run_failure(self, MockDS):
        mock_source = MagicMock()
        mock_source.get_vacancies.return_value = Err("api error")
        MockDS.return_value = mock_source
        s = DataCollectionStage(args=MagicMock(skip_collection=False))
        result = s.run()
        assert result.is_err()


class TestQualityScoringStage:
    def test_init(self):
        s = QualityScoringStage(args=MagicMock())
        assert s.name == "quality_scoring"

    @patch("src.pipeline.stages.VacancyQualityScorer")
    def test_run(self, MockScorer):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (["v1"], 0)
        MockScorer.return_value = mock_scorer
        s = QualityScoringStage(args=MagicMock())
        result = s.run(vacancies=["v1", "v2"], parser=MagicMock())
        assert result.is_ok()


class TestSkillExtractionStage:
    @patch("src.pipeline.stages.SkillExtractor")
    def test_run(self, MockExtractor):
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = (["skill1", "skill2"], ["v1"])
        MockExtractor.return_value = mock_extractor
        s = SkillExtractionStage(args=MagicMock())
        result = s.run(vacancies=["v1"], parser=MagicMock())
        assert result.is_ok()


class TestModelTrainingStage:
    @patch("src.pipeline.stages.create_ranking_predictor")
    def test_run(self, mock_create):
        mock_engine = MagicMock()
        mock_engine.fit.return_value = Ok(MagicMock())
        mock_create.return_value = mock_engine
        s = ModelTrainingStage(args=MagicMock())
        result = s.run(all_vacancies=["v1"], processed_vacancies=["v1"], skills_data={}, clusters={}, levels={})
        assert result.is_ok()


class TestGapAnalysisStage:
    @patch("src.pipeline.stages.GapRunner")
    def test_run(self, MockRunner):
        mock_runner = MagicMock()
        mock_runner.run.return_value = Ok({"base": {}})
        MockRunner.return_value = mock_runner
        s = GapAnalysisStage(args=MagicMock())
        result = s.run(vacancies=[], clusters={}, levels={}, ltr_model=None, skills_data={}, cleaned_weights={})
        assert result.is_ok()


class TestStageConstruction:
    def test_all_stages_have_name(self):
        args = MagicMock()
        stages = [
            DataCollectionStage(args),
            QualityScoringStage(args),
            SkillExtractionStage(args),
            WeightCleaningStage(args),
            LevelBuildingStage(args),
            ClusterTrainingStage(args),
            ModelTrainingStage(args),
            GapAnalysisStage(args),
        ]
        for s in stages:
            assert s.name
            assert len(s.pct_range) == 2
            assert s.pct_range[1] > s.pct_range[0]
