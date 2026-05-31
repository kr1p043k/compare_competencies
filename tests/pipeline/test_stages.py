from unittest.mock import MagicMock

import pytest

from src.pipeline.stages import (
    DataCollectionStage,
    QualityScoringStage,
    SkillExtractionStage,
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
