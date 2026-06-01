from pathlib import Path
from unittest.mock import MagicMock, patch

from src import Ok, Err
from src.pipeline.weight_cleaner import WeightCleaner


class TestWeightCleaner:
    def test_clean_ok(self):
        wc = WeightCleaner()
        with patch("src.pipeline.weight_cleaner.SkillFilter") as MockFilter:
            with patch("src.pipeline.weight_cleaner.config") as cfg:
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = False
                cfg.DATA_PROCESSED_DIR.__truediv__.return_value = mock_path
                MockFilter.return_value.get_clean_weights.return_value = Ok({"python": 0.5})
                result = wc.clean({"python": 0.5})
        assert result.is_ok()
        assert result.unwrap() == {"python": 0.5}

    def test_clean_skill_filter_error(self):
        wc = WeightCleaner()
        with patch("src.pipeline.weight_cleaner.SkillFilter") as MockFilter:
            with patch("src.pipeline.weight_cleaner.config") as cfg:
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = False
                cfg.DATA_PROCESSED_DIR.__truediv__.return_value = mock_path
                MockFilter.return_value.get_clean_weights.return_value = Err("filter failed")
                result = wc.clean({"python": 0.5})
        assert result.is_err()

    def test_clean_exception(self):
        wc = WeightCleaner()
        with patch("src.pipeline.weight_cleaner.SkillFilter") as MockFilter:
            MockFilter.side_effect = RuntimeError("unexpected")
            result = wc.clean({"python": 0.5})
        assert result.is_err()
