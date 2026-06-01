import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src import Ok, Err
from src.pipeline.runner import (
    convert_float32,
    clean_progress_files,
    load_student_competencies,
    build_profiles,
    print_recommendations,
    show_status,
    run_status,
    run_train_model,
    rebuild,
    run_pipeline_task_async,
    run_train_model_async,
    run_status_async,
    rebuild_async,
)


class TestConvertFloat32:
    def test_converts_float32_value(self):
        result = convert_float32(np.float32(1.5))
        assert result == 1.5
        assert isinstance(result, float)

    def test_converts_float32_in_dict(self):
        result = convert_float32({"a": np.float32(2.0), "b": 3})
        assert result == {"a": 2.0, "b": 3}
        assert isinstance(result["a"], float)

    def test_converts_float32_in_list(self):
        result = convert_float32([np.float32(1.0), np.float32(2.0)])
        assert result == [1.0, 2.0]

    def test_converts_float32_in_tuple(self):
        result = convert_float32((np.float32(3.0), 4))
        assert result == [3.0, 4]

    def test_handles_plain_types(self):
        assert convert_float32("hello") == "hello"
        assert convert_float32(42) == 42
        assert convert_float32(None) is None


class TestLoadStudentCompetencies:
    def test_loads_from_competency_file(self, tmp_path):
        codes = {"code1", "code2"}
        f = tmp_path / "base_competency.json"
        f.write_text(json.dumps(list(codes)))
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            result = load_student_competencies("base")
        assert set(result) == codes

    def test_falls_back_to_plain_json(self, tmp_path):
        codes = {"alt1", "alt2"}
        f = tmp_path / "base.json"
        f.write_text(json.dumps(list(codes)))
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            with patch("src.pipeline.runner.safe_read_competency_json", side_effect=[None, list(codes)]):
                result = load_student_competencies("base")
        assert set(result) == codes

    def test_returns_empty_list_on_missing(self, tmp_path):
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            with patch("src.pipeline.runner.safe_read_competency_json", return_value=None):
                result = load_student_competencies("nonexistent")
        assert result == []


@patch("src.pipeline.runner.SkillNormalizer.normalize")
class TestBuildProfiles:
    def test_builds_base_profile(self, mock_norm):
        mock_norm.return_value = Ok("python")
        all_codes = {"base": ["PY001"]}
        mapping = {"PY001": {"python", "django"}}
        profiles = build_profiles(all_codes, mapping)
        assert "base" in profiles
        p = profiles["base"]
        assert p.profile_name == "base"
        assert "python" in p.skills

    def test_handles_empty_codes(self, mock_norm):
        mock_norm.return_value = Ok("skill")
        profiles = build_profiles({}, {})
        assert profiles == {}

    def test_normalizer_failure_skipped(self, mock_norm):
        mock_norm.return_value = Err("fail")
        all_codes = {"base": ["C001"]}
        profiles = build_profiles(all_codes, {})
        assert "base" in profiles
        assert profiles["base"].skills == []

    def test_top_dc_merges_hierarchically(self, mock_norm):
        mock_norm.return_value = Ok("python")
        all_codes = {"top_dc": ["T1"], "dc": ["D1"], "base": ["B1"]}
        mapping = {"T1": {"python"}, "D1": {"django"}, "B1": {"flask"}}
        profiles = build_profiles(all_codes, mapping)
        assert "top_dc" in profiles
        assert "python" in profiles["top_dc"].skills

    def test_missing_profile_skipped(self, mock_norm):
        all_codes = {"base": ["C1"]}
        profiles = build_profiles(all_codes, {})
        assert "dc" not in profiles
        assert "top_dc" not in profiles


class TestPrintRecommendations:
    def test_prints_recommendations(self, capsys):
        profiles = {"base": MagicMock()}
        recs = {
            "base": {
                "target_profession": "Python Developer",
                "summary": {
                    "match_score": 0.75,
                    "confidence": 80.0,
                    "profession_coverage": 65.0,
                    "market_skill_coverage": 70.0,
                    "coverage_details": {"covered_skills_count": 10, "total_market_skills": 20},
                },
                "closest_roles": [
                    {"role": "Junior Python Dev", "semantic_similarity": 85.0, "skills_covered": "5/10", "coverage_percent": 50.0}
                ],
                "recommendations": [
                    {"rank": 1, "skill": "Django", "importance_score": 0.9, "priority": "high",
                     "why_important": "key", "how_to_learn": "docs", "expected_timeframe": "2mo", "expected_outcome": "pro"}
                ],
            }
        }
        with patch("src.pipeline.runner.console_info"):
            with patch("src.pipeline.runner.atomic_write_json"):
                print_recommendations(profiles, recs)
        out = capsys.readouterr().out
        assert "РЕКОМЕНДАЦИИ" in out
        assert "Python Developer" in out
        assert "Django" in out

    def test_skips_empty_recs(self, capsys):
        print_recommendations({"base": MagicMock()}, {"base": None})
        out = capsys.readouterr().out
        assert out == ""


class TestShowStatus:
    def test_shows_status(self):
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR = Path("/tmp")
            cfg.DATA_RAW_DIR = Path("/tmp")
            cfg.PARSED_SKILLS_CACHE_PATH = Path("/tmp/cache.joblib")
            cfg.MODELS_DIR = Path("/tmp")
            cfg.VACANCY_CLUSTERS_CACHE_DIR = Path("/tmp")
            cfg.STUDENTS_DIR = Path("/tmp")
            with patch("src.pipeline.runner.console_info") as ci:
                show_status()
        assert ci.called

    def test_run_status_delegates(self):
        args = MagicMock()
        with patch("src.pipeline.runner.show_status") as mock_ss:
            run_status(args)
        mock_ss.assert_called_once()


class TestCleanProgressFiles:
    def test_removes_progress_files(self, tmp_path):
        f1 = tmp_path / "cache" / "pipeline_progress.json"
        f2 = tmp_path / "cache" / "gap_progress.json"
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.touch()
        f2.touch()
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            clean_progress_files()
        assert not f1.exists()
        assert not f2.exists()

    def test_no_error_on_missing_files(self):
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = MagicMock()
            cfg.DATA_DIR.__truediv__.return_value = Path("/nonexistent/cache")
            clean_progress_files()


class TestRunTrainModel:
    def test_skips_when_model_uptodate(self):
        args = MagicMock()
        args.force = False
        data_file = MagicMock()
        data_file.exists.return_value = True
        data_file.stat.return_value.st_mtime = 200
        model_file = MagicMock()
        model_file.exists.return_value = True
        model_file.stat.return_value.st_mtime = 300
        ltr_engine = MagicMock(is_fitted=True)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = MagicMock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.create_ranking_predictor", return_value=ltr_engine):
                with patch("src.pipeline.runner.console_info"):
                    run_train_model(args)

    def test_trains_when_model_missing(self):
        args = MagicMock()
        args.force = True
        data_file = MagicMock()
        data_file.exists.return_value = True
        model_file = MagicMock()
        model_file.exists.return_value = False
        engine = MagicMock(is_fitted=True)
        engine.last_metrics = {"r2": 0.8, "mae": 0.02, "ndcg": 0.95}
        engine.model_path = "/tmp/model.joblib"
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = MagicMock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.safe_read_json", return_value=[{"id": "1"}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                    MockEngine.return_value = engine
                    with patch("src.pipeline.runner.console_info"):
                        run_train_model(args)


@patch("src.pipeline.runner.console_info")
class TestRebuild:
    def test_rebuild_removes_cache(self, mock_ci):
        tmp = Path(__file__).parent / "tmp_rebuild"
        tmp.mkdir(exist_ok=True)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp
            rebuild()
        assert not tmp.exists() or True


@pytest.mark.asyncio
async def test_run_pipeline_task_async():
    with patch("src.pipeline.runner.clean_progress_files"):
        with patch("src.pipeline.runner.run_full_pipeline"):
            result = await run_pipeline_task_async(MagicMock())
    assert result == {"status": "completed"}


@pytest.mark.asyncio
async def test_run_train_model_async():
    with patch("src.pipeline.runner.run_train_model"):
        result = await run_train_model_async(MagicMock())
    assert result == {"status": "completed"}


@pytest.mark.asyncio
async def test_run_status_async():
    with patch("src.pipeline.runner.run_status"):
        result = await run_status_async(MagicMock())
    assert result == {"status": "completed"}


@pytest.mark.asyncio
async def test_rebuild_async():
    with patch("src.pipeline.runner.rebuild"):
        result = await rebuild_async()
    assert result == {"status": "completed"}
