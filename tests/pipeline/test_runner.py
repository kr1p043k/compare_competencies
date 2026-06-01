import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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
        students = tmp_path / "students"
        students.mkdir()
        codes = ["code1", "code2"]
        f = students / "base_competency.json"
        f.write_text(json.dumps({"codes": codes}))
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            result = load_student_competencies("base")
        assert result == codes

    def test_falls_back_to_plain_json(self, tmp_path):
        students = tmp_path / "students"
        students.mkdir()
        codes = ["alt1", "alt2"]
        f = students / "base.json"
        f.write_text(json.dumps({"codes": codes}))
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            with patch("src.pipeline.runner.safe_read_competency_json", side_effect=[None, codes]):
                result = load_student_competencies("base")
        assert result == codes

    def test_returns_empty_list_on_missing(self, tmp_path):
        students = tmp_path / "students"
        students.mkdir()
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
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
                "trend_bonuses_count": 3,
                "dominant_domain_name": "Backend",
                "closest_roles": [
                    {"role": "Junior Python Dev", "semantic_similarity": 85.0, "skills_covered": "5/10", "coverage_percent": 50.0, "coverage_explanation": "good match"}
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
        assert "Backend" in out

    def test_prints_without_trends_or_roles(self, capsys):
        profiles = {"base": MagicMock()}
        recs = {
            "base": {
                "summary": {
                    "match_score": 0.5, "confidence": 60.0, "profession_coverage": 50.0,
                    "market_skill_coverage": 40.0,
                    "coverage_details": {"covered_skills_count": 5, "total_market_skills": 10},
                },
                "recommendations": [],
            }
        }
        with patch("src.pipeline.runner.console_info"):
            with patch("src.pipeline.runner.atomic_write_json"):
                print_recommendations(profiles, recs)
        out = capsys.readouterr().out
        assert "РЕКОМЕНДАЦИИ" in out

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

    def test_shows_status_with_existing_items(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "cache" / "clusters").mkdir(parents=True)
        (tmp_path / "students").mkdir()
        ltr_model = tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib"
        ltr_model.touch()
        manifest = ltr_model.with_suffix(".manifest.json")
        manifest.write_text('{"metrics": {"r2": 0.8}}')
        cluster_file = tmp_path / "cache" / "clusters" / "vacancy_clusters_junior.pkl"
        cluster_file.touch()
        student_file = tmp_path / "students" / "base_competency.json"
        student_file.touch()
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR = tmp_path / "processed"
            cfg.DATA_RAW_DIR = tmp_path / "raw"
            cfg.PARSED_SKILLS_CACHE_PATH = tmp_path / "parsed_cache.joblib"
            cfg.MODELS_DIR = tmp_path / "models"
            cfg.VACANCY_CLUSTERS_CACHE_DIR = tmp_path / "cache" / "clusters"
            cfg.STUDENTS_DIR = tmp_path / "students"
            with patch("src.pipeline.runner.console_info") as ci:
                show_status()
        assert ci.called

    def test_shows_status_with_corrupt_manifest(self, tmp_path):
        (tmp_path / "models").mkdir()
        ltr_model = tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib"
        ltr_model.touch()
        manifest = ltr_model.with_suffix(".manifest.json")
        manifest.write_text("not json")
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR = tmp_path
            cfg.DATA_RAW_DIR = tmp_path
            cfg.PARSED_SKILLS_CACHE_PATH = tmp_path / "cache.joblib"
            cfg.MODELS_DIR = tmp_path / "models"
            cfg.VACANCY_CLUSTERS_CACHE_DIR = tmp_path
            cfg.STUDENTS_DIR = tmp_path
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
    def _make_path_mock(self, exists=True, mtime=100.0):
        m = MagicMock()
        m.exists.return_value = exists
        stat = MagicMock()
        stat.st_mtime = mtime
        m.stat.return_value = stat
        return m

    def test_skips_when_model_uptodate(self):
        args = MagicMock()
        args.force = False
        data_file = self._make_path_mock(exists=True, mtime=200)
        model_file = self._make_path_mock(exists=True, mtime=300)
        ltr_engine = MagicMock(is_fitted=True)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = self._make_path_mock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.create_ranking_predictor", return_value=ltr_engine):
                with patch("src.pipeline.runner.console_info"):
                    run_train_model(args)

    def test_skips_when_force_and_model_outdated(self):
        args = MagicMock()
        args.force = True
        raw_file = self._make_path_mock(exists=True, mtime=300)
        model_file = self._make_path_mock(exists=True, mtime=200)
        ltr_engine = MagicMock(is_fitted=True)
        ltr_engine.last_metrics = {"r2": 0.8, "mae": 0.02, "ndcg": 0.95}
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = raw_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = self._make_path_mock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.create_ranking_predictor", return_value=ltr_engine):
                with patch("src.pipeline.runner.safe_read_json", return_value=[{"id": "1"}]):
                    with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                        MockEngine.return_value = ltr_engine
                        with patch("src.pipeline.runner.console_info"):
                            run_train_model(args)

    def test_trains_when_model_missing(self):
        args = MagicMock()
        args.force = True
        data_file = self._make_path_mock(exists=True)
        model_file = self._make_path_mock(exists=False)
        engine = MagicMock(is_fitted=True)
        engine.last_metrics = {"r2": 0.8, "mae": 0.02, "ndcg": 0.95}
        engine.model_path = "/tmp/model.joblib"
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = self._make_path_mock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.safe_read_json", return_value=[{"id": "1"}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                    MockEngine.return_value = engine
                    with patch("src.pipeline.runner.console_info"):
                        run_train_model(args)

    def test_training_fit_fails(self):
        args = MagicMock()
        args.force = True
        data_file = self._make_path_mock(exists=True)
        model_file = self._make_path_mock(exists=False)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = self._make_path_mock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.safe_read_json", return_value=[{"id": "1"}]):
                with patch("src.predictors.ltr_recommendation_engine.LTRRecommendationEngine") as MockEngine:
                    engine = MockEngine.return_value
                    engine.is_fitted = False
                    engine.last_metrics = {"r2": 0.8, "mae": 0.02, "ndcg": 0.95}
                    with patch("src.pipeline.runner.console_info"):
                        run_train_model(args)

    def test_training_empty_vacancies(self):
        args = MagicMock()
        args.force = True
        data_file = self._make_path_mock(exists=True)
        model_file = self._make_path_mock(exists=False)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR.__truediv__.return_value = data_file
            cfg.DATA_RAW_DIR.__truediv__.return_value = self._make_path_mock(exists=False)
            cfg.MODELS_DIR.__truediv__.return_value = model_file
            with patch("src.pipeline.runner.safe_read_json", return_value=[]):
                with patch("src.pipeline.runner.console_info"):
                    with pytest.raises(SystemExit):
                        run_train_model(args)


@patch("src.pipeline.runner.console_info")
class TestRebuild:
    def test_rebuild_removes_cache(self, mock_ci):
        tmp = Path(__file__).parent / "tmp_rebuild"
        tmp.mkdir(exist_ok=True)
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp
            rebuild()

    def test_rebuild_with_existing_files(self, mock_ci, tmp_path):
        (tmp_path / "cache" / "embeddings").mkdir(parents=True)
        (tmp_path / "cache" / "clusters").mkdir(parents=True)
        (tmp_path / "processed").mkdir()
        (tmp_path / "models").mkdir()
        to_touch = [
            tmp_path / "cache" / "parsed_skills.joblib",
            tmp_path / "processed" / "skill_weights.json",
            tmp_path / "cache" / "clusters" / "vacancy_clusters_junior.pkl",
            tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib",
        ]
        for f in to_touch:
            f.touch()
        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            rebuild()
        for f in to_touch:
            assert not f.exists()
        assert not (tmp_path / "cache" / "embeddings").exists()


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
