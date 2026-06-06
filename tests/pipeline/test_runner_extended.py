"""Extended tests for pipeline runner — uncovered code paths."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src import Ok, Err
from src.pipeline.runner import (
    build_profiles,
    print_recommendations,
    show_status,
    run_full_pipeline,
    rebuild,
)


def _make_path_mock(exists=True, mtime=100.0):
    m = MagicMock()
    m.exists.return_value = exists
    stat = MagicMock()
    stat.st_mtime = mtime
    m.stat.return_value = stat
    return m


# ── build_profiles (competency_mapping non-empty) ──────────────────────

class TestBuildProfilesExtended:
    """Additional build_profiles tests — competency_mapping non-empty paths."""

    @patch("src.pipeline.runner.SkillNormalizer.normalize")
    def test_mapping_maps_codes_to_skills(self, mock_norm):
        mock_norm.side_effect = lambda s: Ok(s)
        all_codes = {"base": ["PY001"]}
        mapping = {"PY001": {"python", "django"}}
        profiles = build_profiles(all_codes, mapping)
        p = profiles["base"]
        assert "python" in p.skills
        assert "django" in p.skills
        assert p.competencies == ["PY001"]

    @patch("src.pipeline.runner.SkillNormalizer.normalize")
    def test_mapping_multiple_codes(self, mock_norm):
        mock_norm.side_effect = lambda s: Ok(s)
        all_codes = {"base": ["PY001", "PY002"]}
        mapping = {"PY001": {"python"}, "PY002": {"django"}}
        profiles = build_profiles(all_codes, mapping)
        assert "python" in profiles["base"].skills
        assert "django" in profiles["base"].skills

    @patch("src.pipeline.runner.SkillNormalizer.normalize")
    def test_mapping_no_match_returns_empty_skills(self, mock_norm):
        mock_norm.return_value = Ok("x")
        all_codes = {"base": ["CODE1"]}
        mapping = {"NOMATCH": {"skill_a"}}
        profiles = build_profiles(all_codes, mapping)
        assert profiles["base"].skills == []

    @patch("src.pipeline.runner.SkillNormalizer.normalize")
    def test_mapping_normalization_failure_skipped(self, mock_norm):
        mock_norm.return_value = Err("bad")
        all_codes = {"base": ["PY001"]}
        mapping = {"PY001": {"python"}}
        profiles = build_profiles(all_codes, mapping)
        assert profiles["base"].skills == []

    @patch("src.pipeline.runner.SkillNormalizer.normalize")
    def test_top_dc_with_mapping(self, mock_norm):
        mock_norm.side_effect = lambda s: Ok(s)
        all_codes = {"top_dc": ["T1"], "dc": ["D1"], "base": ["B1"]}
        mapping = {"T1": {"python"}, "D1": {"django"}, "B1": {"flask"}}
        from src.models.student import merge_skills_hierarchically
        profiles = build_profiles(all_codes, mapping)
        assert "top_dc" in profiles
        assert profiles["top_dc"].competencies == ["T1"]


# ── print_recommendations (coverage_explanation) ─────────────────────

class TestPrintRecommendationsExtended:
    """Additional print_recommendations tests — coverage_explanation."""

    def test_first_role_coverage_explanation_printed(self, capsys):
        profiles = {"base": MagicMock()}
        recs = {
            "base": {
                "target_profession": "Dev",
                "summary": {
                    "match_score": 0.8, "confidence": 90.0, "profession_coverage": 70.0,
                    "market_skill_coverage": 60.0,
                    "coverage_details": {"covered_skills_count": 8, "total_market_skills": 15},
                },
                "closest_roles": [
                    {"role": "Dev A", "semantic_similarity": 90.0, "skills_covered": "5/10",
                     "coverage_percent": 50.0, "coverage_explanation": "strong alignment"},
                    {"role": "Dev B", "semantic_similarity": 80.0, "skills_covered": "3/10",
                     "coverage_percent": 30.0, "coverage_explanation": "partial match"},
                ],
                "recommendations": [
                    {"rank": 1, "skill": "Python", "importance_score": 0.9, "priority": "high",
                     "why_important": "key", "how_to_learn": "docs", "expected_timeframe": "2mo",
                     "expected_outcome": "pro"},
                ],
            }
        }
        with patch("src.pipeline.runner.console_info"):
            with patch("src.pipeline.runner.atomic_write_json"):
                print_recommendations(profiles, recs)
        out = capsys.readouterr().out
        assert "strong alignment" in out
        assert "partial match" not in out

    def test_no_coverage_explanation_when_key_missing(self, capsys):
        profiles = {"base": MagicMock()}
        recs = {
            "base": {
                "summary": {
                    "match_score": 0.5, "confidence": 50.0, "profession_coverage": 40.0,
                    "market_skill_coverage": 30.0,
                    "coverage_details": {"covered_skills_count": 3, "total_market_skills": 10},
                },
                "closest_roles": [
                    {"role": "Dev A", "semantic_similarity": 85.0, "skills_covered": "4/10",
                     "coverage_percent": 40.0},
                ],
                "recommendations": [],
            }
        }
        with patch("src.pipeline.runner.console_info"):
            with patch("src.pipeline.runner.atomic_write_json"):
                print_recommendations(profiles, recs)
        out = capsys.readouterr().out
        assert "\u2139" not in out


# ── show_status (working manifest) ───────────────────────────────────

class TestShowStatusExtended:
    """Additional show_status tests — existing ltr model + working manifest."""

    def test_working_manifest_displays_r2(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        ltr_path = models / "ltr_ranker_xgb_regressor.joblib"
        ltr_path.touch()
        manifest = ltr_path.with_suffix(".manifest.json")
        manifest.write_text('{"metrics": {"r2": 0.85}}')

        clusters = tmp_path / "clusters"
        clusters.mkdir()
        (clusters / "vacancy_clusters_junior.pkl").touch()

        students = tmp_path / "students"
        students.mkdir()
        (students / "base_competency.json").touch()

        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR = tmp_path / "processed"
            cfg.DATA_RAW_DIR = tmp_path / "raw"
            cfg.PARSED_SKILLS_CACHE_PATH = tmp_path / "parsed.joblib"
            cfg.MODELS_DIR = models
            cfg.VACANCY_CLUSTERS_CACHE_DIR = clusters
            cfg.STUDENTS_DIR = students
            with patch("src.pipeline.runner.console_info") as ci:
                show_status()
        r2_calls = [c for c in ci.call_args_list if "R²=0.85" in str(c.args)]
        assert r2_calls, "Expected R²=0.85 in console_info output"

    def test_model_without_manifest_shows_no_r2(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        ltr_path = models / "ltr_ranker_xgb_regressor.joblib"
        ltr_path.touch()
        clusters = tmp_path / "clusters"
        clusters.mkdir()
        students = tmp_path / "students"
        students.mkdir()
        (students / "base_competency.json").touch()

        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_PROCESSED_DIR = tmp_path / "processed"
            cfg.DATA_RAW_DIR = tmp_path / "raw"
            cfg.PARSED_SKILLS_CACHE_PATH = tmp_path / "parsed.joblib"
            cfg.MODELS_DIR = models
            cfg.VACANCY_CLUSTERS_CACHE_DIR = clusters
            cfg.STUDENTS_DIR = students
            with patch("src.pipeline.runner.console_info") as ci:
                show_status()
        no_manifest = [c for c in ci.call_args_list if "без манифеста" in str(c.args)]
        assert no_manifest, "Expected 'без манифеста' in output"


# ── run_full_pipeline ────────────────────────────────────────────────

class TestRunFullPipeline:
    """Tests for the main pipeline function (run_full_pipeline)."""

    def _setup_mocks(self, **overrides):
        opts = {
            "pipeline_is_err": False,
            "skip_collection": False,
            "skip_gap_analysis": False,
            "run_notebooks": False,
            "csv_exists": False,
            "gap_is_err": False,
            "csv_update_ok": True,
        }
        opts.update(overrides)

        stack = contextlib.ExitStack()
        mocks = {}

        for name in ["console_header", "console_info", "_write_pipeline_progress",
                      "show_context_info", "print_recommendations", "save_all_charts"]:
            mocks[name] = stack.enter_context(patch(f"src.pipeline.runner.{name}"))

        mocks["logger"] = stack.enter_context(patch("src.pipeline.runner.logger"))
        mocks["timed_block"] = stack.enter_context(patch("src.pipeline.runner.timed_block"))
        mocks["sys_exit"] = stack.enter_context(patch("src.pipeline.runner.sys.exit"))

        if opts["run_notebooks"]:
            mocks["run_notebook"] = stack.enter_context(patch("src.pipeline.runner.run_notebook"))

        _Ok = Ok if opts["csv_update_ok"] else Err
        mocks["generate_profiles_from_csv"] = stack.enter_context(
            patch("src.pipeline.runner.generate_profiles_from_csv",
                  return_value=_Ok(None) if opts["csv_update_ok"] else Err(
                      type("E", (), {"message": "csv error"})())))

        mock_cfg = MagicMock()
        mock_cfg.DATA_DIR = MagicMock()
        mock_cfg.REPORTS_DIR = MagicMock()
        mock_cfg.DATA_RAW_DIR = MagicMock()
        mock_cfg.LOG_FILE = MagicMock()
        mocks["config"] = stack.enter_context(patch("src.pipeline.runner.config", mock_cfg))

        csv_mock = MagicMock()
        csv_mock.exists.return_value = opts["csv_exists"]
        mock_cfg.DATA_RAW_DIR.__truediv__.return_value = csv_mock

        fallback_csv = MagicMock()
        fallback_csv.exists.return_value = opts["csv_exists"]
        mock_cfg.DATA_DIR.__truediv__.return_value = fallback_csv

        args = MagicMock()
        args.skip_collection = opts["skip_collection"]
        args.skip_gap_analysis = opts["skip_gap_analysis"]
        args.run_notebooks = opts["run_notebooks"]

        for sn in ["DataCollectionStage", "QualityScoringStage",
                    "SkillExtractionStage", "WeightCleaningStage",
                    "LevelBuildingStage", "ClusterTrainingStage",
                    "ModelTrainingStage"]:
            stack.enter_context(patch(f"src.pipeline.runner.{sn}"))

        if opts["pipeline_is_err"]:
            pipeline_result = MagicMock()
            pipeline_result.is_err.return_value = True
            pipeline_result.err.return_value = "pipeline failure"
        else:
            stage_sr = MagicMock()
            stage_sr.data = {"skill_freq": {}, "hybrid_weights": {},
                              "trend_analyzer": None, "level_data": [],
                              "vacancies_skills": []}
            run_obj = MagicMock()
            run_obj.stages = [stage_sr]
            run_obj.elapsed = 30.0
            pipeline_result = MagicMock()
            pipeline_result.is_err.return_value = False
            pipeline_result.unwrap.return_value = run_obj

        mock_orch = MagicMock()
        mock_orch.run.return_value = pipeline_result
        mocks["PipelineOrchestrator"] = stack.enter_context(
            patch("src.pipeline.runner.PipelineOrchestrator", return_value=mock_orch))

        mocks["load_competency_mapping"] = stack.enter_context(
            patch("src.pipeline.runner.load_competency_mapping", return_value={}))

        mocks["load_student_competencies"] = stack.enter_context(
            patch("src.pipeline.runner.load_student_competencies", return_value=["code1"]))

        mock_profile = MagicMock()
        mock_profile.profile_name = "base"
        mocks["build_profiles"] = stack.enter_context(
            patch("src.pipeline.runner.build_profiles",
                  return_value={"base": mock_profile}))

        if not opts["skip_gap_analysis"]:
            gap_stage = MagicMock()
            if opts["gap_is_err"]:
                gap_stage.run.return_value = Err("gap error")
            else:
                gap_data = MagicMock()
                gap_data.get.side_effect = lambda k, d=None: {
                    "evaluations": [{"eval": "data"}],
                    "recommendations": {"base": {"rec": "data"}},
                }.get(k, d)
                gap_stage.run.return_value = Ok(gap_data)
            mocks["GapAnalysisStage"] = stack.enter_context(
                patch("src.pipeline.runner.GapAnalysisStage", return_value=gap_stage))

        return mocks, args, stack

    # ── error path ──

    def test_error_when_pipeline_fails(self):
        mocks, args, stack = self._setup_mocks(pipeline_is_err=True)
        with stack:
            run_full_pipeline(args)
        mocks["console_info"].assert_any_call(
            "\u274c \u041f\u0430\u0439\u043f\u043b\u0430\u0439\u043d \u043d\u0435 "
            "\u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043d: pipeline failure")

    # ── success path ──

    def test_success_path(self):
        mocks, args, stack = self._setup_mocks()
        with stack:
            run_full_pipeline(args)
        mocks["console_header"].assert_called()
        mocks["show_context_info"].assert_called_once()

    # ── skip_gap_analysis ──

    def test_skip_gap_analysis(self):
        mocks, args, stack = self._setup_mocks(skip_gap_analysis=True)
        with stack:
            run_full_pipeline(args)
        assert "GapAnalysisStage" not in mocks
        mocks["save_all_charts"].assert_not_called()

    # ── run_notebooks ──

    def test_run_notebooks(self):
        mocks, args, stack = self._setup_mocks(run_notebooks=True)
        with stack:
            run_full_pipeline(args)
        mocks["run_notebook"].assert_has_calls([
            call("01_hh_analysis.ipynb", output_dir=mocks["config"].DATA_DIR / "notebooks"),
            call("02_competency_matching.ipynb", output_dir=mocks["config"].DATA_DIR / "notebooks"),
        ], any_order=True)

    # ── csv_path exists ──

    def test_csv_path_exists_ok(self):
        mocks, args, stack = self._setup_mocks(csv_exists=True, csv_update_ok=True)
        with stack:
            run_full_pipeline(args)
        mocks["generate_profiles_from_csv"].assert_called_once()

    def test_csv_path_exists_fail(self):
        mocks, args, stack = self._setup_mocks(csv_exists=True, csv_update_ok=False)
        with stack:
            run_full_pipeline(args)
        mocks["generate_profiles_from_csv"].assert_called_once()
        mocks["logger"].warning.assert_called()

    # ── gap analysis failure ──

    def test_gap_analysis_failure_exits(self):
        mocks, args, stack = self._setup_mocks(gap_is_err=True)
        with stack:
            run_full_pipeline(args)
        mocks["logger"].error.assert_called()

    # ── skip_collection ──

    def test_skip_collection(self):
        mocks, args, stack = self._setup_mocks(skip_collection=True)
        with stack:
            run_full_pipeline(args)
        mocks["console_header"].assert_called()

    # ── no profiles, skip_gap_analysis=False ──

    def test_no_profiles_skips_gap_analysis(self):
        mocks, args, stack = self._setup_mocks(skip_gap_analysis=False)
        mocks["build_profiles"].return_value = {}
        with stack:
            run_full_pipeline(args)
        mocks["GapAnalysisStage"].run.assert_not_called()
        mocks["save_all_charts"].assert_not_called()


# ── rebuild (cache_dir.exists() = True) ─────────────────────────────

class TestRebuildExtended:
    """Additional rebuild tests."""

    def test_all_remove_files_exist(self, tmp_path):
        (tmp_path / "cache" / "embeddings").mkdir(parents=True)
        (tmp_path / "cache" / "clusters").mkdir(parents=True)
        (tmp_path / "processed").mkdir()
        (tmp_path / "models").mkdir()

        to_touch = [
            tmp_path / "cache" / "parsed_skills.joblib",
            tmp_path / "processed" / "skill_weights.json",
            tmp_path / "cache" / "clusters" / "vacancy_clusters_junior.pkl",
            tmp_path / "cache" / "clusters" / "vacancy_clusters_middle.pkl",
            tmp_path / "cache" / "clusters" / "vacancy_clusters_senior.pkl",
            tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib",
            tmp_path / "cache" / "embeddings" / "market_embeddings_junior.pkl",
            tmp_path / "cache" / "embeddings" / "market_embeddings_middle.pkl",
            tmp_path / "cache" / "embeddings" / "market_embeddings_senior.pkl",
        ]
        for f in to_touch:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()

        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            with patch("src.pipeline.runner.console_info"):
                rebuild()

        assert not (tmp_path / "cache" / "embeddings").exists()
        assert not (tmp_path / "cache" / "clusters").exists()
        for f in to_touch:
            assert not f.exists()

    def test_cache_dir_does_not_exist(self, tmp_path):
        (tmp_path / "processed").mkdir()
        (tmp_path / "models").mkdir()
        to_touch = [
            tmp_path / "cache" / "parsed_skills.joblib",
            tmp_path / "models" / "ltr_ranker_xgb_regressor.joblib",
        ]
        for f in to_touch:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()

        with patch("src.pipeline.runner.config") as cfg:
            cfg.DATA_DIR = tmp_path
            with patch("src.pipeline.runner.console_info"):
                rebuild()

        for f in to_touch:
            assert not f.exists()
        assert not (tmp_path / "cache" / "embeddings").exists()
