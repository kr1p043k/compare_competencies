import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src import Ok, Err, GapAnalysisError
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.pipeline.gap_runner import GapRunner, GAP_PROGRESS_FILE
from src.predictors.models import RecommendationResult, RecommendationSummary


@pytest.fixture
def mock_profiles():
    return {
        "john": {"name": "John", "skills": ["python", "django"]},
        "jane": {"name": "Jane", "skills": ["python", "sql"]},
    }


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.hybrid_weights = {"python": 0.8, "django": 0.6}
    ctx.skill_freq = {"python": 100, "django": 50}
    ctx.vacancies_skills = [["python", "django"], ["sql"]]
    ctx.level_vacancies_data = [MagicMock()]
    ctx.trend_analyzer = MagicMock()
    return ctx


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.use_llm = False
    return args


@pytest.fixture
def runner(mock_profiles, mock_ctx, mock_args):
    return GapRunner(mock_profiles, mock_ctx, mock_args)


class TestInit:
    def test_accepts_pipeline_context(self, mock_profiles, mock_ctx, mock_args):
        r = GapRunner(mock_profiles, mock_ctx, mock_args)
        assert r.profiles == mock_profiles
        assert r.ctx is mock_ctx
        assert r.args is mock_args
        assert r.evaluator is None
        assert r.recommendation_engine is None
        assert r._profile_names == ["john", "jane"]
        assert r._total_steps == len(mock_profiles) * 2 + 4

    def test_converts_dict_ctx(self, mock_profiles, mock_args):
        ctx_dict = {
            "hybrid_weights": {"python": 0.8},
            "skill_freq": {"python": 100},
            "vacancies_skills": [["python"]],
            "level_vacancies_data": [{"skills": ["python"], "description": "desc", "experience": "middle"}],
            "trend_analyzer": None,
        }
        r = GapRunner(mock_profiles, ctx_dict, mock_args)
        assert r.ctx.hybrid_weights == {"python": 0.8}


class TestWriteProgress:
    def test_writes_progress_file(self, runner, tmp_path):
        with patch("src.pipeline.gap_runner.GAP_PROGRESS_FILE", tmp_path / "gap_progress.json"):
            runner._write_progress(50.0, "test message", "test_stage")
        data = json.loads((tmp_path / "gap_progress.json").read_text(encoding="utf-8"))
        assert data["pct"] == 50.0
        assert data["message"] == "test message"
        assert data["stage"] == "test_stage"

    def test_handles_write_error(self, runner):
        mock_path = MagicMock()
        mock_path.parent.mkdir.side_effect = PermissionError("denied")
        with patch("src.pipeline.gap_runner.GAP_PROGRESS_FILE", mock_path):
            runner._write_progress(50, "msg")
        # Should not raise


class TestClearProgress:
    def test_removes_existing_file(self, runner, tmp_path):
        f = tmp_path / "gap_progress.json"
        f.touch()
        with patch("src.pipeline.gap_runner.GAP_PROGRESS_FILE", f):
            runner._clear_progress()
        assert not f.exists()

    def test_no_error_when_missing(self, runner, tmp_path):
        f = tmp_path / "gap_progress.json"
        with patch("src.pipeline.gap_runner.GAP_PROGRESS_FILE", f):
            runner._clear_progress()

    def test_handles_unlink_error(self, runner):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.unlink.side_effect = PermissionError("denied")
        with patch("src.pipeline.gap_runner.GAP_PROGRESS_FILE", mock_path):
            runner._clear_progress()


class TestUpdateProgress:
    def test_increments_and_clamps(self, runner):
        runner._total_steps = 4
        runner._steps_done = 0
        assert runner._update_progress(1) == 25.0
        assert runner._update_progress(2) == 75.0
        assert runner._update_progress(10) == 100.0


class TestRunEarlyExit:
    def test_returns_err_when_skill_freq_empty(self, runner):
        runner.ctx.skill_freq = {}
        result = runner.run()
        assert isinstance(result, Err)

    def test_returns_err_when_vacancies_skills_empty(self, runner):
        runner.ctx.vacancies_skills = []
        result = runner.run()
        assert isinstance(result, Err)

    def test_returns_err_when_skill_weights_empty(self, runner):
        runner.ctx.hybrid_weights = None
        runner.ctx.skill_freq = {"py": 1}
        runner.ctx.vacancies_skills = [["py"]]
        result = runner.run()
        assert isinstance(result, Err)


class TestRunSkillLevelError:
    def test_returns_err_on_level_weight_failure(self, runner):
        with patch(
            "src.pipeline.gap_runner.SkillLevelAnalyzer.analyze_vacancies"
        ) as mock_analyze:
            with patch(
                "src.pipeline.gap_runner.SkillLevelAnalyzer.get_weights_for_level",
                return_value=Err("bad level"),
            ):
                result = runner.run()
        assert isinstance(result, Err)
        assert "Не удалось получить веса" in result._error.message


class TestRunSuccess:
    def run_runner_with_mocks(self, runner):
        mock_evaluator = MagicMock()
        mock_engine = MagicMock()
        mock_comparator = MagicMock()

        mock_level_analyzer = MagicMock()
        mock_level_analyzer.analyze_vacancies.return_value = None
        mock_level_analyzer.get_weights_for_level.return_value = Ok({"python": 0.8})

        mock_eval_result = {
            "target_profession": "Python Developer",
            "target_domains": ["backend"],
            "profession_coverage": 75.0,
            "skill_coverage": 80.0,
            "domain_coverage_score": 85.0,
            "market_skill_coverage": 70.0,
            "readiness_score": 65.0,
            "domain_skill_count": 10,
            "market_coverage_score": 72.0,
            "domain_coverage": {"backend": {"score": 85.0}},
            "cluster_context": {"skills": {"python": 0.9}},
            "skill_metrics": {"python": {"cluster_relevance": 0.8}, "django": {"cluster_relevance": 0.6}},
        }
        mock_evaluator.evaluate_profile.return_value = Ok(mock_eval_result)

        rec_summary = RecommendationSummary()
        rec_summary.market_coverage_score = 72.0
        rec_summary.skill_coverage = 80.0
        rec_summary.domain_coverage_score = 85.0
        rec_summary.profession_coverage = 75.0

        rec_result = RecommendationResult(summary=rec_summary)
        rec_result.domain_coverage = {"backend": {"score": 85.0}}
        rec_result.target_profession = "Python Developer"

        mock_engine.generate_recommendations.return_value = Ok(rec_result)
        mock_engine.ltr_engine = MagicMock()
        mock_engine.ltr_engine.is_fitted = True

        mock_taxonomy = MagicMock()
        mock_taxonomy.get_profile_target.return_value = {
            "target_domains": ["backend"],
            "target_profession": "Python Developer",
        }

        with patch("src.pipeline.gap_runner.SkillLevelAnalyzer", return_value=mock_level_analyzer):
            with patch("src.pipeline.gap_runner.ProfileEvaluator", return_value=mock_evaluator):
                with patch("src.pipeline.gap_runner.RecommendationEngine", return_value=mock_engine):
                    with patch("src.pipeline.gap_runner.CompetencyComparator", return_value=mock_comparator):
                        with patch.object(runner, "taxonomy", mock_taxonomy):
                            with patch.object(runner, "_write_progress"):
                                with patch.object(runner, "_print_summary"):
                                    result = runner.run()
        return result, mock_engine, mock_evaluator

    def test_returns_ok_on_success(self, runner):
        result, _, _ = self.run_runner_with_mocks(runner)
        assert isinstance(result, Ok)
        evaluations, recommendations = result._value
        assert "john" in evaluations
        assert "john" in recommendations

    def test_sets_recommendation_summary_fields(self, runner):
        result, mock_engine, _ = self.run_runner_with_mocks(runner)
        _, recommendations = result._value
        rec = recommendations["john"]
        assert rec["summary"]["market_coverage_score"] == 72.0
        assert rec["summary"]["skill_coverage"] == 80.0
        assert rec["summary"]["domain_coverage_score"] == 85.0
        assert rec["summary"]["profession_coverage"] == 75.0
        assert rec["domain_coverage"] == {"backend": {"score": 85.0}}
        assert rec["target_profession"] == "Python Developer"


class TestRunFitFailure:
    def test_returns_err_when_fit_fails(self, runner):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_profile.return_value = Ok({})

        mock_engine = MagicMock()
        mock_engine.fit.return_value = Err("fit failed")
        mock_engine.ltr_engine = None

        mock_level_analyzer = MagicMock()
        mock_level_analyzer.analyze_vacancies.return_value = None
        mock_level_analyzer.get_weights_for_level.return_value = Ok({})

        with patch("src.pipeline.gap_runner.SkillLevelAnalyzer", return_value=mock_level_analyzer):
            with patch("src.pipeline.gap_runner.ProfileEvaluator", return_value=mock_evaluator):
                with patch("src.pipeline.gap_runner.RecommendationEngine", return_value=mock_engine):
                    with patch("src.pipeline.gap_runner.CompetencyComparator"):
                        with patch.object(runner, "_write_progress"):
                            result = runner.run()
        assert isinstance(result, Err)


class TestRunException:
    def test_returns_err_on_unexpected_exception(self, runner):
        with patch.object(runner, "_update_progress", side_effect=ValueError("boom")):
            result = runner.run()
        assert isinstance(result, Err)
        assert "Gap-анализ не выполнен" in result._error.message


class TestEvaluateProfiles:
    def test_evaluates_all_profiles(self, runner):
        mock_evaluator = MagicMock()
        mock_eval_result = {
            "profession_coverage": 75.0,
            "skill_coverage": 80.0,
            "domain_coverage_score": 85.0,
            "market_skill_coverage": 70.0,
            "readiness_score": 65.0,
            "domain_skill_count": 10,
            "skill_metrics": {},
        }
        mock_evaluator.evaluate_profile.return_value = Ok(mock_eval_result)
        mock_taxonomy = MagicMock()
        mock_taxonomy.get_profile_target.return_value = {
            "target_domains": ["backend"],
            "target_profession": "Python Developer",
        }
        runner.evaluator = mock_evaluator
        with patch.object(runner, "taxonomy", mock_taxonomy):
            with patch.object(runner, "_write_progress"):
                result = runner._evaluate_profiles()
        assert len(result) == 2
        assert result["john"]["target_profession"] == "Python Developer"
        assert result["john"]["target_domains"] == ["backend"]

    def test_evaluates_profile_without_config(self, runner):
        mock_evaluator = MagicMock()
        mock_eval_result = {
            "profession_coverage": 50.0,
            "skill_coverage": 60.0,
            "domain_coverage_score": 70.0,
            "market_skill_coverage": 50.0,
            "readiness_score": 40.0,
            "domain_skill_count": 5,
            "skill_metrics": {},
        }
        mock_evaluator.evaluate_profile.return_value = Ok(mock_eval_result)
        mock_taxonomy = MagicMock()
        mock_taxonomy.get_profile_target.return_value = None
        runner.evaluator = mock_evaluator
        with patch.object(runner, "taxonomy", mock_taxonomy):
            with patch.object(runner, "_write_progress"):
                result = runner._evaluate_profiles()
        assert result["john"]["target_profession"] == ""
        assert result["john"]["target_domains"] == []

    def test_skips_failed_evaluation(self, runner):
        mock_evaluator = MagicMock()
        call_count = 0

        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Err("failed")
            return Ok({
                "profession_coverage": 75.0,
                "skill_coverage": 80.0,
                "domain_coverage_score": 85.0,
                "market_skill_coverage": 70.0,
                "readiness_score": 65.0,
                "domain_skill_count": 10,
                "skill_metrics": {},
            })

        mock_evaluator.evaluate_profile.side_effect = side_effect
        runner.evaluator = mock_evaluator
        with patch.object(runner, "_write_progress"):
            result = runner._evaluate_profiles()
        assert "john" not in result
        assert "jane" in result


class TestEvaluateProfilesParallel:
    def test_evaluates_all_in_parallel(self, runner):
        mock_evaluator = MagicMock()
        mock_eval_result = {
            "profession_coverage": 75.0,
            "skill_coverage": 80.0,
            "domain_coverage_score": 85.0,
            "market_skill_coverage": 70.0,
            "readiness_score": 65.0,
            "domain_skill_count": 10,
            "skill_metrics": {},
        }
        mock_evaluator.evaluate_profile.return_value = Ok(mock_eval_result)
        mock_taxonomy = MagicMock()
        mock_taxonomy.get_profile_target.return_value = {
            "target_domains": ["backend"],
            "target_profession": "Python Developer",
        }
        runner.evaluator = mock_evaluator
        with patch.object(runner, "taxonomy", mock_taxonomy):
            with patch.object(runner, "_write_progress"):
                result = runner._evaluate_profiles_parallel()
        assert len(result) == 2

    def test_handles_failed_evaluation_in_parallel(self, runner):
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_profile.return_value = Err("failed")
        mock_taxonomy = MagicMock()
        mock_taxonomy.get_profile_target.return_value = None
        runner.evaluator = mock_evaluator
        with patch.object(runner, "taxonomy", mock_taxonomy):
            with patch.object(runner, "_write_progress"):
                result = runner._evaluate_profiles_parallel()
        assert len(result) == 2
        assert result["john"] is None
        assert result["jane"] is None


class TestPrintSummary:
    def test_prints_all_metrics(self, runner, capsys):
        evaluations = {
            "john": {
                "target_profession": "Python Developer",
                "profession_coverage": 75.0,
                "skill_coverage": 80.0,
                "domain_coverage_score": 85.0,
                "market_skill_coverage": 70.0,
                "readiness_score": 65.0,
                "domain_skill_count": 10,
            }
        }
        runner._print_summary(evaluations)
        out = capsys.readouterr().out
        assert "СВОДКА МЕТРИК" in out
        assert "JOHN" in out
        assert "Python Developer" in out
        assert "75.0%" in out
        assert "80.0%" in out
        assert "85.0%" in out
        assert "70.0%" in out
        assert "65.0%" in out

    def test_prints_with_default_target(self, runner, capsys):
        evaluations = {
            "jane": {
                "skill_coverage": 50.0,
                "domain_coverage_score": 60.0,
                "market_skill_coverage": 40.0,
                "readiness_score": 30.0,
                "domain_skill_count": 5,
                "profession_coverage": 0.0,
            }
        }
        runner._print_summary(evaluations)
        out = capsys.readouterr().out
        assert "JANE" in out
        assert "не задана" in out


class TestGenerateRecommendations:
    def make_rec_result(self):
        rec_summary = RecommendationSummary()
        rec_summary.market_coverage_score = 72.0
        rec_summary.skill_coverage = 80.0
        rec_summary.domain_coverage_score = 85.0
        rec_summary.profession_coverage = 75.0
        rec_result = RecommendationResult(summary=rec_summary)
        rec_result.domain_coverage = {"backend": {"score": 85.0}}
        rec_result.target_profession = "Python Developer"
        return rec_result

    def test_generates_with_ltr(self, runner):
        evaluations = {
            "john": {
                "target_profession": "Python Developer",
                "target_domains": ["backend"],
                "profession_coverage": 75.0,
                "skill_coverage": 80.0,
                "domain_coverage_score": 85.0,
                "market_skill_coverage": 70.0,
                "readiness_score": 65.0,
                "market_coverage_score": 72.0,
                "domain_coverage": {"backend": {"score": 85.0}},
                "cluster_context": {"skills": {"python": 0.9}},
                "skill_metrics": {"python": {"cluster_relevance": 0.8}},
            }
        }
        rec_result = self.make_rec_result()
        mock_engine = MagicMock()
        mock_engine.ltr_engine = MagicMock()
        mock_engine.ltr_engine.is_fitted = True
        mock_engine.generate_recommendations.return_value = Ok(rec_result)
        runner.recommendation_engine = mock_engine
        with patch.object(runner, "_write_progress"):
            result = runner._generate_recommendations(evaluations)
        assert "john" in result

    def test_generates_without_ltr(self, runner):
        evaluations = {
            "john": {
                "market_coverage_score": 72.0,
                "skill_coverage": 80.0,
                "domain_coverage_score": 85.0,
                "domain_coverage": {"backend": {"score": 85.0}},
                "target_profession": "Python Developer",
                "profession_coverage": 75.0,
                "cluster_context": {},
                "skill_metrics": {},
            }
        }
        rec_result = self.make_rec_result()
        mock_engine = MagicMock()
        mock_engine.ltr_engine = None
        mock_engine.generate_recommendations.return_value = Ok(rec_result)
        runner.recommendation_engine = mock_engine
        with patch.object(runner, "_write_progress"):
            result = runner._generate_recommendations(evaluations)
        assert "john" in result

    def test_skips_missing_evaluation(self, runner):
        evaluations = {"john": None}
        runner.recommendation_engine = MagicMock()
        with patch.object(runner, "_write_progress"):
            result = runner._generate_recommendations(evaluations)
        assert result == {}

    def test_handles_recommendation_error(self, runner):
        evaluations = {
            "john": {
                "market_coverage_score": 72.0,
                "skill_coverage": 80.0,
                "domain_coverage_score": 85.0,
                "domain_coverage": {"backend": {"score": 85.0}},
                "target_profession": "Python Developer",
                "profession_coverage": 75.0,
                "cluster_context": {},
                "skill_metrics": {},
            }
        }
        mock_engine = MagicMock()
        mock_engine.ltr_engine = MagicMock()
        mock_engine.ltr_engine.is_fitted = True
        mock_engine.generate_recommendations.return_value = Err("rec error")
        runner.recommendation_engine = mock_engine
        with patch.object(runner, "_write_progress"):
            result = runner._generate_recommendations(evaluations)
        assert result == {}


class TestBuildSkillContext:
    def test_uses_cluster_skills_when_available(self):
        ctx = GapRunner._build_skill_context(None, {
            "cluster_context": {"skills": {"python": 0.9}},
            "skill_metrics": {"python": {"cluster_relevance": 0.8}, "django": {"cluster_relevance": 0.6}},
        })
        assert ctx["python"] == 0.9
        assert ctx["django"] == 0.6

    def test_falls_back_to_default_relevance(self):
        ctx = GapRunner._build_skill_context(None, {
            "cluster_context": {},
            "skill_metrics": {"python": {"cluster_relevance": 0.8}},
        })
        assert ctx["python"] == 0.8

    def test_empty_skill_metrics(self):
        ctx = GapRunner._build_skill_context(None, {
            "cluster_context": {},
            "skill_metrics": {},
        })
        assert ctx == {}

    def test_missing_cluster_context(self):
        ctx = GapRunner._build_skill_context(None, {
            "skill_metrics": {"python": {"cluster_relevance": 0.8}},
        })
        assert ctx["python"] == 0.8
