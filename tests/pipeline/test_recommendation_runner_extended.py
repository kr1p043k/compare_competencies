from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from src import Ok, Err
from src.models.data_contracts import PipelineContext


@pytest.fixture
def profiles():
    return {
        "student1": MagicMock(),
        "student2": MagicMock(),
    }


@pytest.fixture
def ctx():
    return PipelineContext(
        hybrid_weights={"python": 0.5},
        vacancies_skills=[["python"], ["sql"]],
        trend_analyzer=MagicMock(),
    )


@pytest.fixture
def args():
    mock = MagicMock()
    mock.use_llm = False
    return mock


@pytest.fixture
def evaluator():
    return MagicMock()


class TestRecommendationRunnerInit:
    def test_init_with_ctx_dict(self, profiles, args):
        ctx_dict = {
            "hybrid_weights": {"py": 0.5},
            "vacancies_skills": [["py"]],
            "trend_analyzer": None,
        }
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx_dict, args)
        assert isinstance(runner.ctx, PipelineContext)
        assert runner.ctx.hybrid_weights == {"py": 0.5}

    def test_init_with_pipeline_context(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        assert runner.ctx is ctx


class TestRecommendationRunnerInitializeEngine:
    def test_initialize_engine_ok(self, profiles, ctx, args, evaluator):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            engine_instance = MagicMock()
            MockEngine.return_value = engine_instance
            engine_instance.fit.return_value = Ok(engine_instance)
            with patch("src.pipeline.recommendation_runner.CompetencyComparator"):
                result = runner.initialize_engine(evaluator)
        assert result.is_ok()
        assert runner.engine is engine_instance

    def test_initialize_engine_fit_err(self, profiles, ctx, args, evaluator):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        with patch("src.pipeline.recommendation_runner.RecommendationEngine") as MockEngine:
            engine_instance = MagicMock()
            MockEngine.return_value = engine_instance
            engine_instance.fit.return_value = Err("fit failed")
            with patch("src.pipeline.recommendation_runner.CompetencyComparator"):
                result = runner.initialize_engine(evaluator)
        assert result.is_err()

    def test_initialize_engine_exception(self, profiles, ctx, args, evaluator):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        with patch("src.pipeline.recommendation_runner.RecommendationEngine",
                   side_effect=ValueError("init error")):
            result = runner.initialize_engine(evaluator)
        assert result.is_err()


class TestRecommendationRunnerRun:
    def test_run_engine_none(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        runner.engine = None
        result = runner.run({})
        assert result.is_err()

    def test_run_ok(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        runner.engine = MagicMock()
        mock_rec_result = MagicMock()
        mock_rec_result.summary.market_coverage_score = 0.8
        mock_rec_result.summary.skill_coverage = 0.7
        mock_rec_result.summary.domain_coverage_score = 0.6
        mock_rec_result.domain_coverage = {"AI": 0.5}
        mock_rec_result.model_dump.return_value = {"summary": {"market_coverage_score": 0.8}}
        runner.engine.generate_recommendations.return_value = Ok(mock_rec_result)

        evaluations = {
            "student1": {
                "market_coverage_score": 0.8,
                "skill_coverage": 0.7,
                "domain_coverage_score": 0.6,
                "domain_coverage": {"AI": 0.5},
                "skill_metrics": {},
                "cluster_context": {"skills": {}},
            }
        }
        result = runner.run(evaluations)
        assert result.is_ok()
        data = result.unwrap()
        assert "student1" in data

    def test_run_skips_missing_evaluation(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        runner.engine = MagicMock()
        evaluations = {}
        result = runner.run(evaluations)
        assert result.is_ok()
        assert result.unwrap() == {}

    def test_run_recommendation_err(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        runner.engine = MagicMock()
        runner.engine.generate_recommendations.return_value = Err("gen failed")
        evaluations = {
            "student1": {
                "market_coverage_score": 0.8,
                "skill_coverage": 0.7,
                "domain_coverage_score": 0.6,
                "skill_metrics": {"py": {"cluster_relevance": 0.3}},
                "cluster_context": {"skills": {"py": {"relevance": 0.9}}},
            }
        }
        result = runner.run(evaluations)
        assert result.is_ok()
        assert result.unwrap() == {}


class TestRecommendationRunnerBuildSkillContext:
    def test_build_skill_context_with_cluster(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        eval_result = {
            "skill_metrics": {
                "python": {"cluster_relevance": 0.3},
                "sql": {"cluster_relevance": 0.2},
            },
            "cluster_context": {
                "skills": {
                    "python": {"relevance": 0.95},
                }
            },
        }
        ctx_result = runner._build_skill_context(eval_result)
        assert ctx_result["python"] == {"relevance": 0.95}
        assert ctx_result["sql"] == 0.2

    def test_build_skill_context_no_cluster(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        eval_result = {
            "skill_metrics": {
                "python": {"cluster_relevance": 0.3},
            },
        }
        ctx_result = runner._build_skill_context(eval_result)
        assert ctx_result["python"] == 0.3

    def test_build_skill_context_empty(self, profiles, ctx, args):
        from src.pipeline.recommendation_runner import RecommendationRunner
        runner = RecommendationRunner(profiles, ctx, args)
        result = runner._build_skill_context({})
        assert result == {}
