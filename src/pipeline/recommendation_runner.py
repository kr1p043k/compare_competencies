"""RecommendationRunner — генерирует персональные рекомендации."""

import structlog
from tqdm import tqdm

from src.analyzers.comparison.comparator import CompetencyComparator
from src.models.enums import ComparisonLevel
from src.predictors.recommendation_engine import RecommendationEngine

logger = structlog.get_logger("recommendation_runner")


class RecommendationRunner:
    """Инициализирует движок и генерирует рекомендации."""

    def __init__(self, profiles: dict, data: dict, args):
        self.profiles = profiles
        self.data = data
        self.args = args
        self.engine = None

    def initialize_engine(self, evaluator):
        hybrid_weights = self.data.get("hybrid_weights", {})
        self.engine = RecommendationEngine(
            use_ltr=True,
            use_llm=self.args.use_llm,
            profile_evaluator=evaluator,
            trend_analyzer=self.data.get("trend_analyzer"),
        )
        self.engine.comparator = CompetencyComparator(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            use_embeddings=True,
            level=ComparisonLevel.MIDDLE,
            similarity_threshold=0.80,
        )
        self.engine.fit(self.data.get("vacancies_skills", []), skill_weights=hybrid_weights)

    def run(self, evaluations: dict) -> dict:
        if self.engine is None:
            raise RuntimeError("Сначала вызовите initialize_engine()")
        recs = {}
        for pname, student in tqdm(self.profiles.items(), desc="Генерация рекомендаций"):
            try:
                v2_result = evaluations[pname]
                skill_ctx = self._build_skill_context(v2_result)
                self.engine.set_cluster_context(skill_ctx)
                full_rec = self.engine.generate_recommendations(student, user_type="student")
                if full_rec is None:
                    continue
                full_rec["summary"]["market_coverage_score"] = v2_result["market_coverage_score"]
                full_rec["summary"]["skill_coverage"] = v2_result["skill_coverage"]
                full_rec["summary"]["domain_coverage_score"] = v2_result["domain_coverage_score"]
                full_rec["domain_coverage"] = v2_result.get("domain_coverage", {})
                recs[pname] = full_rec
            except Exception as e:
                logger.error("recommendation_generation_failed", profile=pname, error=str(e))
        return recs

    def _build_skill_context(self, eval_result):
        ctx = {}
        cluster_ctx = eval_result.get("cluster_context") or {}
        cluster_skills = cluster_ctx.get("skills", {})
        for skill, metric in eval_result.get("skill_metrics", {}).items():
            if skill in cluster_skills:
                ctx[skill] = cluster_skills[skill]
            else:
                ctx[skill] = metric.get("cluster_relevance", 0.15)
        return ctx
