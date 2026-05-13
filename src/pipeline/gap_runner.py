"""
GapRunner — выполняет gap‑анализ и генерирует персональные рекомендации.
"""

import structlog
from tqdm import tqdm

from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.predictors.recommendation_engine import RecommendationEngine

logger = structlog.get_logger("gap_runner")


class GapRunner:
    """Выполняет gap‑анализ и генерирует рекомендации."""

    def __init__(self, profiles: dict, data: dict, args):
        self.profiles = profiles
        self.data = data
        self.args = args
        self.evaluator = None
        self.recommendation_engine = None

    def run(self) -> tuple[dict, dict]:
        """
        Возвращает (evaluations_new, all_recommendations).
        """
        skill_weights = self.data["hybrid_weights"]
        if not self.data["skill_freq"] or not self.data["vacancies_skills"] or not skill_weights:
            return {}, {}

        level_analyzer = SkillLevelAnalyzer()
        level_analyzer.analyze_vacancies(self.data["level_vacancies_data"])

        skill_weights_by_level = {}
        for level in ExperienceLevel:
            skill_weights_by_level[level] = level_analyzer.get_weights_for_level(skill_weights, level)

        self.evaluator = ProfileEvaluator(
            skill_weights=skill_weights,
            vacancies_skills=self.data["vacancies_skills"],
            vacancies_skills_dict=self.data["level_vacancies_data"],
            hybrid_weights=skill_weights,
            skill_weights_by_level=skill_weights_by_level,
        )

        self.recommendation_engine = RecommendationEngine(
            use_ltr=True,
            use_llm=self.args.use_llm,
            profile_evaluator=self.evaluator,
            trend_analyzer=self.data["trend_analyzer"],
        )
        self.recommendation_engine.comparator = CompetencyComparator(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            use_embeddings=True,
            level=ComparisonLevel.MIDDLE,
            similarity_threshold=0.80,
        )
        self.recommendation_engine.fit(self.data["vacancies_skills"], skill_weights=skill_weights)

        evaluations_new = self._evaluate_profiles()
        self._print_summary(evaluations_new)

        all_recommendations = self._generate_recommendations(evaluations_new)

        return evaluations_new, all_recommendations

    def _evaluate_profiles(self) -> dict:
        evals = {}
        with tqdm(total=len(self.profiles), desc="Оценка профилей") as pbar:
            for pname, student in self.profiles.items():
                evals[pname] = self.evaluator.evaluate_profile(student, user_type="student")
                pbar.update(1)
        return evals

    def _print_summary(self, evaluations):
        print(f"\n{'=' * 70}")
        print("  СВОДКА МЕТРИК ПО ПРОФИЛЯМ")
        print(f"{'=' * 70}")
        for pname, ev in evaluations.items():
            print(f"\n  📊 {pname.upper()} (целевой уровень: {self.profiles[pname].target_level}):")
            print(f"     Общее покрытие рынка: {ev['market_coverage_score']:.1f}%")
            print(f"     Навыковое покрытие:   {ev['skill_coverage']:.1f}%")
            print(f"     Доменное покрытие:    {ev['domain_coverage_score']:.1f}%")
            print(f"     Реальное покрытие:    {ev['market_skill_coverage']:.1f}%")
            print(f"     Готовность к уровню:  {ev['readiness_score']:.1f}%")

    def _generate_recommendations(self, evaluations):
        recs = {}
        rec_engine = self.recommendation_engine
        if rec_engine.ltr_engine is None or not rec_engine.ltr_engine.is_fitted:
            self._console_info("⚠️ LTR-модель не загружена.Рекомендации будут построены только на основе анализа рынка.")
            logger.warning("ltr_model_unavailable_recommendations_without_ml")

        for pname, student in tqdm(self.profiles.items(), desc="Генерация рекомендаций"):
            try:
                v2_result = evaluations[pname]
                skill_weights_context = self._build_skill_context(v2_result)
                rec_engine.set_cluster_context(skill_weights_context)
                full_rec = rec_engine.generate_recommendations(student, user_type="student")
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

    def _console_info(self, msg: str):
        print(f"  {msg}")
