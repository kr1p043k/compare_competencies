"""GapRunner — выполняет gap‑анализ и генерирует персональные рекомендации."""

import structlog
from tqdm import tqdm

from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.predictors.recommendation_engine import RecommendationEngine

logger = structlog.get_logger("gap_runner")


class GapRunner:
    def __init__(self, profiles: dict, data: dict, args):
        self.profiles = profiles
        self.data = data
        self.args = args
        self.evaluator = None
        self.recommendation_engine = None
        self.taxonomy = ProfessionTaxonomy()

    def run(self) -> tuple[dict, dict]:
        skill_weights = self.data["hybrid_weights"] or self.data["skill_freq"]
        if not self.data["skill_freq"] or not self.data["vacancies_skills"] or not skill_weights:
            logger.warning("gap_runner_skipped", reason="missing required data")
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

        logger.info(
            "gap_analysis_pipeline_complete",
            profiles_evaluated=len(evaluations_new),
            total_recommendations=sum(len(r.get("recommendations", [])) for r in all_recommendations.values()),
        )
        return evaluations_new, all_recommendations

    def _evaluate_profiles(self) -> dict:
        evals = {}
        with tqdm(total=len(self.profiles), desc="Оценка профилей") as pbar:
            for pname, student in self.profiles.items():
                profile_config = self.taxonomy.get_profile_target(pname)
                if profile_config:
                    target_domains = profile_config.get("target_domains", [])
                    target_profession = profile_config.get("target_profession", "")
                    logger.info(
                        "profile_target_set",
                        profile=pname,
                        profession=target_profession,
                        domains=target_domains,
                    )
                else:
                    target_domains = []
                    target_profession = ""
                logger.info(
                    "profile_evaluation_start",
                    profile=pname,
                    target_profession=target_profession,
                    target_domains=target_domains,
                    skills_count=len(student.skills),
                )
                eval_result = self.evaluator.evaluate_profile(
                    student,
                    user_type="student",
                    target_domains=target_domains,
                    taxonomy=self.taxonomy,
                )
                logger.info(
                    "profile_evaluation_complete",
                    profile=pname,
                    profession_coverage=eval_result.get("profession_coverage", 0),
                    readiness_score=eval_result.get("readiness_score", 0),
                    krm_codes=len(eval_result.get("krm_coverage", {})),
                )
                eval_result["target_profession"] = target_profession
                eval_result["target_domains"] = target_domains
                evals[pname] = eval_result
                pbar.update(1)
        return evals

    def _print_summary(self, evaluations):
        print(f"\n{'=' * 70}")
        print("  СВОДКА МЕТРИК ПО ПРОФИЛЯМ")
        print(f"{'=' * 70}")
        for pname, ev in evaluations.items():
            target = ev.get("target_profession", "не задана")
            print(f"\n  📊 {pname.upper()} (цель: {target}):")
            print(f"     Покрытие по профессии:    {ev.get('profession_coverage', 0):.1f}%")
            print(f"     Навыковое покрытие:       {ev['skill_coverage']:.1f}%")
            print(f"     Доменное покрытие:        {ev['domain_coverage_score']:.1f}%")
            print(f"     Реальное покрытие рынка:  {ev['market_skill_coverage']:.1f}%")
            print(f"     Готовность к уровню:      {ev['readiness_score']:.1f}%")
            print(f"     Всего навыков в домене:   {ev.get('domain_skill_count', 0)}")

    def _generate_recommendations(self, evaluations):
        recs = {}
        rec_engine = self.recommendation_engine
        if rec_engine.ltr_engine is None or not rec_engine.ltr_engine.is_fitted:
            self._console_info("LTR-модель не загружена. Рекомендации будут построены только на основе анализа рынка.")
            logger.warning("ltr_model_unavailable_recommendations_without_ml")

        for pname, student in tqdm(self.profiles.items(), desc="Генерация рекомендаций"):
            try:
                v2_result = evaluations[pname]
                skill_weights_context = self._build_skill_context(v2_result)
                rec_engine.set_cluster_context(skill_weights_context)
                full_rec = rec_engine.generate_recommendations(
                    student,
                    user_type="student",
                    precomputed_eval=v2_result,
                )
                if full_rec is None:
                    continue
                full_rec["summary"]["market_coverage_score"] = v2_result["market_coverage_score"]
                full_rec["summary"]["skill_coverage"] = v2_result["skill_coverage"]
                full_rec["summary"]["domain_coverage_score"] = v2_result["domain_coverage_score"]
                full_rec["summary"]["profession_coverage"] = v2_result.get("profession_coverage", 0)
                full_rec["domain_coverage"] = v2_result.get("domain_coverage", {})
                full_rec["target_profession"] = v2_result.get("target_profession", "")
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
