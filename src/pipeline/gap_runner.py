"""GapRunner — выполняет gap‑анализ и генерирует персональные рекомендации."""

import json
import os
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from src import Err, GapAnalysisError, Ok, Result
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.profile_evaluator import ProfileEvaluator
from src.analyzers.skills.profession_taxonomy import ProfessionTaxonomy
from src.analyzers.skills.skill_level_analyzer import SkillLevelAnalyzer
from src.models.data_contracts import PipelineContext
from src.models.enums import ComparisonLevel, ExperienceLevel
from src.predictors.recommendation_engine import RecommendationEngine

logger = structlog.get_logger("gap_runner")

GAP_PROGRESS_FILE = Path(__file__).parent.parent.parent / "data" / "cache" / "gap_progress.json"


class GapRunner:
    def __init__(self, profiles: dict, ctx: PipelineContext | dict, args):
        self.profiles = profiles
        self.ctx = PipelineContext(**ctx) if isinstance(ctx, dict) else ctx
        self.args = args
        self.evaluator = None
        self.recommendation_engine = None
        self.taxonomy = ProfessionTaxonomy()
        self._profile_names = list(profiles.keys())
        self._total_steps = len(self._profile_names) * 2
        self._steps_done = 0

    def _write_progress(self, pct: float, message: str, stage: str = ""):
        try:
            GAP_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(GAP_PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({"pct": round(pct, 1), "message": message, "stage": stage}, f, ensure_ascii=False)
        except Exception:
            pass

    def _clear_progress(self):
        try:
            if GAP_PROGRESS_FILE.exists():
                GAP_PROGRESS_FILE.unlink()
        except Exception:
            pass

    def run(self) -> Result[tuple[dict, dict], GapAnalysisError]:
        skill_weights = self.ctx.hybrid_weights or self.ctx.skill_freq
        if not self.ctx.skill_freq or not self.ctx.vacancies_skills or not skill_weights:
            logger.warning("gap_runner_skipped", reason="missing required data")
            return Err(GapAnalysisError(message="Недостаточно данных для gap-анализа"))

        try:
            level_analyzer = SkillLevelAnalyzer()
            level_analyzer.analyze_vacancies(self.ctx.level_vacancies_data)

            skill_weights_by_level = {}
            for level in ExperienceLevel:
                skill_weights_by_level[level] = level_analyzer.get_weights_for_level(skill_weights, level)

            self.evaluator = ProfileEvaluator(
                skill_weights=skill_weights,
                vacancies_skills=self.ctx.vacancies_skills,
                vacancies_skills_dict=self.ctx.level_vacancies_data,
                hybrid_weights=skill_weights,
                skill_weights_by_level=skill_weights_by_level,
            )

            self.recommendation_engine = RecommendationEngine(
                use_ltr=True,
                use_llm=self.args.use_llm,
                profile_evaluator=self.evaluator,
                trend_analyzer=self.ctx.trend_analyzer,
            )
            self.recommendation_engine.comparator = CompetencyComparator(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                use_embeddings=True,
                level=ComparisonLevel.MIDDLE,
                similarity_threshold=0.80,
            )
            self.recommendation_engine.fit(self.ctx.vacancies_skills, skill_weights=skill_weights)

            evaluations_new = self._evaluate_profiles_parallel()
            self._print_summary(evaluations_new)

            all_recommendations = self._generate_recommendations(evaluations_new)

            self._clear_progress()
            logger.info(
                "gap_analysis_pipeline_complete",
                profiles_evaluated=len(evaluations_new),
                total_recommendations=sum(len(r.get("recommendations", [])) for r in all_recommendations.values()),
            )
            return Ok((evaluations_new, all_recommendations))
        except Exception as e:
            self._clear_progress()
            logger.exception("gap_analysis_failed", error=str(e))
            return Err(GapAnalysisError(message=f"Gap-анализ не выполнен: {e}"))

    def _update_progress(self, delta: int = 1):
        self._steps_done += delta
        pct = min(100.0, (self._steps_done / self._total_steps) * 100)
        return pct

    def _evaluate_profiles(self) -> dict:
        n = len(self.profiles)
        evals = {}
        with tqdm(total=n, desc="Оценка профилей") as pbar:
            for pname, student in self.profiles.items():
                profile_config = self.taxonomy.get_profile_target(pname)
                if profile_config:
                    target_domains = profile_config.get("target_domains", [])
                    target_profession = profile_config.get("target_profession", "")
                else:
                    target_domains = []
                    target_profession = ""
                eval_result = self.evaluator.evaluate_profile(
                    student,
                    user_type="student",
                    target_domains=target_domains,
                    taxonomy=self.taxonomy,
                )
                eval_result["target_profession"] = target_profession
                eval_result["target_domains"] = target_domains
                evals[pname] = eval_result
                pbar.update(1)
                pct = self._update_progress()
                idx = len(evals)
                self._write_progress(pct, f"Оценка профилей... {idx}/{n}", "evaluation")
        return evals

    def _evaluate_profiles_parallel(self) -> dict:
        n = len(self.profiles)
        evals = {}
        max_workers = min(4, os.cpu_count() or 1)

        def _eval_one(pname: str):
            student = self.profiles[pname]
            profile_config = self.taxonomy.get_profile_target(pname)
            if profile_config:
                target_domains = profile_config.get("target_domains", [])
                target_profession = profile_config.get("target_profession", "")
            else:
                target_domains = []
                target_profession = ""
            eval_result = self.evaluator.evaluate_profile(
                student,
                user_type="student",
                target_domains=target_domains,
                taxonomy=self.taxonomy,
            )
            eval_result["target_profession"] = target_profession
            eval_result["target_domains"] = target_domains
            return pname, eval_result

        with tqdm(total=n, desc="Оценка профилей") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_eval_one, pname): pname for pname in self._profile_names}
                for future in as_completed(futures):
                    pname, result = future.result()
                    evals[pname] = result
                    pbar.update(1)
                    pct = self._update_progress()
                    self._write_progress(pct, f"Оценка профилей... {len(evals)}/{n}", "evaluation")
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
        n = len(evaluations)
        recs = {}
        rec_engine = self.recommendation_engine
        if rec_engine.ltr_engine is None or not rec_engine.ltr_engine.is_fitted:
            self._console_info("LTR-модель не загружена. Рекомендации будут построены только на основе анализа рынка.")
            logger.warning("ltr_model_unavailable_recommendations_without_ml")

        with tqdm(total=n, desc="Генерация рекомендаций") as pbar:
            for idx, (pname, student) in enumerate(self.profiles.items(), 1):
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
                pbar.update(1)
                pct = self._update_progress()
                self._write_progress(pct, f"Генерация рекомендаций... {idx}/{n}", "recommendations")
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
