# src/predictors/recommendation_engine.py
import json
import re
import time
from typing import Any

import numpy as np
import requests
import structlog
from sklearn.preprocessing import MinMaxScaler

from src import Err, Ok, RecommendationError, Result, config
from src.errors import DomainError
from src.analyzers.comparison.comparator import CompetencyComparator
from src.analyzers.gap.gap_analyzer import GapAnalyzer
from src.analyzers.skills.skill_filter import SkillFilter
from src.analyzers.skills.skill_taxonomy import SkillTaxonomy
from src.models.enums import PriorityLevel, SkillCategory, TrendType
from src.models.student import StudentProfile
from src.predictors.base import RecommenderPredictor
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine
from src.predictors.reranker import BaseReranker, CrossEncoderReranker, RerankerBuilder
from src.predictors.models import (
    ClosestRole,
    Recommendation,
    RecommendationResult,
    RecommendationSummary,
)

logger = structlog.get_logger(__name__)


class RecommendationEngine(RecommenderPredictor["RecommendationEngine", RecommendationResult]):
    """
    Движок рекомендаций с профильной оценкой, LTR-ранжированием
    и генерацией естественноязыковых объяснений.
    """

    def __init__(
        self,
        use_ltr: bool = True,
        use_llm: bool = False,
        use_reranker: bool = False,
        profile_evaluator=None,
        trend_analyzer=None,
    ):
        self.comparator = CompetencyComparator(use_embeddings=True, level="middle")
        self.gap_analyzer: GapAnalyzer | None = None
        self.skill_filter = SkillFilter()
        self.is_fitted = False
        self.profile_evaluator = profile_evaluator
        self.trend_analyzer = trend_analyzer

        self.use_llm = use_llm and bool(config.YC_API_KEY and config.YC_FOLDER_ID)
        self.client = None
        self.explanation_model = None
        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.YC_API_KEY, base_url="https://llm.api.cloud.yandex.net/v1")
                self.explanation_model = config.YANDEXGPT_MODEL or "yandexgpt-lite"
            except Exception as exc:
                logger.warning("llm_client_init_failed", error=str(exc))
                self.use_llm = False
        self.use_ltr = use_ltr
        self.ltr_engine: LTRRecommendationEngine | None = None

        self.use_reranker = use_reranker
        self.reranker: BaseReranker | None = RerankerBuilder.build_cross_encoder() if use_reranker else None
        if use_reranker:
            logger.info("reranker_enabled")

        if use_ltr:
            self.ltr_engine = LTRRecommendationEngine()
            model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
            if model_path.exists():
                match self.ltr_engine.load_model(model_path):
                    case Ok(_):
                        logger.info("ltr_model_loaded")
                    case Err(err):
                        logger.warning("ltr_model_load_failed", error=str(err))
                        self.ltr_engine = None

        self.cluster_weights = None

        # Кешируем SkillTaxonomy
        self._taxonomy = SkillTaxonomy()

        # Загружаем список hard-навыков (если файла нет — пустой fallback)
        hard_path = config.HARD_SKILLS_PATH
        if hard_path.exists():
            self._hard_keywords = set(json.loads(hard_path.read_text(encoding="utf-8")))
        else:
            logger.warning("hard_skills_file_not_found", path=str(hard_path))
            self._hard_keywords = set()

        # Загружаем список горячих навыков для трендов
        hot_path = config.TREND_HOT_SKILLS_PATH
        if hot_path.exists():
            self._always_hot = set(json.loads(hot_path.read_text(encoding="utf-8")))
            logger.info("hot_skills_loaded", count=len(self._always_hot))
        else:
            logger.warning("hot_skills_file_not_found", path=str(hot_path))
            self._always_hot = set()

        # Загружаем группы временных рамок
        timeframe_path = config.TIMEFRAME_GROUPS_PATH
        if timeframe_path.exists():
            timeframe_data = json.loads(timeframe_path.read_text(encoding="utf-8"))
            self._timeframe_easy = set(timeframe_data.get("easy", []))
            self._timeframe_medium = set(timeframe_data.get("medium", []))
            self._timeframe_hard = set(timeframe_data.get("hard", []))
            logger.info("timeframe_groups_loaded")
        else:
            logger.warning("timeframe_groups_file_not_found", path=str(timeframe_path))
            self._timeframe_easy = {"git", "html", "css", "sass", "english", "английский язык"}
            self._timeframe_medium = {"javascript", "python", "sql", "redis", "docker", "react"}
            self._timeframe_hard = {"java", "kubernetes", "aws", "machine learning", "tensorflow"}

        self._cached_trend_bonuses: dict[str, float] | None = None
        self._trend_bonuses_cached_at: float = 0.0
        self._load_templates()
        logger.info(
            "recommendation_engine_initialized",
            ltr=self.use_ltr,
            ltr_fitted=self.ltr_engine.is_fitted if self.ltr_engine else False,
            llm=self.use_llm,
        )

    @property
    def name(self) -> str:
        return "RecommendationEngine"

    # ------------------------------------------------------------------
    # ПУБЛИЧНЫЙ API
    # ------------------------------------------------------------------

    def set_cluster_context(self, weights: dict[str, float]) -> None:
        self.cluster_weights = weights
        if weights:
            logger.info(
                "cluster_context_set",
                skills_count=len(weights),
                total_weight=round(sum(weights.values()), 2),
            )
        else:
            logger.info("cluster_context_empty")

    def clear_cluster_context(self) -> None:
        self.cluster_weights = None
        logger.info("cluster_context_cleared")

    def fit(self, vacancies_skills: list[list[str]], skill_weights: dict[str, float]) -> Result["RecommendationEngine", Exception]:
        if not vacancies_skills:
            return Err(RecommendationError(message="no_vacancy_data_for_training", profile=""))
        if not skill_weights:
            return Err(RecommendationError(message="skill_weights_required", profile=""))

        self.comparator.fit_market(vacancies_skills)
        self.gap_analyzer = GapAnalyzer(skill_weights)
        self.comparator.set_skill_weights(skill_weights)
        self.is_fitted = True
        logger.info(
            "recommendation_engine_fitted",
            vacancies=len(vacancies_skills),
            skills=len(skill_weights),
        )
        return Ok(self)

    def generate_recommendations(
        self,
        student: StudentProfile,
        user_type: str = "student",
        precomputed_eval: dict[str, Any] | None = None,
        target_domains: list[str] | None = None,
        taxonomy: Any | None = None,
    ) -> Result[RecommendationResult, RecommendationError]:
        from src.utils import atomic_write_json

        if not hasattr(self, "profile_evaluator") or self.profile_evaluator is None:
            return Err(RecommendationError(message="RecommendationEngine не инициализирован с profile_evaluator", profile=student.profile_name))

        profile_name = student.profile_name
        logger.info("generate_recommendations_started", profile=profile_name)

        try:
            if precomputed_eval is not None:
                eval_result = precomputed_eval
            else:
                match self.profile_evaluator.evaluate_profile(
                    student,
                    user_type=user_type,
                    target_domains=target_domains,
                    taxonomy=taxonomy,
                ):
                    case Ok(eval_result):
                        pass
                    case Err(e):
                        return Err(RecommendationError(message=str(e), profile=profile_name))
                    case _:
                        return Err(RecommendationError(message="Оценка профиля вернула неожиданный результат", profile=profile_name))

            cluster_context = eval_result.get("cluster_context") or {}
            closest_clusters = cluster_context.get("closest_clusters", [])
            cluster_skills_map = cluster_context.get("skills", {})
            student_set = set(s.lower() for s in student.skills)

            closest_roles = self._build_closest_roles(closest_clusters, cluster_skills_map, student_set)

            top_recs: list[tuple[str, float]] = eval_result.get("top_recommendations", [])
            evaluator_scores: dict[str, float] = {skill: score for skill, score in top_recs}

            ltr_scores: dict[str, float] = {}
            if self.ltr_engine and self.ltr_engine.is_fitted:
                all_market = list(self.ltr_engine.skill_metadata.keys())
                missing_for_ltr = [s for s in all_market if s not in student_set]
                match self.ltr_engine.predict_impact(student.skills, missing_for_ltr):
                    case Ok(ltr_impacts):
                        if ltr_impacts:
                            skills = [im.skill for im in ltr_impacts]
                            raw_scores = [im.score for im in ltr_impacts]
                            scaler = MinMaxScaler()
                            normalized_scores = scaler.fit_transform(np.array(raw_scores).reshape(-1, 1)).flatten()
                            ltr_scores = {
                                skill.lower(): float(score) for skill, score in zip(skills, normalized_scores, strict=False)
                            }
                        logger.info("ltr_scores_normalized", profile=profile_name, ltr_skills=len(ltr_scores))
                    case Err(e):
                        logger.warning("ltr_scoring_failed", error=str(e))
            else:
                logger.info("ltr_not_used_generating_without_ml", profile=profile_name)

            all_skills_to_rank = set(evaluator_scores) | set(ltr_scores)
            combined_scores: dict[str, float] = {}
            for skill in all_skills_to_rank:
                ev = evaluator_scores.get(skill, 0.0)
                ltr = ltr_scores.get(skill, 0.0)
                if ltr_scores:
                    combined_scores[skill] = config.BLEND_EVALUATOR_WEIGHT * ev + config.BLEND_LTR_WEIGHT * ltr
                else:
                    combined_scores[skill] = ev

            now = time.time()
            if (self._cached_trend_bonuses is None or now - self._trend_bonuses_cached_at > 3600) and self.trend_analyzer is not None:
                match self.trend_analyzer.get_trending_skills(top_n=500, min_change_percent=0.0):
                    case Ok(trends):
                        self.trend_analyzer.save_trends(trends)
                        tb: dict[str, float] = {}
                        for t in trends.get(TrendType.RISING, []):
                            tb[t["skill"]] = min(t["change_pct"] / 100.0, 0.3)
                        for hot in self._always_hot:
                            if hot not in tb:
                                tb[hot] = config.TREND_ALWAYS_HOT_BONUS
                        self._cached_trend_bonuses = tb
                        self._trend_bonuses_cached_at = now
                    case _:
                        pass

            trend_bonuses = self._cached_trend_bonuses or {}

            dominant_domain = max(
                eval_result.get("domain_coverage", {}).items(),
                key=lambda x: x[1].get("coverage", 0),
                default=(None, None),
            )[0]
            domain_skills: set[str] = set()
            if dominant_domain and hasattr(self.profile_evaluator, "domain_analyzer"):
                domain_skills = set(
                    s.lower() for s in self.profile_evaluator.domain_analyzer.domain_map.get(dominant_domain, [])
                )

            for skill in list(combined_scores.keys()):
                bonus = 1.0
                if skill in trend_bonuses:
                    bonus += trend_bonuses[skill]
                if skill.lower() in domain_skills:
                    bonus += config.DOMAIN_BONUS
                combined_scores[skill] *= bonus

            before_reranker = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

            if self.reranker is not None:
                query = " ".join(s.lower() for s in student.skills)
                if dominant_domain:
                    query += f" {dominant_domain}"
                documents = list(before_reranker.keys())
                match self.reranker.rerank(query, documents, top_k=len(documents)):
                    case Ok(rr):
                        raw = {s: float(sc) for s, sc in rr.top_k(len(documents))}
                        vals = list(raw.values())
                        vmin, vmax = min(vals), max(vals)
                        reranked_norm: dict[str, float] = {}
                        if vmax > vmin:
                            reranked_norm = {s: (v - vmin) / (vmax - vmin) for s, v in raw.items()}
                        else:
                            reranked_norm = {s: 0.5 for s in raw}
                        for skill in combined_scores:
                            rerank_bonus = reranked_norm.get(skill, 0.0)
                            combined_scores[skill] = 0.7 * combined_scores[skill] + 0.3 * rerank_bonus
                        logger.info("reranker_applied", profile=profile_name, skills=len(reranked_norm))
                    case Err(e):
                        logger.warning("reranker_skipped", error=str(e))

            after_reranker = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

            result_dir = config.DATA_DIR / "result" / profile_name
            result_dir.mkdir(parents=True, exist_ok=True)
            try:
                atomic_write_json(
                    {"before_reranker": before_reranker, "after_reranker": after_reranker},
                    result_dir / "reranker_comparison.json",
                )
            except Exception as e:
                logger.warning("reranker_comparison_save_failed", error=str(e))

            rec_objects: list[Recommendation] = []
            skill_metrics = eval_result.get("skill_metrics", {})
            for skill, score in combined_scores.items():
                metric = skill_metrics.get(skill, {})
                try:
                    explanation = self._generate_explanation(skill, score, eval_result)
                    if skill in trend_bonuses:
                        explanation += f" 📈 Растущий тренд (+{trend_bonuses[skill] * 100:.0f}%)."
                    if skill.lower() in domain_skills:
                        explanation += f" 🔗 Ключевой навык для домена «{dominant_domain}»."
                    is_soft = not self._is_hard_skill(skill)
                except Exception as e:
                    logger.error("explanation_failed", skill=skill, error=str(e))
                    explanation = f"Навык '{skill}' востребован на рынке."
                    is_soft = False

                rec_objects.append(Recommendation(
                    skill=skill,
                    importance_score=score,
                    priority=(
                        PriorityLevel.HIGH
                        if score > config.PRIORITY_HIGH_THRESHOLD
                        else PriorityLevel.MEDIUM
                        if score > config.PRIORITY_MEDIUM_THRESHOLD
                        else PriorityLevel.LOW
                    ),
                    category=metric.get("category", SkillCategory.MISSING),
                    why_important=explanation,
                    how_to_learn=self._get_learning_path(skill, is_soft, student),
                    expected_timeframe=self._get_timeframe(skill),
                    expected_outcome=self._get_role_outcome(skill, closest_roles),
                    is_soft_skill=is_soft,
                    market_frequency_percent=score * 100,
                ))

            rec_objects.sort(key=lambda x: x.importance_score, reverse=True)

            top_recs_obj = self._diversify_recommendations(
                [r.model_dump() for r in rec_objects], max_per_category=config.DIVERSIFY_MAX_PER_CATEGORY
            )[:15]

            rec_objects = [Recommendation(**r) for r in top_recs_obj]
            for idx, rec in enumerate(rec_objects, 1):
                rec.rank = idx

            ltr_file = config.DATA_DIR / "result" / profile_name / f"ltr_recommendations_{profile_name}.json"
            ltr_file.parent.mkdir(parents=True, exist_ok=True)
            ltr_data = {
                "profile": profile_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ltr_used": bool(ltr_scores),
                "recommendations": [
                    {"skill": r.skill, "score": round(r.importance_score * 100, 2), "explanation": r.why_important}
                    for r in rec_objects[:10]
                ],
            }
            try:
                atomic_write_json(ltr_data, ltr_file)
            except Exception as e:
                logger.warning("ltr_save_failed", profile=profile_name, error=str(e))

            logger.info(
                "generate_recommendations_completed",
                profile=profile_name,
                recs=len(rec_objects),
                ltr_contributed=bool(ltr_scores),
            )

            profession_coverage = eval_result.get("profession_coverage", 0)
            result = RecommendationResult(
                summary=RecommendationSummary(
                    match_score=eval_result.get("market_coverage_score", 0),
                    confidence=eval_result.get("readiness_score", 0),
                    market_coverage_score=eval_result.get("market_coverage_score", 0),
                    skill_coverage=eval_result.get("skill_coverage", 0),
                    domain_coverage_score=eval_result.get("domain_coverage_score", 0),
                    readiness_score=eval_result.get("readiness_score", 0),
                    profession_coverage=profession_coverage,
                    avg_gap=eval_result.get("avg_gap", 0),
                    coverage=eval_result.get("market_coverage_score", 0),
                    coverage_details={
                        "covered_skills_count": len(student_set & set(eval_result.get("skill_metrics", {}).keys())),
                        "total_market_skills": len(eval_result.get("skill_metrics", {})),
                    },
                    market_skill_coverage=eval_result.get("market_skill_coverage", 0.0),
                ),
                profession_coverage_detail=eval_result.get("profession_coverage_detail", {}),
                closest_roles=[ClosestRole(**r) for r in closest_roles],
                recommendations=rec_objects,
                domain_coverage=eval_result.get("domain_coverage", {}),
                gaps=eval_result.get("gaps", {}),
                trend_bonuses_count=len(trend_bonuses),
                dominant_domain_name=dominant_domain,
            )
            return Ok(result)

        except Exception as e:
            logger.exception("generate_recommendations_crashed", profile=profile_name, error=str(e))
            return Err(RecommendationError(message=str(e), profile=profile_name))

    # ------------------------------------------------------------------
    # ПРИВАТНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------

    def _build_closest_roles(
        self,
        closest_clusters: list[dict],
        cluster_skills_map: dict,
        student_set: set[str],
    ) -> list[dict]:
        roles = []
        for c in closest_clusters[:3]:
            name = c.get("name", f"Кластер {c['id']}")
            sim = c.get("similarity", 0)
            cluster_id = c["id"]

            cluster_all_skills: set[str] = set()
            if hasattr(self.profile_evaluator, "clusterer") and self.profile_evaluator.clusterer:
                try:
                    cluster_all_skills = set(
                        s.lower()
                        for s in self.profile_evaluator.clusterer.get_top_skills_in_cluster(cluster_id, top_n=50)
                    )
                except Exception as e:
                    logger.warning("cluster_skills_fetch_failed", cluster=cluster_id, error=str(e))
                    cluster_all_skills = set(cluster_skills_map.keys())
            else:
                cluster_all_skills = set(cluster_skills_map.keys())

            covered = len(student_set & cluster_all_skills)
            total = len(cluster_all_skills)

            roles.append(
                {
                    "role": name,
                    "semantic_similarity": round(sim * 100, 1),
                    "similarity_explanation": (
                        f"Ваш профиль семантически близок к этой роли на {sim * 100:.0f}%. "
                        f"Это означает, что ваш набор навыков похож на требования вакансий "
                        f"в этом кластере, но не гарантирует полного соответствия."
                    ),
                    "skills_covered": f"{covered}/{total}",
                    "coverage_percent": round(covered / total * 100, 1) if total > 0 else 0,
                    "coverage_explanation": (
                        f"Вы уже знаете {covered} из {total} ключевых навыков этой роли "
                        f"({round(covered / total * 100, 1) if total > 0 else 0}%). "
                        f"Рекомендации ниже помогут закрыть пробелы."
                    ),
                }
            )
        return roles

    def _diversify_recommendations(self, recs: list[dict], max_per_category: int = 3) -> list[dict]:
        seen: dict[str, int] = {}
        priority: list[dict] = []
        leftover: list[dict] = []
        for rec in recs:
            cat = rec.get("category", "other")
            if seen.get(cat, 0) < max_per_category:
                seen[cat] = seen.get(cat, 0) + 1
                priority.append(rec)
            else:
                leftover.append(rec)
        return priority + leftover

    def _empty_recommendations(self) -> dict[str, Any]:
        return {
            "summary": {
                "match_score": 0,
                "confidence": 0,
                "market_coverage_score": 0,
                "skill_coverage": 0,
                "domain_coverage_score": 0,
                "readiness_score": 0,
                "avg_gap": 0,
                "coverage": 0,
                "coverage_details": {
                    "covered_skills_count": 0,
                    "total_market_skills": 0,
                },
                "market_skill_coverage": 0,
            },
            "closest_roles": [],
            "recommendations": [],
            "domain_coverage": {},
            "gaps": {},
        }

    def _get_role_outcome(self, skill: str, closest_roles: list[dict]) -> str:
        if not closest_roles:
            return f"Освоение '{skill}' расширит ваш технический кругозор."
        top_role = closest_roles[0]
        role_name = top_role["role"]
        similarity = top_role["semantic_similarity"]
        coverage = top_role["coverage_percent"]
        total_skills = int(top_role["skills_covered"].split("/")[1]) if "/" in top_role["skills_covered"] else 50
        new_coverage = round((coverage * total_skills / 100 + 1) / total_skills * 100, 1)
        improvement = round(new_coverage - coverage, 1)
        return (
            f"После освоения '{skill}' ваше покрытие навыков для роли "
            f"«{role_name}» вырастет с {coverage}% до {new_coverage}% (+{improvement}%). "
            f"Семантическая близость к роли сейчас {similarity}% — навык усилит ваши позиции "
            f"и откроет доступ к смежным вакансиям."
        )

    def _generate_explanation(self, skill: str, score: float, eval_result: dict) -> str:
        metric = eval_result.get("skill_metrics", {}).get(skill, {})
        cluster_rel = metric.get("cluster_relevance", 0)
        category = metric.get("category", "missing")
        cluster_context = eval_result.get("cluster_context") or {}
        closest = cluster_context.get("closest_clusters", [])
        top_cluster = closest[0] if closest else None
        top_cluster_name = top_cluster.get("name") if top_cluster else None
        top_cluster_sim = top_cluster.get("similarity", 0) if top_cluster else 0
        cluster_skills = cluster_context.get("skills", {})
        skill_in_top_cluster = skill in cluster_skills

        try:
            skill_cat = self._taxonomy.get_category_label(skill)
        except Exception:
            skill_cat = "технический"

        prefix = "🔶 УСИЛИТЬ: " if category == SkillCategory.WEAK else ""
        suffix = " У вас уже есть базовое понимание — углубите его." if category == SkillCategory.WEAK else ""

        if skill_in_top_cluster and top_cluster_name and cluster_rel > 0.5:
            return (
                f"{prefix}🎯 Ключевой навык для роли «{top_cluster_name}».\n"
                f"   Ваш профиль семантически близок к этой роли "
                f"({top_cluster_sim * 100:.0f}%), но для полного соответствия "
                f"не хватает '{skill}'.\n"
                f"   Навык входит в топ-требований этого направления."
                f"{suffix}"
            )
        if score > 0.6:
            return (
                f"{prefix}🔴 Высокий рыночный спрос.\n"
                f"   '{skill}' ({skill_cat}) — один из самых востребованных навыков "
                f"на рынке. Работодатели часто указывают его в требованиях."
                f"{suffix}"
            )
        if score > 0.35:
            return (
                f"{prefix}🟡 Умеренный спрос.\n"
                f"   '{skill}' ({skill_cat}) дополнит ваш профиль и повысит "
                f"привлекательность для работодателей в смежных областях."
                f"{suffix}"
            )
        return (
            f"{prefix}🟢 Дополнительное преимущество.\n"
            f"   '{skill}' полезен для расширения кругозора в категории '{skill_cat}'."
            f"{suffix}"
        )

    def _get_learning_path(
        self,
        skill: str,
        is_soft: bool,
        student_profile: StudentProfile | None = None,
    ) -> str:
        skill_lower = skill.lower()
        level = student_profile.target_level if student_profile else "middle"
        if is_soft:
            base = self.SOFT_LEARNING_PATHS.get(skill_lower, "Практикуйте навык постоянно.")
        else:
            base = self.HARD_LEARNING_PATHS.get(skill_lower, f"Изучите документацию '{skill}' и выполните проекты.")
        if level == "junior":
            base = "Сфокусируйтесь на основах: " + base
        elif level == "senior":
            base = "Углублённое изучение: " + base + " + архитектурные паттерны."
        if re.search(r"(усилить|🔶)", base, re.IGNORECASE):
            base = base.replace("Изучите документацию", "Углубите знания")
        return base

    def _is_hard_skill(self, skill_lower: str) -> bool:
        """Определяет, является ли навык техническим (hard skill).
        Использует кешированную таксономию и внешний список."""
        try:
            cat = self._taxonomy.get_category(skill_lower)
            soft_cats = {"soft_skills", "management"}
            if cat in soft_cats:
                return False
            hard_cats = {
                "programming_languages",
                "frameworks",
                "databases",
                "devops",
                "cloud",
                "data_science",
                "ml_advanced",
                "frontend",
                "mobile",
                "testing_qa",
                "security",
                "llm_ai",
                "enterprise",
                "gis",
                "embedded",
                "game_dev",
                "mathematics",
                "methodologies_concepts",
            }
            if cat in hard_cats:
                return True
        except Exception:
            pass
        return skill_lower in self._hard_keywords

    def _get_timeframe(self, skill: str) -> str:
        skill_lower = skill.lower()
        if skill_lower in self._timeframe_easy:
            return "1-2 недели"
        if skill_lower in self._timeframe_medium:
            return "1-2 месяца"
        if skill_lower in self._timeframe_hard:
            return "2-6 месяцев"
        return "1-3 месяца"

    # ------------------------------------------------------------------
    # LLM (YandexGPT) — вспомогательный, необязательный
    # ------------------------------------------------------------------

    def _llm_explain_with_retry(
        self, gap: Any, context: str, previous_explanations: list[str]
    ) -> Result[str, DomainError]:
        """Генерирует LLM-объяснение с повторными попытками."""
        prompt = self._build_explanation_prompt(gap, context, previous_explanations)

        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.explanation_model or "yandexgpt-lite",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                explanation = completion.choices[0].message.content
                if explanation:
                    return Ok(explanation)
            except Exception as e:
                logger.warning("llm_explain_attempt_failed", attempt=attempt + 1, error=str(e))
                time.sleep(1.0)

        return Err(DomainError(message="LLM explanation failed after retries", detail=f"gap={gap.skill_name}"))

    # ------------------------------------------------------------------
    # ШАБЛОНЫ
    # ------------------------------------------------------------------

    def _load_templates(self) -> None:
        templates_path = config.DATA_DIR / "templates" / "recommendation_templates.json"
        if templates_path.exists():
            try:
                with open(templates_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.HARD_LEARNING_PATHS = data.get("hard_paths", {})
                self.SOFT_LEARNING_PATHS = data.get("soft_paths", {})
                logger.info("templates_loaded", path=str(templates_path))
                return
            except Exception as e:
                logger.warning("templates_load_error", error=str(e))

        self.HARD_LEARNING_PATHS = {
            "python": "1. Основы Python. 2. Практика: 10+ мини-проектов. 3. Углубление.",
            "sql": "1. Основы SELECT. 2. JOIN и подзапросы. 3. Оптимизация.",
        }
        self.SOFT_LEARNING_PATHS = {
            "английский язык": "Занимайтесь ежедневно 30 минут.",
        }
