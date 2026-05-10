"""
Оценка профилей студентов на основе уровня опыта и gap-анализа.
Гибридная версия: новый API (для main.py) + кэширование.
"""

import hashlib
import json
from typing import Any

import numpy as np
import structlog

from src import config
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.vacancy_clustering import VacancyClusterer
from src.models.data_contracts import ProfileEvaluationResult
from src.models.student import StudentProfile
from src.parsing.embedding_loader import get_embedding_model
from src.parsing.skill_normalizer import SkillNormalizer

logger = structlog.get_logger(__name__)


class ProfileEvaluator:
    """Оценка профилей с использованием единой модели SkillMetrics + Domain Coverage + Cluster Context."""

    def __init__(
        self,
        skill_weights: dict[str, float],
        vacancies_skills: list[list[str]],
        vacancies_skills_dict: list[dict],
        hybrid_weights: dict[str, float] | None = None,
        use_clustering: bool = True,
        skill_weights_by_level: dict[str, dict[str, float]] | None = None,
        readiness_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        # Legacy параметры (для совместимости, если понадобятся)
        level_difficulty: dict[str, float] | None = None,
    ):
        self.skill_weights = skill_weights
        self.hybrid_weights = hybrid_weights or {}
        self.vacancies_skills = vacancies_skills
        self.vacancies_skills_dict = vacancies_skills_dict
        self.comparators = {}
        self.clusterer = VacancyClusterer()
        self.use_clustering = use_clustering
        self.domain_analyzer = DomainAnalyzer()
        self.readiness_weights = readiness_weights

        # Для новой единой модели
        self.skill_weights_by_level = skill_weights_by_level or {}
        if self.skill_weights_by_level:
            self.gap_analyzer_new = GapAnalyzer(self.skill_weights_by_level)
        else:
            self.gap_analyzer_new = None

        # Загружаем модели кластеризации при инициализации
        self.cluster_models_loaded = {
            "junior": self.clusterer.load_model("junior"),
            "middle": self.clusterer.load_model("middle"),
            "senior": self.clusterer.load_model("senior"),
        }
        logger.info(
            "cluster_models_loaded",
            junior=self.cluster_models_loaded["junior"],
            middle=self.cluster_models_loaded["middle"],
            senior=self.cluster_models_loaded["senior"],
        )

        # Кэширование
        self._cache = {}
        self._cache_path = config.DATA_PROCESSED_DIR / "evaluation_cache.json"
        self._load_cache()

    # ------------------------------------------------------------------
    # НОВЫЙ ОСНОВНОЙ МЕТОД (используется в main.py)
    # ------------------------------------------------------------------
    def evaluate_profile(self, student: StudentProfile, user_type: str = "student") -> dict[str, Any]:
        if not self.gap_analyzer_new:
            raise RuntimeError("skill_weights_by_level не были переданы в конструктор")

        user_skills_list = student.skills
        user_levels = {skill: 1.0 for skill in user_skills_list}
        user_skills_set = set(s.lower().strip() for s in user_skills_list)

        # === 1. Skill-level метрики ===
        metrics = self.gap_analyzer_new.compute_metrics(user_skills_list, user_levels)

        # === 2. Cluster Context + Relevance ===
        target_level = getattr(student, "target_level", "middle")
        cluster_context = self._get_cluster_context(student, target_level)

        for skill, metric in metrics.items():
            if cluster_context is not None and skill in cluster_context.get("skills", {}):
                metric.cluster_relevance = cluster_context["skills"][skill]
            else:
                metric.cluster_relevance = 0.15 * getattr(metric, "cluster_relevance", 0.0)

        # === 3. Domain-level coverage (с весом доминирующего домена) ===
        domain_coverages = self.domain_analyzer.compute_domain_coverage(user_skills_list)

        # Определяем доминирующий домен студента
        if domain_coverages:
            dominant_domain = max(domain_coverages.items(), key=lambda x: x[1].coverage)
            dominant_name = dominant_domain[0]
        else:
            dominant_name = None

        # Взвешенное покрытие: доминирующий домен получает вес 0.5
        weighted_cov_sum = 0.0
        other_count = len(domain_coverages) - 1 if len(domain_coverages) > 1 else 0

        for dom_name, dom in domain_coverages.items():
            weight = 0.5 if dom_name == dominant_name else 0.5 / other_count if other_count > 0 else 0.5
            weighted_cov_sum += dom.coverage * weight

        # Итоговое доменное покрытие (0-100%)
        domain_coverage_score = weighted_cov_sum * 100

        logger.debug(
            "domain_coverage_calculated",
            dominant_domain=dominant_name,
            domain_coverage_score=round(domain_coverage_score, 1),
        )

        # === 4. Бонусы от доменов ===
        skill_to_domain_bonus = {}
        for _dom_name, dom in domain_coverages.items():
            for req_skill in dom.required_skills:
                req_norm = req_skill.lower().strip()
                bonus = dom.coverage if req_norm in user_skills_set else dom.coverage * 0.5
                if req_norm not in skill_to_domain_bonus or bonus > skill_to_domain_bonus[req_norm]:
                    skill_to_domain_bonus[req_norm] = bonus

        # === 5. Веса уровней ===
        level_weights = {
            "student": {"junior": 0.60, "middle": 0.30, "senior": 0.10},
            "junior": {"junior": 0.40, "middle": 0.40, "senior": 0.20},
            "middle": {"junior": 0.20, "middle": 0.50, "senior": 0.30},
        }.get(user_type, {"junior": 0.33, "middle": 0.34, "senior": 0.33})

        # === 6. Финальные скоры — ПРЯМАЯ ФИЛЬТРАЦИЯ ПО УЖЕ ИМЕЮЩИМСЯ НАВЫКАМ ===
        final_scores = {}
        user_skills_set = set(s.lower().strip() for s in user_skills_list)
        min_gap_for_fallback = 0.05

        for skill, metric in metrics.items():
            skill_norm = skill.lower().strip()
            # Пропускаем уже освоенные навыки
            if skill_norm in user_skills_set:
                continue

            max_gap = max(metric.gap_j, metric.gap_m, metric.gap_s)
            if max_gap > 0.05:
                bonus = skill_to_domain_bonus.get(skill_norm, 0.0)
                final_scores[skill] = metric.score(level_weights, domain_bonus=bonus)

        if not final_scores and metrics:
            logger.warning(
                "all_market_skills_covered_fallback",
                profile_name=student.profile_name,
            )
            fallback_candidates = [
                (s, m)
                for s, m in metrics.items()
                if s.lower().strip() not in user_skills_set and max(m.gap_j, m.gap_m, m.gap_s) > min_gap_for_fallback
            ]
            for skill, metric in sorted(fallback_candidates, key=lambda x: x[1].cluster_relevance, reverse=True)[:15]:
                bonus = skill_to_domain_bonus.get(skill.lower().strip(), 0.0)
                final_scores[skill] = metric.score(level_weights, domain_bonus=bonus) * 0.65

        # === 7. Итоговые метрики (улучшенные) ===
        total_market = len(metrics)

        # Категоризация навыков
        strong_count = 0
        weak_count = 0
        missing_count = 0
        weighted_cov = 0.0
        max_possible = 0.0

        for m in metrics.values():
            max_gap = max(m.gap_j, m.gap_m, m.gap_s)
            max_demand = max(m.demand_j, m.demand_m, m.demand_s)
            max_possible += max_demand

            if max_gap < 0.2:
                strong_count += 1
                weighted_cov += 1.0 * max_demand
            elif max_gap < 0.6:
                weak_count += 1
                weighted_cov += 0.5 * max_demand
            else:
                missing_count += 1

        # Навыковое покрытие (взвешенное по категориям)
        skill_coverage = weighted_cov / max_possible * 100 if max_possible > 0 else 0.0

        # Общее покрытие рынка
        market_coverage_score = 0.60 * skill_coverage + 0.40 * domain_coverage_score

        # Готовность к уровню
        readiness = (
            0.50 * market_coverage_score
            + 0.20 * (strong_count / total_market * 100)
            + 0.15 * (weak_count / total_market * 100)
            + 0.15 * domain_coverage_score
        )

        total_gap = sum((m.gap_j + m.gap_m + m.gap_s) / 3 for m in metrics.values())
        avg_gap = total_gap / max(len(metrics), 1) if metrics else 0.0

        readiness_score = round(max(0.0, min(100.0, readiness)), 2)

        # Реальное покрытие (доля навыков студента от рынка)
        user_skills_norm = {SkillNormalizer.normalize(s) for s in user_skills_list}
        all_market_skills = list(metrics.keys())
        covered_market = sum(1 for s in all_market_skills if s in user_skills_norm)
        market_skill_coverage_pct = (
            round(covered_market / len(all_market_skills) * 100, 2) if all_market_skills else 0.0
        )

        gaps = {s: m.__dict__ for s, m in metrics.items() if max(m.gap_j, m.gap_m, m.gap_s) > 0.15}

        # Добавляем статистику категорий
        skill_categories = {"strong": strong_count, "weak": weak_count, "missing": missing_count, "total": total_market}

        # Собираем строгую модель
        eval_result = ProfileEvaluationResult(
            market_coverage_score=round(market_coverage_score, 2),
            skill_coverage=round(skill_coverage, 2),
            domain_coverage_score=round(domain_coverage_score, 2),
            readiness_score=readiness_score,
            avg_gap=round(avg_gap * 100, 2),
            skill_metrics={s: m.__dict__ for s, m in metrics.items()},
            domain_coverage={d: dm.__dict__ for d, dm in domain_coverages.items()},
            cluster_context=cluster_context,
            top_recommendations=sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:15],
            gaps=gaps,
            level_weights_used=level_weights,
            student_skills=user_skills_list,
            market_skill_coverage=market_skill_coverage_pct,
            skill_categories=skill_categories,
        )
        return eval_result.model_dump()

    # ------------------------------------------------------------------
    # Вспомогательные методы нового API
    # ------------------------------------------------------------------
    def _get_cluster_context(self, student: StudentProfile, target_level: str) -> dict | None:
        """Получить кластерный контекст, если модель загружена."""
        if not self.use_clustering:
            return None
        if target_level not in self.cluster_models_loaded or not self.cluster_models_loaded[target_level]:
            logger.info("clusterer_not_trained_for_level", level=target_level)
            return None

        try:
            embedding_model = get_embedding_model()
            user_skills = student.skills
            if not user_skills:
                student_emb = np.zeros(embedding_model.get_sentence_embedding_dimension())
            else:
                embs = embedding_model.encode(user_skills, convert_to_numpy=True, show_progress_bar=False)
                student_emb = np.mean(embs, axis=0)

            cluster_context = self.clusterer.get_cluster_context(
                profile_embedding=student_emb, level=target_level, top_k_clusters=5, top_k_skills_per_cluster=25
            )
            logger.info(
                "cluster_context_obtained",
                level=target_level,
                total_skills=cluster_context["total_skills_in_context"],
                clusters_count=len(cluster_context.get("closest_clusters", [])),
            )
            return cluster_context
        except Exception as e:
            logger.warning("cluster_context_failed", error=str(e))
            return None

    def _calculate_readiness(
        self, market_coverage_score: float, skill_coverage: float, domain_coverage_score: float, avg_gap: float
    ) -> float:
        """Новая readiness — полностью согласована с новыми метриками"""
        w_market = 0.50
        w_skill = 0.20
        w_domain = 0.20
        w_gap_penalty = 0.10

        readiness = (
            w_market * market_coverage_score
            + w_skill * skill_coverage
            + w_domain * domain_coverage_score
            - w_gap_penalty * (avg_gap * 100)
        )
        return round(max(0.0, min(100.0, readiness)), 2)

    def _get_or_create_comparator(self, target_level: str, level_analyzer=None) -> CompetencyComparator:
        if target_level in self.comparators:
            return self.comparators[target_level]

        logger.info("creating_level_comparator", level=target_level)
        comparator = CompetencyComparator(use_embeddings=True, level=target_level)
        success = comparator.fit_market(self.vacancies_skills)
        if success:
            logger.info("level_comparator_trained", level=target_level)
        else:
            logger.warning("level_comparator_training_failed", level=target_level)

        self.comparators[target_level] = comparator
        return comparator

    def _get_recommendation(self, readiness_score: float, target_level: str) -> str:
        if readiness_score >= 80:
            return f"✅ Готов к {target_level} уровню"
        elif readiness_score >= 60:
            return f"📈 Неплохо для {target_level}, но есть пробелы"
        elif readiness_score >= 40:
            return f"⚠️ Нужно подготовиться к {target_level}"
        else:
            return f"❌ Недостаточно готов к {target_level}"

    # ------------------------------------------------------------------
    # Кэширование
    # ------------------------------------------------------------------
    def _load_cache(self):
        if self._cache_path.exists():
            try:
                with open(self._cache_path, encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def _get_student_hash(self, student: StudentProfile, level: str) -> str:
        skills_str = ",".join(sorted(set(s.lower() for s in student.skills)))
        data = f"{level}:{skills_str}"
        return hashlib.sha256(data.encode()).hexdigest()
