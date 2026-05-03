"""
Оценка профилей студентов на основе уровня опыта и gap-анализа.
Гибридная версия: новый API (для main.py) + кэширование.
"""
import logging
import math
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.models.student import StudentProfile
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.vacancy_clustering import VacancyClusterer
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.embedding_loader import get_embedding_model
from src import config

logger = logging.getLogger(__name__)


class ProfileEvaluator:
    """Оценка профилей с использованием единой модели SkillMetrics + Domain Coverage + Cluster Context."""

    def __init__(
        self,
        skill_weights: Dict[str, float],
        vacancies_skills: List[List[str]],
        vacancies_skills_dict: List[Dict],
        hybrid_weights: Optional[Dict[str, float]] = None,
        use_clustering: bool = True,
        skill_weights_by_level: Optional[Dict[str, Dict[str, float]]] = None,
        readiness_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        # Legacy параметры (для совместимости, если понадобятся)
        level_difficulty: Optional[Dict[str, float]] = None,
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
            'junior': self.clusterer.load_model('junior'),
            'middle': self.clusterer.load_model('middle'),
            'senior': self.clusterer.load_model('senior')
        }
        logger.info(f"Модели кластеризации загружены: {self.cluster_models_loaded}")

        # Кэширование
        self._cache = {}
        self._cache_path = config.DATA_PROCESSED_DIR / "evaluation_cache.json"
        self._load_cache()

    # ------------------------------------------------------------------
    # НОВЫЙ ОСНОВНОЙ МЕТОД (используется в main.py)
    # ------------------------------------------------------------------
    def evaluate_profile(self, student: StudentProfile, user_type: str = 'student') -> Dict[str, Any]:
        if not self.gap_analyzer_new:
            raise RuntimeError("skill_weights_by_level не были переданы в конструктор")

        user_skills_list = student.skills
        user_levels = {skill: 1.0 for skill in user_skills_list}
        user_skills_set = set(s.lower().strip() for s in user_skills_list)

        # === 1. Skill-level метрики ===
        metrics = self.gap_analyzer_new.compute_metrics(user_skills_list, user_levels)

        # === 2. Cluster Context + Relevance ===
        target_level = getattr(student, 'target_level', 'middle')
        cluster_context = self._get_cluster_context(student, target_level)

        for skill, metric in metrics.items():
            if cluster_context is not None and skill in cluster_context.get("skills", {}):
                metric.cluster_relevance = cluster_context["skills"][skill]
            else:
                metric.cluster_relevance = 0.15 * getattr(metric, 'cluster_relevance', 0.0)

        # === 3. Domain-level coverage ===
        domain_coverages = self.domain_analyzer.compute_domain_coverage(user_skills_list)

        # === 4. Бонусы от доменов ===
        skill_to_domain_bonus = {}
        for dom_name, dom in domain_coverages.items():
            for req_skill in dom.required_skills:
                req_norm = req_skill.lower().strip()
                bonus = dom.coverage if req_norm in user_skills_set else dom.coverage * 0.5
                if req_norm not in skill_to_domain_bonus or bonus > skill_to_domain_bonus[req_norm]:
                    skill_to_domain_bonus[req_norm] = bonus

        # === 5. Веса уровней ===
        level_weights = {
            'student': {'junior': 0.60, 'middle': 0.30, 'senior': 0.10},
            'junior':  {'junior': 0.40, 'middle': 0.40, 'senior': 0.20},
            'middle':  {'junior': 0.20, 'middle': 0.50, 'senior': 0.30}
        }.get(user_type, {'junior': 0.33, 'middle': 0.34, 'senior': 0.33})

        # === 6. Финальные скоры — ПРЯМАЯ ФИЛЬТРАЦИЯ ПО УЖЕ ИМЕЮЩИМСЯ НАВЫКАМ ===
        final_scores = {}
        user_skills_set = set(s.lower().strip() for s in user_skills_list)
        MIN_GAP_FOR_FALLBACK = 0.05

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
            logger.warning(f"Все рыночные навыки уже покрыты профилем {student.profile_name}. Fallback по cluster_relevance.")
            fallback_candidates = [
                (s, m) for s, m in metrics.items()
                if s.lower().strip() not in user_skills_set and max(m.gap_j, m.gap_m, m.gap_s) > MIN_GAP_FOR_FALLBACK
            ]
            for skill, metric in sorted(fallback_candidates, key=lambda x: x[1].cluster_relevance, reverse=True)[:15]:
                bonus = skill_to_domain_bonus.get(skill.lower().strip(), 0.0)
                final_scores[skill] = metric.score(level_weights, domain_bonus=bonus) * 0.65

        # === 7. Итоговые метрики ===
        # Реальное покрытие рынка (доля навыков студента от всех рыночных навыков)
        user_skills_norm = {SkillNormalizer.normalize(s) for s in user_skills_list}
        all_market_skills = list(metrics.keys())
        covered_market = sum(1 for s in all_market_skills if s in user_skills_norm)
        market_skill_coverage_pct = round(covered_market / len(all_market_skills) * 100, 2)

        skill_coverage = sum(final_scores.values()) / max(len(final_scores), 1) * 100
        avg_domain_cov = sum(d.coverage for d in domain_coverages.values()) / max(len(domain_coverages), 1)
        domain_coverage_score = avg_domain_cov * 100
        market_coverage_score = 0.70 * skill_coverage + 0.30 * domain_coverage_score

        total_gap = sum((m.gap_j + m.gap_m + m.gap_s) / 3 for m in metrics.values())
        avg_gap = total_gap / max(len(metrics), 1)

        readiness_score = self._calculate_readiness(
            market_coverage_score=market_coverage_score,
            skill_coverage=skill_coverage,
            domain_coverage_score=domain_coverage_score,
            avg_gap=avg_gap
        )

        gaps = {s: m.__dict__ for s, m in metrics.items()
                if max(m.gap_j, m.gap_m, m.gap_s) > 0.15}

        return {
            "market_coverage_score": round(market_coverage_score, 2),   # потребность в дообучении
            "skill_coverage": round(skill_coverage, 2),
            "domain_coverage_score": round(domain_coverage_score, 2),
            "readiness_score": readiness_score,
            "avg_gap": round(avg_gap * 100, 2),
            "skill_metrics": {s: m.__dict__ for s, m in metrics.items()},
            "domain_coverage": {d: dm.__dict__ for d, dm in domain_coverages.items()},
            "cluster_context": cluster_context,
            "top_recommendations": sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:15],
            "gaps": gaps,
            "level_weights_used": level_weights,
            "student_skills": user_skills_list,
            "market_skill_coverage": market_skill_coverage_pct   # новое: реальное покрытие
        }
    # ------------------------------------------------------------------
    # Вспомогательные методы нового API
    # ------------------------------------------------------------------
    def _get_cluster_context(self, student: StudentProfile, target_level: str) -> Optional[Dict]:
        """Получить кластерный контекст, если модель загружена."""
        if not self.use_clustering:
            return None
        if target_level not in self.cluster_models_loaded or not self.cluster_models_loaded[target_level]:
            logger.info(f"Кластеризатор для уровня {target_level} не обучен, пропускаем")
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
                profile_embedding=student_emb,
                level=target_level,
                top_k_clusters=3,
                top_k_skills_per_cluster=25
            )
            logger.info(f"Кластерный контекст для {target_level}: "
                       f"{cluster_context['total_skills_in_context']} навыков из "
                       f"{len(cluster_context.get('closest_clusters', []))} кластеров")
            return cluster_context
        except Exception as e:
            logger.warning(f"Не удалось получить cluster_context: {e}")
            return None

    def _calculate_readiness(self, market_coverage_score: float,
                             skill_coverage: float,
                             domain_coverage_score: float,
                             avg_gap: float) -> float:
        """Новая readiness — полностью согласована с новыми метриками"""
        w_market = 0.50
        w_skill = 0.20
        w_domain = 0.20
        w_gap_penalty = 0.10

        readiness = (
            w_market * market_coverage_score +
            w_skill * skill_coverage +
            w_domain * domain_coverage_score -
            w_gap_penalty * (avg_gap * 100)
        )
        return round(max(0.0, min(100.0, readiness)), 2)

    def _get_or_create_comparator(self, target_level: str, level_analyzer=None) -> CompetencyComparator:
        if target_level in self.comparators:
            return self.comparators[target_level]

        logger.info(f"Создаём level-specific Embedding Comparator для {target_level}...")
        comparator = CompetencyComparator(
            use_embeddings=True,
            level=target_level
        )
        success = comparator.fit_market(self.vacancies_skills)
        if success:
            logger.info(f"  ✓ {target_level} comparator успешно обучен (embeddings)")
        else:
            logger.warning(f"  ⚠️ Не удалось обучить {target_level} comparator")

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
                with open(self._cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, indent=2)

    def _get_student_hash(self, student: StudentProfile, level: str) -> str:
        skills_str = ",".join(sorted(set(s.lower() for s in student.skills)))
        data = f"{level}:{skills_str}"
        return hashlib.md5(data.encode()).hexdigest()