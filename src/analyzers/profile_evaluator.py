"""
Оценка профилей студентов на основе уровня опыта и gap-анализа
"""

import logging
import math
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.models.student import (
    StudentProfile,
    ProfileEvaluation,
    ProfileComparison,
    ExperienceLevel
)
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.comparator import CompetencyComparator
from src.analyzers.vacancy_clustering import VacancyClusterer
from src.parsing.skill_normalizer import SkillNormalizer
from src import config

logger = logging.getLogger(__name__)


class ProfileEvaluator:
    DEFAULT_LEVEL_DIFFICULTY = {
        'junior': 1.0,
        'middle': 1.1,
        'senior': 1.25
    }
    DEFAULT_READINESS_WEIGHTS = (0.5, 0.3, 0.2)

    def __init__(
        self,
        skill_weights: Dict[str, float],
        vacancies_skills: List[List[str]],
        vacancies_skills_dict: List[Dict],
        hybrid_weights: Optional[Dict[str, float]] = None,
        level_difficulty: Dict[str, float] = None,
        readiness_weights: Tuple[float, float, float] = None,
        use_clustering: bool = True
    ):
        self.skill_weights = skill_weights
        self.hybrid_weights = hybrid_weights or {}
        self.vacancies_skills = vacancies_skills
        self.vacancies_skills_dict = vacancies_skills_dict
        self.comparators = {}
        self.level_difficulty = level_difficulty or self.DEFAULT_LEVEL_DIFFICULTY
        self.readiness_weights = readiness_weights or self.DEFAULT_READINESS_WEIGHTS
        self.clusterer = VacancyClusterer()
        self.use_clustering = use_clustering

        # Загружаем модели кластеризации для всех уровней при инициализации
        self.cluster_models_loaded = {
            'junior': self.clusterer.load_model('junior'),
            'middle': self.clusterer.load_model('middle'),
            'senior': self.clusterer.load_model('senior')
        }
        logger.info(f"Модели кластеризации загружены: {self.cluster_models_loaded}")

    # ----------------------------------------------------------------------
    # Кэширование
    # ----------------------------------------------------------------------
    def _get_cache_path(self) -> Path:
        return config.DATA_PROCESSED_DIR / "evaluation_cache.json"

    def _load_cache(self) -> Dict:
        path = self._get_cache_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self, cache: Dict):
        path = self._get_cache_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

    def _get_student_hash(self, student: StudentProfile, level: str) -> str:
        skills_str = ",".join(sorted(set(s.lower() for s in student.skills)))
        data = f"{level}:{skills_str}"
        return hashlib.md5(data.encode()).hexdigest()

    # ----------------------------------------------------------------------
    # Вспомогательные метрики
    # ----------------------------------------------------------------------
    def _shannon_diversity(self, skills: List[str], weights: Dict[str, float]) -> float:
        if not skills or not weights:
            return 0.0

        total = sum(weights.values())
        if total == 0:
            return 0.0

        skill_weights = [weights.get(s.lower(), 0.0) for s in skills]
        norm_weights = [w / total for w in skill_weights if w > 0]

        if not norm_weights:
            return 0.0

        entropy = -sum(p * math.log(p) for p in norm_weights)
        max_entropy = math.log(len(skills)) if len(skills) > 0 else 1.0
        return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0

    # ----------------------------------------------------------------------
    # Оценка одного профиля
    # ----------------------------------------------------------------------
    def evaluate_profile(
        self,
        student: StudentProfile,
        profile_name: str,
        level_analyzer,
        profile_skill_weights: Dict[str, float],
        profile_hybrid_weights: Optional[Dict[str, float]] = None
    ) -> ProfileEvaluation:
        target_level = student.target_level
        logger.info(f"Оцениваем профиль {profile_name} ({target_level})...")

        # Проверяем кэш
        cache = self._load_cache()
        student_hash = self._get_student_hash(student, target_level)
        if student_hash in cache:
            cached_data = cache[student_hash]
            logger.info(f"  ✓ {profile_name}: загружено из кэша")
            return ProfileEvaluation.parse_obj(cached_data)

        # 1. Получаем comparator (нужен для embedding_comparator)
        profile_comparator = self._get_or_create_comparator(target_level, level_analyzer)
        embedding_comp = profile_comparator.embedding_comparator

        # 2. Ищем ближайшие вакансии целевого уровня
        level_vacancies = [v for v in self.vacancies_skills_dict if v.get('experience') == target_level]
        if not level_vacancies:
            level_vacancies = self.vacancies_skills_dict  # fallback

        top_vacancies = embedding_comp.find_closest_vacancies(
            student.skills,
            level_vacancies,
            level=target_level,
            top_k=50
        )

        if not top_vacancies:
            # Fallback на старый метод
            score, confidence = profile_comparator.compare(student.skills)
            gap_analyzer_freq = GapAnalyzer(profile_skill_weights)
            gaps = gap_analyzer_freq.analyze_gap(student.skills)
            weighted_freq_cov, wf_details = gap_analyzer_freq.coverage(student.skills, method='weighted')
            simple_cov, s_details = gap_analyzer_freq.coverage(student.skills, method='simple')
            weighted_hybrid_cov = None
            wh_details = None
            global_cov = weighted_freq_cov
            cluster_score = None
            best_cluster = None
            cluster_similarity = 0.0
            all_clusters = []
        else:
            # 3. Строим взвешенное покрытие только по навыкам из топ вакансий
            market_skills = set()
            for vac in top_vacancies:
                market_skills.update(vac.get('skills', []))
            market_skills = list(market_skills)

            # Ограничиваем веса только этими навыками
            top_skill_weights = {s: profile_skill_weights.get(s, 0.0) for s in market_skills}
            top_hybrid_weights = None
            if profile_hybrid_weights:
                top_hybrid_weights = {s: profile_hybrid_weights.get(s, 0.0) for s in market_skills}

            # GapAnalyzer на урезанных весах
            gap_analyzer_top = GapAnalyzer(top_skill_weights)
            gaps = gap_analyzer_top.analyze_gap(student.skills)
            weighted_freq_cov, wf_details = gap_analyzer_top.coverage(student.skills, method='weighted')
            simple_cov, s_details = gap_analyzer_top.coverage(student.skills, method='simple')

            # Score = доля покрытых навыков среди топ-рынка
            covered = len(set(student.skills) & set(market_skills))
            score = covered / len(market_skills) if market_skills else 0.0
            confidence = min(1.0, covered / max(1, len(student.skills)))

            # Гибридное покрытие (если есть)
            weighted_hybrid_cov = None
            wh_details = None
            if top_hybrid_weights:
                gap_analyzer_hybrid = GapAnalyzer(top_hybrid_weights)
                weighted_hybrid_cov, wh_details = gap_analyzer_hybrid.coverage(student.skills, method='weighted')

            global_cov = weighted_freq_cov

            # === ГИБРИДНЫЙ СКОРИНГ: ГЛОБАЛЬНЫЙ + КЛАСТЕРНЫЙ ===
            cluster_score = None
            best_cluster = None
            cluster_similarity = 0.0
            all_clusters = []

            if self.use_clustering and top_vacancies:
                if self.cluster_models_loaded.get(target_level, False):
                    closest = self.clusterer.find_closest_clusters(student.skills, top_k=3)
                    if closest:
                        all_clusters = []
                        for cid, sim in closest:
                            raw_skills = self.clusterer.get_cluster_skills(cid, level_vacancies)
                            # Нормализуем и фильтруем навыки
                            norm_skills = []
                            for s in raw_skills:
                                norm = SkillNormalizer.normalize(s)
                                if norm and len(norm.split()) <= 3:
                                    norm_skills.append(norm)
                            # Убираем дубликаты
                            seen = set()
                            unique_skills = []
                            for s in norm_skills:
                                if s not in seen:
                                    seen.add(s)
                                    unique_skills.append(s)
                            all_clusters.append({
                                "cluster_id": cid,
                                "similarity": sim,
                                "coverage": 0.0,
                                "top_skills": unique_skills[:10]
                            })
                        best_cluster, cluster_similarity = closest[0]
                        cluster_skills = all_clusters[0]["top_skills"]  # уже нормализованные
                        if cluster_skills:
                            cluster_weights = {s: profile_skill_weights.get(s, 0.0) for s in cluster_skills}
                            cluster_gap = GapAnalyzer(cluster_weights)
                            cluster_score, _ = cluster_gap.coverage(student.skills, method='weighted')
                            # Обновим coverage в all_clusters
                            for i, _ in enumerate(closest):
                                if i == 0:
                                    all_clusters[i]["coverage"] = cluster_score
                                else:
                                    all_clusters[i]["coverage"] = cluster_score * 0.9  # приблизительно
                            logger.info(f"  Лучший кластер: {best_cluster}, сходство: {cluster_similarity:.2f}, покрытие: {cluster_score:.1f}%")
                else:
                    logger.warning(f"Модель кластеризации для уровня {target_level} не загружена")

            if cluster_score is not None:
                weighted_freq_cov = 0.4 * global_cov + 0.6 * cluster_score
            else:
                weighted_freq_cov = global_cov

        # 4. Adjusted coverage (используем weighted_freq)
        difficulty_multiplier = self.level_difficulty.get(target_level, 1.0)
        adjusted_coverage = weighted_freq_cov / difficulty_multiplier

        # 5. Readiness
        readiness_score = self._calculate_readiness(
            score, adjusted_coverage, gaps, difficulty_multiplier
        )

        # 6. Индекс разнообразия
        diversity_index = self._shannon_diversity(student.skills, profile_skill_weights)

        # 7. Формируем coverage
        coverage_dict = {
            'weighted_freq': weighted_freq_cov,
            'weighted_hybrid': weighted_hybrid_cov if weighted_hybrid_cov is not None else weighted_freq_cov,
            'simple': simple_cov,
            'adjusted': adjusted_coverage,
            'difficulty_multiplier': difficulty_multiplier,
            'uniqueness_ratio': round(simple_cov / weighted_freq_cov, 2) if weighted_freq_cov > 0 else 0.0,
            'diversity_index': diversity_index,
            'global_coverage': global_cov,
            'cluster_coverage': cluster_score,
            'best_cluster': best_cluster,
            'cluster_similarity': cluster_similarity,
            'all_clusters': all_clusters,
            'weighted_freq_details': wf_details,
            'weighted_hybrid_details': wh_details,
            'simple_details': s_details
        }

        evaluation = ProfileEvaluation(
            profile_name=profile_name,
            student=student,
            level=target_level,
            raw_score=score,
            confidence=confidence,
            coverage=coverage_dict,
            readiness_score=readiness_score,
            recommendation=self._get_recommendation(readiness_score, target_level),
            gaps={
                'high_priority': len(gaps['high_priority']),
                'medium_priority': len(gaps['medium_priority']),
                'low_priority': len(gaps['low_priority']),
                'total': gaps['total_gaps'],
                'high_priority_skills': [g['skill'] for g in gaps['high_priority'][:5]],
                'medium_priority_skills': [g['skill'] for g in gaps['medium_priority'][:5]]
            }
        )

        logger.info(f"  ✓ {profile_name}: readiness={readiness_score:.2f}%, "
                    f"freq_cov={weighted_freq_cov:.2f}%, hybrid_cov={coverage_dict['weighted_hybrid']:.2f}%, "
                    f"simple_cov={simple_cov:.2f}%, diversity={diversity_index:.3f}")

        # Сохраняем в кэш
        cache[student_hash] = json.loads(evaluation.json())
        self._save_cache(cache)

        return evaluation

    # ----------------------------------------------------------------------
    # Оценка нескольких профилей
    # ----------------------------------------------------------------------
    def evaluate_multiple_profiles(
        self,
        profiles: Dict[str, StudentProfile],
        level_analyzer,
        skill_weights_by_level: Dict[str, Dict[str, float]],
        hybrid_weights_by_level: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ProfileComparison:
        if not profiles:
            raise ValueError("Нужен хотя бы один профиль для оценки")

        base_student = next(iter(profiles.values()))
        evaluations = []
        readiness_scores = []

        hybrid_by_level = hybrid_weights_by_level or {}

        for profile_name, student in profiles.items():
            target_level = student.target_level
            profile_weights = skill_weights_by_level.get(target_level, self.skill_weights)
            profile_hybrid = hybrid_by_level.get(target_level)

            evaluation = self.evaluate_profile(
                student,
                profile_name,
                level_analyzer,
                profile_weights,
                profile_hybrid
            )
            evaluations.append(evaluation)
            readiness_scores.append(evaluation.readiness_score)

        best_evaluation = max(evaluations, key=lambda e: e.readiness_score)
        average_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0
        summary = self._generate_summary(evaluations, best_evaluation, average_readiness)

        comparison = ProfileComparison(
            student=base_student,
            evaluations=evaluations,
            best_evaluation=best_evaluation,
            average_readiness=average_readiness,
            summary=summary
        )

        logger.info(f"✅ Оценены {len(evaluations)} профилей")
        logger.info(f"   Average readiness: {average_readiness:.2f}%")
        logger.info(f"   Best: {best_evaluation.profile_name} ({best_evaluation.readiness_score:.2f}%)")

        return comparison

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    def _calculate_readiness(self, score, adjusted_coverage, gaps, difficulty_multiplier):
        w_cov, w_score, w_gap = self.readiness_weights
        coverage_component = adjusted_coverage
        score_component = min(score * 100, 100)
        high_gaps = gaps.get('high_priority', [])
        gap_component = max(0, 100 - (len(high_gaps) * 10))
        readiness = (
            w_cov * coverage_component +
            w_score * score_component +
            w_gap * gap_component
        )
        return min(readiness, 100)

    def _get_or_create_comparator(self, target_level: str, level_analyzer) -> CompetencyComparator:
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

    def _generate_summary(
        self,
        evaluations: List[ProfileEvaluation],
        best: ProfileEvaluation,
        avg_readiness: float
    ) -> str:
        lines = [
            f"Оценено {len(evaluations)} профилей",
            f"Средняя готовность: {avg_readiness:.2f}%",
            f"Лучший профиль: {best.profile_name} ({best.readiness_score:.2f}%)",
            f"Рекомендация: {best.recommendation}"
        ]
        return "\n".join(lines)