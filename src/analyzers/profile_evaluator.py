"""
Оценка профилей студентов на основе уровня опыта и gap-анализа
"""

import logging
from typing import Dict, List, Tuple
from src.models.student import (
    StudentProfile, 
    ProfileEvaluation, 
    ProfileComparison,
    ExperienceLevel
)
from src.analyzers.gap_analyzer import GapAnalyzer
from src.analyzers.comparator import CompetencyComparator

logger = logging.getLogger(__name__)


class ProfileEvaluator:
    # Коэффициенты по умолчанию
    DEFAULT_LEVEL_DIFFICULTY = {
        'junior': 1.0,
        'middle': 1.1,
        'senior': 1.25
    }
    DEFAULT_READINESS_WEIGHTS = (0.5, 0.3, 0.2)  # coverage, score, gap

    def __init__(
        self,
        skill_weights: Dict[str, float],
        vacancies_skills: List[List[str]],
        level_difficulty: Dict[str, float] = None,
        readiness_weights: Tuple[float, float, float] = None
    ):
        self.skill_weights = skill_weights
        self.vacancies_skills = vacancies_skills
        self.comparators = {}
        self.level_difficulty = level_difficulty or self.DEFAULT_LEVEL_DIFFICULTY
        self.readiness_weights = readiness_weights or self.DEFAULT_READINESS_WEIGHTS

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
    
    def evaluate_profile(
        self,
        student: StudentProfile,
        profile_name: str,
        level_analyzer,  # SkillLevelAnalyzer
        profile_skill_weights: Dict[str, float]  # веса для уровня
    ) -> ProfileEvaluation:
        """
        Полная оценка профиля студента
        
        Args:
            student: StudentProfile объект
            profile_name: Название профиля (base, dc, top_dc)
            level_analyzer: SkillLevelAnalyzer для анализа навыков по уровням
            profile_skill_weights: Переупорядоченные веса для уровня
        
        Returns:
            ProfileEvaluation с результатами
        """
        target_level = student.target_level
        logger.info(f"Оцениваем профиль {profile_name} ({target_level})...")
        
        # 1. Получаем или создаём comparator для этого уровня
        profile_comparator = self._get_or_create_comparator(target_level, level_analyzer)
        
        # 2. Сравниваем студента
        score, confidence = profile_comparator.compare(student.skills)
        
        # 3. Анализируем gaps с уровень-специфичными весами
        profile_gap_analyzer = GapAnalyzer(profile_skill_weights)
        gaps = profile_gap_analyzer.analyze_gap(student.skills)
        coverage, coverage_details = profile_gap_analyzer.coverage(student.skills)
        
        # 4. Вычисляем adjusted coverage с учётом сложности
        difficulty_multiplier = self.level_difficulty.get(target_level, 1.0)
        adjusted_coverage = coverage / difficulty_multiplier
        
        # 5. Вычисляем readiness скор
        readiness_score = self._calculate_readiness(
            score, 
            adjusted_coverage, 
            gaps,
            difficulty_multiplier
        )
        
        # 6. Создаём объект оценки
        evaluation = ProfileEvaluation(
            profile_name=profile_name,
            student=student,
            level=target_level,
            raw_score=score,
            confidence=confidence,
            coverage={
                'raw': coverage,
                'adjusted': adjusted_coverage,
                'difficulty_multiplier': difficulty_multiplier
            },
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
        
        logger.info(f"  ✓ {profile_name}: readiness={readiness_score:.2f}%, coverage={adjusted_coverage:.2f}%")
        
        return evaluation
    
    def evaluate_multiple_profiles(
        self,
        profiles: Dict[str, StudentProfile],  # {profile_name: StudentProfile}
        level_analyzer,
        skill_weights_by_level: Dict[str, Dict[str, float]]  # {level: {skill: weight}}
    ) -> ProfileComparison:
        """
        Оценивает несколько профилей и возвращает сравнение
        
        Args:
            profiles: Словарь профилей {name: StudentProfile}
            level_analyzer: SkillLevelAnalyzer
            skill_weights_by_level: Веса по уровням
        
        Returns:
            ProfileComparison с полной сравнительной оценкой
        """
        if not profiles:
            raise ValueError("Нужен хотя бы один профиль для оценки")
        
        # Берём первый профиль как базовый для сравнения
        base_student = next(iter(profiles.values()))
        
        evaluations = []
        readiness_scores = []
        
        for profile_name, student in profiles.items():
            target_level = student.target_level
            profile_weights = skill_weights_by_level.get(target_level, self.skill_weights)
            
            evaluation = self.evaluate_profile(
                student,
                profile_name,
                level_analyzer,
                profile_weights
            )
            
            evaluations.append(evaluation)
            readiness_scores.append(evaluation.readiness_score)
        
        # Находим лучшую оценку
        best_evaluation = max(evaluations, key=lambda e: e.readiness_score)
        
        # Вычисляем среднюю готовность
        average_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0
        
        # Генерируем сводку
        summary = self._generate_summary(evaluations, best_evaluation, average_readiness)
        
        # Создаём объект сравнения
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
    
    def _get_or_create_comparator(self, target_level: str, level_analyzer) -> CompetencyComparator:
        """Получает или создаёт level-specific comparator с embeddings"""
        if target_level in self.comparators:
            return self.comparators[target_level]

        logger.info(f"Создаём level-specific Embedding Comparator для {target_level}...")

        comparator = CompetencyComparator(
            use_embeddings=True,      # ← ВКЛЮЧАЕМ embeddings
            level=target_level
        )
        
        # fit_market вызывается здесь — каждый уровень получает свой кэш
        success = comparator.fit_market(self.vacancies_skills)
        
        if success:
            logger.info(f"  ✓ {target_level} comparator успешно обучен (embeddings)")
        else:
            logger.warning(f"  ⚠️ Не удалось обучить {target_level} comparator")

        self.comparators[target_level] = comparator
        return comparator
    def _calculate_readiness(
        self, 
        score: float, 
        adjusted_coverage: float,
        gaps: Dict,
        difficulty_multiplier: float
    ) -> float:
        """Вычисляет скор готовности студента к уровню (0-100)"""
        coverage_component = adjusted_coverage
        score_component = min(score * 100, 100)
        
        high_gaps = gaps.get('high_priority', [])
        gap_component = max(0, 100 - (len(high_gaps) * 10))
        
        readiness = (
            coverage_component * 0.5 +
            score_component * 0.3 +
            gap_component * 0.2
        )
        
        return min(readiness, 100)
    
    def _get_recommendation(self, readiness_score: float, target_level: str) -> str:
        """Генерирует рекомендацию"""
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
        """Генерирует текстовую сводку"""
        lines = [
            f"Оценено {len(evaluations)} профилей",
            f"Средняя готовность: {avg_readiness:.2f}%",
            f"Лучший профиль: {best.profile_name} ({best.readiness_score:.2f}%)",
            f"Рекомендация: {best.recommendation}"
        ]
        return "\n".join(lines)