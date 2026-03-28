"""
Анализатор пробелов (gaps) между компетенциями студента и рынком.
С улучшениями: детальная статистика, категоризация, confidence.
"""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GapAnalyzer:
    """
    Анализатор пробелов между студентом и рынком.
    
    Улучшения:
    - Категоризация gaps по важности (high/medium/low)
    - Детальная статистика
    - Confidence для каждого gap
    - Рекомендации по приоритизации
    """

    # Пороги для категоризации важности
    HIGH_IMPORTANCE = 0.70    # 70% - критичные навыки
    MEDIUM_IMPORTANCE = 0.30  # 30-70% - желательные
    LOW_IMPORTANCE = 0.0      # 0-30% - опциональные

    def __init__(self, skill_weights: Dict[str, float]):
        """
        Args:
            skill_weights: Веса навыков (важность на рынке)
        """
        self.skill_weights = skill_weights or {}
        self.total_weight = sum(self.skill_weights.values())
        
        if not self.skill_weights:
            logger.warning("GapAnalyzer: получены пустые веса навыков")
        else:
            logger.info(f"GapAnalyzer инициализирован с {len(self.skill_weights)} навыками")

    def analyze_gap(self, student_skills: List[str], top_n: int = 20) -> Dict:
        """
        Анализирует пробелы и возвращает категоризированный результат.
        
        Args:
            student_skills: Навыки студента
            top_n: Количество результатов в каждой категории
        
        Returns:
            Словарь с high/medium/low priority gaps
        """
        if not student_skills:
            student_skills = []

        # Нормализуем навыки студента
        student_set = {s.lower().strip() for s in student_skills if s}
        
        # Находим недостающие навыки с весами
        missing = []
        for skill, weight in self.skill_weights.items():
            if skill.lower().strip() not in student_set:
                # Нормализуем вес к диапазону 0-1
                normalized_weight = weight / self.total_weight if self.total_weight > 0 else 0
                missing.append({
                    "skill": skill,
                    "weight": float(weight),
                    "importance": normalized_weight,
                    "priority": self._get_priority(normalized_weight)
                })

        # Сортируем по весу
        missing_sorted = sorted(missing, key=lambda x: x["weight"], reverse=True)
        
        # Категоризируем
        result = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "total_gaps": len(missing),
            "stats": self._calculate_stats(missing)
        }
        
        for gap in missing_sorted:
            if gap["priority"] == "HIGH":
                if len(result["high_priority"]) < top_n:
                    result["high_priority"].append(gap)
            elif gap["priority"] == "MEDIUM":
                if len(result["medium_priority"]) < top_n:
                    result["medium_priority"].append(gap)
            else:
                if len(result["low_priority"]) < top_n:
                    result["low_priority"].append(gap)
        
        logger.info(f"Gap анализ: {len(result['high_priority'])} высокий + {len(result['medium_priority'])} средний + {len(result['low_priority'])} низкий")
        
        return result

    def coverage(self, student_skills: List[str]) -> Tuple[float, Dict]:
        """
        Доля пок��ытия рынка (weighted).
        
        Returns:
            (coverage_percent, details_dict)
        """
        if not self.skill_weights:
            return 0.0, {}

        student_set = {s.lower().strip() for s in student_skills if s}
        
        covered_weight = sum(
            weight for skill, weight in self.skill_weights.items()
            if skill.lower().strip() in student_set
        )

        coverage = (covered_weight / self.total_weight * 100) if self.total_weight > 0 else 0.0
        
        details = {
            "covered_weight": round(covered_weight, 2),
            "total_weight": round(self.total_weight, 2),
            "coverage_percent": round(coverage, 2),
            "covered_skills_count": len(student_set & set(s.lower().strip() for s in self.skill_weights.keys())),
            "total_market_skills": len(self.skill_weights)
        }
        
        return coverage, details

    def top_market_skills(self, top_n: int = 20) -> List[Dict]:
        """
        Топ-N самых важных навыков на рынке.
        
        Returns:
            Список с метаданными о каждом навыке
        """
        if not self.skill_weights:
            return []

        sorted_skills = sorted(
            self.skill_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = []
        for rank, (skill, weight) in enumerate(sorted_skills[:top_n], 1):
            normalized = weight / self.total_weight if self.total_weight > 0 else 0
            result.append({
                "rank": rank,
                "skill": skill,
                "weight": round(weight, 4),
                "importance": round(normalized, 4),
                "priority": self._get_priority(normalized)
            })
        
        return result

    def get_recommendations(self, student_skills: List[str], gaps: Dict) -> List[str]:
        """
        Генерирует текстовые рекомендации на основе gaps.
        
        Args:
            student_skills: Навыки студента
            gaps: Результат analyze_gap()
        
        Returns:
            Список рекомендаций
        """
        recommendations = []
        
        coverage, _ = self.coverage(student_skills)
        
        if coverage < 30:
            recommendations.append("⚠️ КРИТИЧНО: Покрытие менее 30%. Требуется переквалификация.")
        elif coverage < 50:
            recommendations.append("⚠️ Низкое покрытие (50%). Нужно срочно добавить skills.")
        elif coverage < 70:
            recommendations.append("📈 Среднее покрытие (50-70%). Есть потенциал для роста.")
        else:
            recommendations.append("✅ Хорошее покрытие (>70%). Продолжайте развиваться.")
        
        if gaps["high_priority"]:
            top_3 = [g["skill"] for g in gaps["high_priority"][:3]]
            recommendations.append(f"🔴 Приоритет №1: {', '.join(top_3)}")
        
        if gaps["medium_priority"]:
            top_3 = [g["skill"] for g in gaps["medium_priority"][:3]]
            recommendations.append(f"🟡 Приоритет №2: {', '.join(top_3)}")
        
        return recommendations

    def _get_priority(self, normalized_weight: float) -> str:
        """Определяет приоритет по нормализованному весу"""
        if normalized_weight >= self.HIGH_IMPORTANCE:
            return "HIGH"
        elif normalized_weight >= self.MEDIUM_IMPORTANCE:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_stats(self, missing: List[Dict]) -> Dict:
        """Вычисляет статистику по gaps"""
        if not missing:
            return {}
        
        weights = [g["weight"] for g in missing]
        
        return {
            "avg_weight": round(sum(weights) / len(weights), 4),
            "max_weight": round(max(weights), 4),
            "min_weight": round(min(weights), 4),
            "total_missing_weight": round(sum(weights), 4)
        }