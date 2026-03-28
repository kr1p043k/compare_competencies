"""
Анализ трендов: какие навыки растут/падают в спросе
"""

from typing import List, Dict, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Анализирует тренды в спросе на навыки.
    Может сравнивать спрос по времени (если будут исторические данные).
    """

    def __init__(self, skill_weights: Dict[str, float]):
        """
        Args:
            skill_weights: Веса навыков на рынке
        """
        self.skill_weights = skill_weights or {}

    def get_emerging_skills(self, min_weight: float = 0.01) -> List[Dict]:
        """
        Возвращает emerging skills (новые, но уже в тренде).
        
        Args:
            min_weight: Минимальный вес для включения
        
        Returns:
            Список навыков с низким весом (потенциально растущие)
        """
        emerging = [
            {
                "skill": skill,
                "weight": weight,
                "potential": "RISING" if "ai" in skill.lower() or "machine" in skill.lower() else "STABLE"
            }
            for skill, weight in self.skill_weights.items()
            if 0 < weight < min_weight * 10
        ]
        
        return sorted(emerging, key=lambda x: x["weight"], reverse=True)

    def get_declining_skills(self) -> List[Dict]:
        """
        Возвращает навыки, спрос на которые падает.
        (пока основано на эвристике - можно улучшить)
        """
        declining = []
        
        # Эвристика: старые технологии
        old_tech = ["flash", "java applet", "coldfusion", "asp.net", "actionscript"]
        
        for skill, weight in self.skill_weights.items():
            if any(old in skill.lower() for old in old_tech):
                declining.append({
                    "skill": skill,
                    "weight": weight,
                    "reason": "OUTDATED"
                })
        
        return declining

    def get_stable_skills(self) -> List[Dict]:
        """
        Возвращает стабильные (high-demand) навыки.
        Это основной набор skills, которые всегда в спросе.
        """
        avg_weight = sum(self.skill_weights.values()) / len(self.skill_weights) if self.skill_weights else 0
        
        stable = [
            {
                "skill": skill,
                "weight": weight,
                "stability": "CRITICAL" if weight > avg_weight * 2 else "STABLE"
            }
            for skill, weight in self.skill_weights.items()
            if weight > avg_weight
        ]
        
        return sorted(stable, key=lambda x: x["weight"], reverse=True)