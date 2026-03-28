from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class GapAnalyzer:
    """
    Анализатор пробелов (gaps) между компетенциями студента и рынком.
    """

    def __init__(self, skill_weights: Dict[str, float]):
        self.skill_weights = skill_weights or {}
        if not self.skill_weights:
            logger.warning("GapAnalyzer: получены пустые веса навыков")

    def analyze_gap(self, student_skills: List[str], top_n: int = 20) -> List[Dict]:
        """Возвращает список недостающих навыков, отсортированных по важности."""
        if not student_skills:
            student_skills = []

        student_set = {s.lower().strip() for s in student_skills}

        missing = []
        for skill, weight in self.skill_weights.items():
            if skill.lower().strip() not in student_set:
                missing.append({
                    "skill": skill,
                    "weight": float(weight)
                })

        missing_sorted = sorted(missing, key=lambda x: x["weight"], reverse=True)
        return missing_sorted[:top_n]

    def coverage(self, student_skills: List[str]) -> float:
        """Доля покрытия рынка (weighted)."""
        if not self.skill_weights:
            return 0.0

        student_set = {s.lower().strip() for s in student_skills}
        total_weight = sum(self.skill_weights.values())

        if total_weight == 0:
            return 0.0

        covered_weight = sum(
            weight for skill, weight in self.skill_weights.items()
            if skill.lower().strip() in student_set
        )

        return covered_weight / total_weight

    def top_market_skills(self, top_n: int = 20) -> List[str]:
        """Топ-N самых важных навыков на рынке."""
        if not self.skill_weights:
            return []

        sorted_skills = sorted(self.skill_weights.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in sorted_skills[:top_n]]