import logging
from typing import List, Dict, Optional
from pathlib import Path

from src.analyzers.comparator import CompetencyComparator
from src.analyzers.gap_analyzer import GapAnalyzer
from src.models.student import StudentProfile

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Движок рекомендаций с TF‑IDF анализом и генерацией естественного языка.
    """

    # Шаблоны для hard навыков
    HARD_SKILL_TEMPLATES = {
        "pytorch": "Освоение PyTorch позволит вам уверенно работать с современными нейронными сетями.",
        "docker": "Docker — ключевой инструмент для промышленного развертывания моделей.",
        "langchain": "LangChain сейчас очень востребован для работы с LLM.",
        "fastapi": "FastAPI отлично подходит для создания API вокруг ML-моделей.",
        "kubernetes": "Kubernetes необходим для масштабирования ML-систем.",
    }

    # Шаблоны для soft навыков
    SOFT_SKILL_TEMPLATES = {
        "английский язык": "Английский язык на уровне B2+ существенно расширяет ваши возможности: чтение документации, участие в международных конференциях.",
        "аналитическое мышление": "Аналитическое мышление — основа работы дата-сайентиста.",
        "системный анализ": "Системный анализ поможет вам лучше понимать требования бизнеса.",
        "постановка задач разработчикам": "Умение чётко ставить задачи — один из самых ценных навыков тимлида/аналитика.",
    }

    # Пути обучения для hard навыков
    HARD_LEARNING_PATHS = {
        "pytorch": "Начните с официального туториала PyTorch, затем выполните 2–3 проекта на Kaggle.",
        "docker": "Пройдите Docker Tutorial, создайте Dockerfile для своей ML-модели и изучите docker-compose.",
        "langchain": "Изучите официальную документацию, соберите RAG-приложение.",
        "fastapi": "Пройдите FastAPI tutorial и создайте API для инференса вашей модели.",
        "kubernetes": "Начните с Kubernetes Basics, затем попробуйте развернуть модель через KServe.",
    }

    # Пути обучения для soft навыков
    SOFT_LEARNING_PATHS = {
        "английский язык": "Занимайтесь регулярно (20–40 минут в день): читайте статьи по ИИ на английском, смотрите технические видео.",
        "аналитическое мышление": "Решайте задачи на LeetCode (Data Analysis), разбирайте реальные кейсы.",
        "системный анализ": "Практикуйтесь в составлении технических заданий и проведении интервью.",
        "постановка задач разработчикам": "Пишите детальные ТЗ, проводите grooming-сессии.",
    }

    def __init__(self):
        self.comparator = CompetencyComparator()
        self.is_fitted = False

    def fit(self, vacancies_skills: List[List[str]]):
        """Обучение на рынке вакансий (TF‑IDF)."""
        if not vacancies_skills:
            logger.warning("Нет данных вакансий для обучения")
            return
        self.comparator.fit_market(vacancies_skills)
        self.is_fitted = True
        logger.info("RecommendationEngine обучен")

    def analyze(self, student_skills: List[str]) -> Dict:
        """
        Базовый анализ студента: match_score, coverage, missing_skills, top_market_skills.
        """
        if not self.is_fitted:
            logger.warning("Engine не обучен")
            return {}

        score = self.comparator.compare(student_skills)
        skill_weights = self.comparator.get_skill_weights()
        gap_analyzer = GapAnalyzer(skill_weights)
        coverage = gap_analyzer.coverage(student_skills)
        missing = gap_analyzer.analyze_gap(student_skills, top_n=20)
        top_market = gap_analyzer.top_market_skills(20)

        recommendations = self._build_basic_recommendations(missing)

        return {
            "match_score": round(score, 4),
            "coverage": round(coverage, 4),
            "missing_skills": missing,
            "top_market_skills": top_market,
            "recommendations": recommendations
        }

    def generate_human_recommendations(self,
                                       student_skills: List[str],
                                       student_profile: Optional[StudentProfile] = None) -> List[Dict]:
        """
        Генерирует человеко‑читаемые рекомендации на основе TF‑IDF анализа.
        """
        analysis = self.analyze(student_skills)
        if not analysis:
            return []

        missing = analysis["missing_skills"]
        match_score = analysis["match_score"]
        coverage = analysis["coverage"]

        human_recs = []
        for item in missing[:10]:
            skill = item["skill"]
            weight = item["weight"]

            priority = self._define_priority(weight)
            priority_text = "Высокий" if priority == "high" else "Средний" if priority == "medium" else "Низкий"

            # Определяем тип навыка и подбираем шаблоны
            is_soft = self._is_soft_skill(skill)
            suggestion = self._get_suggestion(skill, is_soft)
            why_important = self._why_important(skill, weight, analysis["top_market_skills"])
            how_to_learn = self._get_learning_path(skill, is_soft)
            expected_outcome = self._get_expected_outcome(skill, student_profile)

            human_recs.append({
                "skill": skill,
                "weight": weight,
                "priority": priority,
                "priority_text": priority_text,
                "suggestion": suggestion,
                "why_important": why_important,
                "how_to_learn": how_to_learn,
                "expected_outcome": expected_outcome,
                "market_frequency": int(weight * 100) if weight else 0,
            })

        # Добавляем общую метрику
        summary = {
            "match_score": match_score,
            "coverage": coverage,
            "recommendations": human_recs
        }
        return summary

    # ---------- Вспомогательные методы ----------
    def _is_soft_skill(self, skill_lower: str) -> bool:
        soft_keywords = ["английский", "мышление", "аналитическое", "системный анализ",
                         "постановка задач", "коммуникация", "лидерство", "саморазвитие"]
        return any(kw in skill_lower for kw in soft_keywords)

    def _define_priority(self, weight: float) -> str:
        if weight > 0.05:
            return "high"
        elif weight > 0.02:
            return "medium"
        else:
            return "low"

    def _get_suggestion(self, skill: str, is_soft: bool) -> str:
        skill_lower = skill.lower()
        if is_soft:
            return self.SOFT_SKILL_TEMPLATES.get(skill_lower,
                f"Развитие навыка '{skill}' поможет вам стать более эффективным специалистом.")
        else:
            return self.HARD_SKILL_TEMPLATES.get(skill_lower,
                f"Рекомендуется освоить {skill} — один из наиболее востребованных технических навыков на рынке.")

    def _why_important(self, skill: str, weight: float, top_market: List[Dict]) -> str:
        # Вычисляем примерное количество упоминаний
        freq = int(weight * 100) if weight else 0
        if freq > 10:
            return f"Навык '{skill}' встречается в {freq}% вакансий. Это критически важный навык для вашей роли."
        elif freq > 5:
            return f"Навык '{skill}' востребован в {freq}% вакансий. Его освоение повысит вашу конкурентоспособность."
        else:
            return f"Навык '{skill}' встречается в {freq}% вакансий, но может быть важным в специализированных проектах."

    def _get_learning_path(self, skill: str, is_soft: bool) -> str:
        skill_lower = skill.lower()
        if is_soft:
            return self.SOFT_LEARNING_PATHS.get(skill_lower,
                "Регулярно практикуйте навык в реальных проектах и просите обратную связь.")
        else:
            return self.HARD_LEARNING_PATHS.get(skill_lower,
                "Изучите официальную документацию и выполните несколько практических проектов.")

    def _get_expected_outcome(self, skill: str, student_profile: Optional[StudentProfile]) -> str:
        role = student_profile.target_role if student_profile else "целевой роли"
        return (f"Освоение '{skill}' заметно повысит вашу конкурентоспособность на рынке "
                f"и позволит более уверенно решать практические задачи в {role}.")

    def _build_basic_recommendations(self, missing_skills: List[Dict]) -> List[Dict]:
        """Сохраняем базовый список рекомендаций для обратной совместимости."""
        recommendations = []
        for item in missing_skills[:10]:
            skill = item["skill"]
            weight = item["weight"]
            priority = self._define_priority(weight)
            recommendations.append({
                "skill": skill,
                "priority": priority,
                "reason": f"Высокая важность на рынке ({round(weight, 4)})"
            })
        return recommendations