# src/predictors/recommendation_engine.py
"""
Улучшенный движок рекомендаций — более человечный и контекстный.
Разделяет hard и soft навыки, использует естественный язык.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# Автоматическое определение корня проекта
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.student import StudentProfile
from src.models.comparison import ComparisonReport, GapResult
from src.analyzers.gap_analyzer import GapAnalyzer


class RecommendationEngine:
    """Генерирует естественные, человечные и полезные рекомендации."""

    def __init__(self):
        self.hard_skill_templates = {
            "pytorch": "Освоение PyTorch позволит вам уверенно работать с современными нейронными сетями. Рекомендую начать с официального туториала, затем выполнить 2–3 проекта на Kaggle (например, компьютерное зрение или табличные данные).",
            "docker": "Docker — ключевой инструмент для промышленного развертывания моделей. Начните с официального туториала, создайте Dockerfile для своей модели и изучите docker-compose.",
            "langchain": "LangChain сейчас очень востребован для работы с LLM. Рекомендую пройти официальную документацию и собрать RAG-приложение на своих данных.",
            "fastapi": "FastAPI отлично подходит для создания API вокруг ML-моделей. Начните с базового туториала и создайте простое API для инференса модели.",
            "kubernetes": "Kubernetes необходим для масштабирования ML-систем. После базового знакомства попробуйте развернуть модель с помощью KServe или стандартного Deployment.",
        }

        self.soft_skill_templates = {
            "английский язык": "Английский язык на уровне B2+ существенно расширяет ваши возможности: чтение документации, участие в международных конференциях и общение с зарубежными коллегами. Рекомендую заниматься регулярно (например, 20–30 минут в день) через чтение статей по ИИ и просмотр технических видео.",
            "аналитическое мышление": "Аналитическое мышление — основа работы дата-сайентиста. Развивайте его через разбор реальных кейсов, решение задач на LeetCode (в категории Data Analysis) и обсуждение подходов с коллегами.",
            "системный анализ": "Системный анализ поможет вам лучше понимать требования бизнеса и переводить их в технические задачи. Полезно будет поработать с реальными проектами и научиться составлять грамотные технические задания.",
            "постановка задач разработчикам": "Умение чётко ставить задачи — один из самых ценных навыков тимлида/аналитика. Практикуйтесь в написании детальных ТЗ и проведении grooming-сессий.",
        }

    def generate_recommendations(self, report: ComparisonReport, student: StudentProfile) -> List[Dict]:
        recommendations = []

        # Обрабатываем сначала высокоприоритетные дефициты
        for gap in report.high_demand_gaps[:8]:
            rec = self._create_human_recommendation(gap, student)
            recommendations.append(rec)

        # Добавляем несколько из среднего приоритета
        if len(recommendations) < 7:
            for gap in report.medium_demand_gaps[:5]:
                rec = self._create_human_recommendation(gap, student, priority="medium")
                recommendations.append(rec)

        return recommendations

    def _create_human_recommendation(self, gap: GapResult, student: StudentProfile, priority: str = "high") -> Dict:
        skill = gap.skill.strip()
        skill_lower = skill.lower()

        # Проверяем, является ли навык soft-навыком
        if self._is_soft_skill(skill_lower):
            suggestion = self.soft_skill_templates.get(skill_lower, 
                f"Развитие навыка '{skill}' поможет вам стать более эффективным специалистом.")
            how_to = self._get_soft_learning_path(skill_lower)
        else:
            suggestion = self.hard_skill_templates.get(skill_lower, 
                f"Рекомендуется освоить {skill} — один из наиболее востребованных технических навыков на рынке.")
            how_to = self._get_hard_learning_path(skill_lower)

        priority_text = "Высокий приоритет" if priority == "high" else "Средний приоритет"

        return {
            "skill": skill,
            "frequency": gap.frequency,
            "priority": priority,
            "priority_text": priority_text,
            "suggestion": suggestion,
            "why_important": f"Навык '{skill}' встречается в {gap.frequency} вакансиях. Это один из ключевых требований для роли {student.target_role}.",
            "how_to_learn": how_to,
            "expected_outcome": self._get_expected_outcome(skill, student)
        }

    def _is_soft_skill(self, skill_lower: str) -> bool:
        soft_keywords = ["английский", "мышление", "аналитическое", "системный анализ", "постановка задач", 
                        "коммуникация", "лидерство", "саморазвитие"]
        return any(kw in skill_lower for kw in soft_keywords)

    def _get_hard_learning_path(self, skill_lower: str) -> str:
        paths = {
            "pytorch": "Начните с официального туториала PyTorch, затем выполните 2–3 проекта на Kaggle или Hugging Face.",
            "docker": "Пройдите Docker Tutorial, создайте Dockerfile для своей ML-модели и изучите docker-compose.",
            "langchain": "Изучите официальную документацию, соберите RAG-приложение и поэкспериментируйте с агентами.",
            "fastapi": "Пройдите FastAPI tutorial и создайте API для инференса вашей модели.",
            "kubernetes": "Начните с Kubernetes Basics, затем попробуйте развернуть модель через KServe."
        }
        return paths.get(skill_lower, "Изучите официальную документацию и выполните несколько практических проектов.")

    def _get_soft_learning_path(self, skill_lower: str) -> str:
        paths = {
            "английский язык": "Занимайтесь регулярно (20–40 минут в день): читайте статьи по ИИ на английском, смотрите технические видео и практикуйте speaking с партнёром или в языковом клубе.",
            "аналитическое мышление": "Решайте задачи на LeetCode (Data Analysis), разбирайте реальные кейсы и обсуждайте подходы с коллегами или менторами.",
            "системный анализ": "Практикуйтесь в составлении технических заданий и проведении интервью с заказчиками/стейкхолдерами.",
            "постановка задач разработчикам": "Пишите детальные ТЗ, проводите grooming-сессии и просите обратную связь от разработчиков."
        }
        return paths.get(skill_lower, "Регулярно практикуйте навык в реальных проектах и просите обратную связь.")

    def _get_expected_outcome(self, skill: str, student: StudentProfile) -> str:
        return (f"Освоение '{skill}' заметно повысит вашу конкурентоспособность на рынке и позволит более уверенно решать "
                f"практические задачи в роли {student.target_role}.")


# ====================== САМОСТОЯТЕЛЬНЫЙ ЗАПУСК ДЛЯ ОТЛАДКИ ======================
if __name__ == "__main__":
    print("=== Recommendation Engine — улучшенная человечная версия ===\n")

    # Автоматическое определение путей
    DATA_DIR = PROJECT_ROOT / "data"

    with open(DATA_DIR / "processed" / "competency_mapping.json", encoding='utf-8') as f:
        mapping = json.load(f)

    with open(DATA_DIR / "processed" / "competency_frequency.json", encoding='utf-8') as f:
        market_skills = json.load(f)

    # Можно легко менять профиль для теста
    profile_name = "base"   # поменяйте на "dc" или "top_dc"
    with open(DATA_DIR / "students" / f"{profile_name}_competency.json", encoding='utf-8') as f:
        data = json.load(f)

    student = StudentProfile(
        student_id=profile_name,
        name=profile_name.upper(),
        competencies=data.get("компетенции") or data.get("навыки") or []
    )

    gap_analyzer = GapAnalyzer(mapping)
    report = gap_analyzer.analyze(student, market_skills)

    engine = RecommendationEngine()
    recommendations = engine.generate_recommendations(report, student)

    print(f"Персональные рекомендации для студента: {student.name}\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2}. {rec['priority_text']}: {rec['skill']} ({rec['frequency']} упоминаний)")
        print(f"   → {rec['suggestion']}")
        print(f"   Почему важно: {rec['why_important']}")
        print(f"   Как развивать: {rec['how_to_learn']}")
        print(f"   Ожидаемый результат: {rec['expected_outcome']}")
        print("-" * 100)

    # Сохранение
    output_path = DATA_DIR / "result" / profile_name / f"recommendations_{profile_name}_debug.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)

    print(f"\nРекомендации сохранены в: {output_path}")