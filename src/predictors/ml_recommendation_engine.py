"""
НАСТОЯЩИЙ ML Recommendation Engine
Использует RandomForestRegressor для предсказания salary uplift от навыков.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.models.student import StudentProfile
from src.models.vacancy import Vacancy  # предполагаю, что у тебя есть
from src.predictors.recommendation_engine import RecommendationEngine  # наследуем шаблоны

logger = logging.getLogger(__name__)

class MLRecommendationEngine(RecommendationEngine):
    def __init__(self):
        super().__init__()
        self.mlb = MultiLabelBinarizer()
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = None
        logger.info("MLRecommendationEngine инициализирован (RandomForest)")

    def fit(self, vacancies: List[Vacancy]):
        """Обучаем на РЕАЛЬНЫХ вакансиях с salary"""
        if not vacancies:
            raise ValueError("Нужны вакансии с salary")

        # Подготовка данных
        data = []
        for v in vacancies:
            if v.salary is None or v.salary <= 0:
                continue
            data.append({
                'skills': v.skills,           # List[str]
                'salary': float(v.salary)     # target
            })

        df = pd.DataFrame(data)
        
        # One-hot encoding всех навыков
        X = self.mlb.fit_transform(df['skills'])
        self.feature_names = self.mlb.classes_
        
        y = df['salary'].values

        # Train/test split (чтобы не переобучаться)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        
        # Метрика качества (R²)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        logger.info(f"ML модель обучена. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        self.is_fitted = True
        return self

    def predict_skill_impact(self, student_skills: List[str], missing_skills: List[str]) -> List[Tuple[str, float, str]]:
        """
        Для каждого missing_skill считаем uplift = (predicted_salary с навыком) - (predicted_salary без)
        Возвращает: [(skill, uplift_rub, explanation), ...] топ-10
        """
        if not self.is_fitted:
            raise RuntimeError("Сначала вызови .fit()")

        # Базовый вектор студента
        base_vector = self.mlb.transform([student_skills])[0]

        base_salary_pred = self.model.predict([base_vector])[0]

        impacts = []
        for skill in missing_skills:
            if skill not in self.feature_names:
                continue  # навык не встречался в обучении

            # Симулируем добавление навыка
            new_vector = base_vector.copy()
            idx = list(self.feature_names).index(skill)
            new_vector[idx] = 1

            new_salary_pred = self.model.predict([new_vector])[0]
            uplift = new_salary_pred - base_salary_pred

            # Берём объяснение из старых шаблонов (наследуем)
            explanation = self.HARD_SKILL_TEMPLATES.get(skill, 
                           self.SOFT_SKILL_TEMPLATES.get(skill, 
                           f"{skill} — высокочастотный навык на рынке"))

            impacts.append((skill, round(uplift), explanation))

        # Сортируем по uplift descending
        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts[:10]

    def recommend(self, student: StudentProfile, market_vacancies: List[Vacancy]) -> Dict:
        """Полный цикл: gap → ML uplift → рекомендации"""
        # Используем старый comparator/gap_analyzer
        comparator = self.comparator
        comparison = comparator.compare(student, market_vacancies)  # предполагаю, что возвращает Comparison

        missing_hard = [s.name for s in comparison.deficits if s.category == "hard"]
        missing_soft = [s.name for s in comparison.deficits if s.category == "soft"]

        # ML-часть
        ml_impacts = self.predict_skill_impact(student.skills, missing_hard + missing_soft)

        # Формируем финальный отчёт
        recommendations = {
            "ml_top_recommendations": [
                {
                    "skill": skill,
                    "salary_uplift": uplift,
                    "explanation": expl,
                    "learning_path": self.HARD_LEARNING_PATHS.get(skill) or 
                                    self.SOFT_LEARNING_PATHS.get(skill, "Практика + проекты")
                }
                for skill, uplift, expl in ml_impacts
            ],
            "gap_summary": comparison.summary,   # твоя старая структура
            "model_metrics": {
                "n_vacancies_trained": len(market_vacancies),
                "feature_importance_top5": dict(zip(
                    self.feature_names[np.argsort(self.model.feature_importances_)[-5:]][::-1],
                    sorted(self.model.feature_importances_, reverse=True)[:5]
                ))
            }
        }

        logger.info(f"ML-рекомендации для {student.name} готовы. Топ-1: {ml_impacts[0][0]} (+{ml_impacts[0][1]}₽)")
        return recommendations