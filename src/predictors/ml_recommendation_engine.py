"""
ML Recommendation Engine v4 (регрессия важности навыков)
Использует RandomForestRegressor для предсказания непрерывного скора важности.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from src.parsing.vacancy_parser import VacancyParser
from src.parsing.skill_normalizer import SkillNormalizer
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src import config

logger = logging.getLogger(__name__)


class MLRecommendationEngine:
    """
    ML-движок для предсказания важности навыков (регрессия).
    Интегрируется с существующими модулями и данными.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.vacancy_parser = VacancyParser()
        self.skill_filter = SkillFilter()
        self.level_analyzer = SkillLevelAnalyzer()

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ])

        self.is_fitted = False
        self.feature_names: List[str] = []
        self.skill_metadata: Dict[str, Dict[str, Any]] = {}
        self.total_vacancies = 0

        if model_path is None:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model_path = config.MODELS_DIR / "skill_importance_regressor.joblib"
        else:
            self.model_path = model_path

    def fit(self, vacancies: List[Dict], test_size: float = 0.2) -> "MLRecommendationEngine":
        if len(vacancies) < 10:
            logger.warning("Недостаточно вакансий для обучения ML-модели")
            self.is_fitted = False
            return self

        logger.info(f"Обучение ML-модели (регрессия) на {len(vacancies)} вакансиях...")

        # 1. Извлекаем частоты и гибридные веса через VacancyParser
        extraction_result = self.vacancy_parser.extract_skills_from_vacancies(vacancies)
        frequencies = extraction_result["frequencies"]
        hybrid_weights = extraction_result.get("hybrid_weights", {})

        # 2. Подготавливаем данные для SkillLevelAnalyzer вручную
        processed_vacancies = []
        for vac in vacancies:
            skills = set()
            # Из key_skills
            for ks in vac.get("key_skills", []):
                name = ks.get("name", "")
                if name:
                    norm = SkillNormalizer.normalize(name)
                    if norm:
                        skills.add(norm)
            # Из description (опционально)
            desc_skills = self.vacancy_parser.extract_skills_from_description(vac.get("description", ""))
            for s in desc_skills:
                norm = SkillNormalizer.normalize(s)
                if norm:
                    skills.add(norm)

            # Обработка experience
            exp_raw = vac.get("experience", {})
            if isinstance(exp_raw, dict):
                exp_name = exp_raw.get("name", "").lower()
                if "junior" in exp_name or "младший" in exp_name:
                    experience = "junior"
                elif "senior" in exp_name or "старший" in exp_name:
                    experience = "senior"
                else:
                    experience = "middle"
            else:
                experience = exp_raw if isinstance(exp_raw, str) else "middle"

            processed_vacancies.append({
                "skills": list(skills),
                "experience": experience
            })

        self.level_analyzer.analyze_vacancies(processed_vacancies)

        # 3. Сохраняем метаданные для каждого навыка
        self.skill_metadata = {}
        for skill, freq in frequencies.items():
            self.skill_metadata[skill] = {
                "frequency": freq,
                "hybrid_weight": hybrid_weights.get(skill, 0.0),
                "level": self.level_analyzer.get_skill_level(skill),
                "category": self._get_skill_category(skill),
                "filtered": self.skill_filter.validate_skills([skill]) != []
            }
        self.total_vacancies = len(vacancies)

        # 4. Формируем обучающую выборку
        all_skills = list(self.skill_metadata.keys())
        X_rows = []
        y_rows = []

        # Целевая переменная: нормализованная важность = лог-частота + 0.3 * hybrid_weight
        max_freq = max(frequencies.values()) if frequencies else 1
        max_log = np.log1p(max_freq)

        for skill in all_skills:
            freq = frequencies[skill]
            hw = hybrid_weights.get(skill, 0.0)
            # Комбинированный скор: лог-частота (0-1) и hybrid_weight (0-1)
            target = (np.log1p(freq) / max_log) * 0.7 + hw * 0.3
            features = self._extract_features(skill)
            X_rows.append(features)
            y_rows.append(target)

        X = pd.DataFrame(X_rows)
        y = np.array(y_rows)
        self.feature_names = X.columns.tolist()

        # --- Разделение на train/test для оценки ---
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y  # fallback

        # 5. Обучение
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        # --- Оценка качества ---
        if len(X_test) > 0:
            y_pred = self.pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logger.info("\n" + "="*60)
            logger.info("Оценка качества регрессионной модели на тестовой выборке")
            logger.info("="*60)
            logger.info(f"Размер тестовой выборки: {len(X_test)}")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")

            # График: предсказанные vs истинные значения
            plt.figure(figsize=(6,5))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlabel("Истинная важность")
            plt.ylabel("Предсказанная важность")
            plt.title("Predicted vs Actual Importance")
            pred_vs_actual_path = config.MODELS_DIR / "pred_vs_actual.png"
            plt.savefig(pred_vs_actual_path, dpi=150)
            plt.close()
            logger.info(f"График предсказаний сохранён: {pred_vs_actual_path}")

            # Распределение остатков
            residuals = y_test - y_pred
            plt.figure(figsize=(6,4))
            sns.histplot(residuals, kde=True, bins=15)
            plt.xlabel("Ошибка предсказания")
            plt.title("Distribution of Residuals")
            resid_path = config.MODELS_DIR / "residuals_dist.png"
            plt.savefig(resid_path, dpi=150)
            plt.close()
            logger.info(f"Распределение остатков сохранено: {resid_path}")

        # --- Важность признаков ---
        feature_importances = self.pipeline.named_steps['reg'].feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        logger.info("\nТоп-10 важных признаков:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Сохраняем модель
        joblib.dump({
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "skill_metadata": self.skill_metadata,
            "total_vacancies": self.total_vacancies
        }, self.model_path)
        logger.info(f"Модель сохранена в {self.model_path}")

        return self

    def load_model(self, path: Optional[Path] = None) -> "MLRecommendationEngine":
        """Загружает ранее обученную модель."""
        model_path = path or self.model_path
        if not model_path.exists():
            logger.error(f"Файл модели не найден: {model_path}")
            return self

        data = joblib.load(model_path)
        self.pipeline = data["pipeline"]
        self.feature_names = data["feature_names"]
        self.skill_metadata = data["skill_metadata"]
        self.total_vacancies = data["total_vacancies"]
        self.is_fitted = True
        logger.info(f"Модель загружена из {model_path}")
        return self

    def predict_skill_impact(
        self,
        student_skills: List[str],
        missing_skills: List[str]
    ) -> List[Tuple[str, float, str]]:
        """
        Предсказывает важность недостающих навыков (непрерывный скор).
        Возвращает список кортежей (навык, скор_важности, объяснение).
        """
        if not self.is_fitted:
            logger.warning("Модель не обучена, возвращаю fallback на частоты")
            return self._fallback_impacts(missing_skills)

        impacts = []
        for skill in missing_skills:
            features = self._extract_features(skill)
            X_skill = pd.DataFrame([features])[self.feature_names]

            try:
                # Предсказываем непрерывное значение важности (0-1)
                pred = self.pipeline.predict(X_skill)[0]
                # Ограничиваем диапазон и переводим в проценты
                score = max(0.0, min(1.0, pred)) * 100
            except Exception as e:
                logger.debug(f"Ошибка предсказания для {skill}: {e}")
                score = self._fallback_score(skill)

            explanation = self._generate_explanation(skill, score / 100)
            impacts.append((skill, round(score, 2), explanation))

        return sorted(impacts, key=lambda x: x[1], reverse=True)[:10]

    def _extract_features(self, skill: str) -> Dict[str, float]:
        """Извлекает признаки для одного навыка."""
        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)
        level = meta.get("level", "middle")
        category = meta.get("category", "other")
        hybrid_weight = meta.get("hybrid_weight", 0.0)

        level_map = {"junior": 1, "middle": 2, "senior": 3, "all_levels": 2}
        category_map = {
            "programming_languages": 5,
            "frameworks": 4,
            "databases": 3,
            "devops": 4,
            "cloud": 4,
            "data_science": 4,
            "frontend": 3,
            "testing": 3,
            "tools": 2,
            "other": 1
        }

        return {
            "frequency": freq,
            "freq_log": np.log1p(freq),
            "hybrid_weight": hybrid_weight,
            "level_encoded": level_map.get(level, 2),
            "category_encoded": category_map.get(category, 1),
            "filtered": 1.0 if meta.get("filtered", False) else 0.0,
        }

    def _get_skill_category(self, skill: str) -> str:
        """Определяет категорию навыка через SkillFilter."""
        categories = self.skill_filter.get_skill_categories([skill])
        for cat, skills in categories.items():
            if skill in skills:
                return cat
        return "other"

    def _fallback_score(self, skill: str) -> float:
        """Fallback оценка на основе частоты и hybrid_weight."""
        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)
        hw = meta.get("hybrid_weight", 0.0)
        max_freq = max((m["frequency"] for m in self.skill_metadata.values()), default=1)
        max_log = np.log1p(max_freq)
        score = (np.log1p(freq) / max_log) * 0.7 + hw * 0.3
        return score * 100

    def _fallback_impacts(self, missing_skills: List[str]) -> List[Tuple[str, float, str]]:
        """Fallback на частоты, если модель не обучена."""
        impacts = []
        for skill in missing_skills:
            score = self._fallback_score(skill)
            impacts.append((skill, round(score, 2), self._generate_explanation(skill, score/100)))
        return sorted(impacts, key=lambda x: x[1], reverse=True)[:10]

    def _generate_explanation(self, skill: str, importance: float) -> str:
        """Генерирует текстовое объяснение для навыка."""
        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)
        level = meta.get("level", "middle")
        category = meta.get("category", "other")
        return (f"Частота: {freq}, уровень: {level}, категория: {category}, "
                f"важность: {importance:.2f}")


# ----------------------------------------------------------------------
# Блок для отладки и тестирования (__main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Отладка MLRecommendationEngine")
    parser.add_argument("--load-raw", action="store_true", help="Загрузить сырые вакансии из data/raw/hh_vacancies_basic.json")
    parser.add_argument("--train", action="store_true", help="Принудительно обучить модель (иначе загрузит готовую)")
    parser.add_argument("--student", type=str, default="base", help="Профиль студента для демонстрации (base, dc, top_dc)")
    args = parser.parse_args()

    engine = MLRecommendationEngine()

    if args.train or not engine.model_path.exists():
        if args.load_raw:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if not raw_file.exists():
                logger.error(f"Файл {raw_file} не найден.")
                sys.exit(1)
            with open(raw_file, 'r', encoding='utf-8') as f:
                vacancies = json.load(f)
            logger.info(f"Загружено {len(vacancies)} сырых вакансий из {raw_file}")
        else:
            logger.warning("Использую синтетические вакансии для демонстрации.")
            vacancies = [
                {"skills": ["python", "sql", "pandas"], "experience": "middle", "description": "Python developer"},
                {"skills": ["python", "docker", "flask"], "experience": "senior", "description": "Senior Python dev"},
                {"skills": ["java", "spring", "sql"], "experience": "middle", "description": "Java backend"},
                {"skills": ["python", "machine learning", "pytorch"], "experience": "senior", "description": "ML engineer"},
                {"skills": ["javascript", "react", "html"], "experience": "junior", "description": "Frontend dev"},
            ]
        engine.fit(vacancies)
    else:
        engine.load_model()

    student_file = config.STUDENTS_DIR / f"{args.student}_competency.json"
    student_skills = []
    if student_file.exists():
        with open(student_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            student_skills = data.get("навыки", [])
        logger.info(f"Навыки студента '{args.student}': {student_skills}")
    else:
        logger.warning(f"Файл студента {student_file} не найден. Использую заглушку ['python', 'sql']")
        student_skills = ["python", "sql"]

    market_skills = list(engine.skill_metadata.keys())
    missing = [s for s in market_skills if s not in student_skills]

    print("\n" + "=" * 60)
    print(f"Рекомендации для профиля '{args.student}' (недостающие навыки):")
    print("=" * 60)

    recommendations = engine.predict_skill_impact(student_skills, missing)
    for skill, score, expl in recommendations:
        print(f"• {skill:<20} важность: {score:>5.1f}% | {expl}")

    print("\n" + "=" * 60)
    print("Топ-10 самых важных рыночных навыков (по модели):")
    print("=" * 60)
    top_skills = sorted(engine.skill_metadata.items(), key=lambda x: x[1]["frequency"], reverse=True)[:10]
    for skill, meta in top_skills:
        print(f"• {skill:<20} частота: {meta['frequency']} | уровень: {meta['level']}")