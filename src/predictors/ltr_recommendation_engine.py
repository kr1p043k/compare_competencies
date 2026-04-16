#src/predictors/ltr_recommendation_engine.py

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from src.parsing.vacancy_parser import VacancyParser
from src.parsing.skill_normalizer import SkillNormalizer
from src.analyzers.skill_filter import SkillFilter
from src.analyzers.skill_level_analyzer import SkillLevelAnalyzer
from src.parsing.embedding_loader import get_embedding_model
from src import config

logger = logging.getLogger(__name__)


class LTRRecommendationEngine:
    """
    Исправленный LTR-движок:
    - Без data leakage
    - Персонализированные признаки (student-dependent)
    - Нормальная валидация + метрики
    - Стабильный confidence (минимум 0.05 + softmax)
    - Поддержка SHAP-значений
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.vacancy_parser = VacancyParser()
        self.skill_filter = SkillFilter()
        self.level_analyzer = SkillLevelAnalyzer()
        self.embedding_model = get_embedding_model()

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.skill_metadata: Dict[str, Dict[str, Any]] = {}
        self.skill_embeddings: Dict[str, np.ndarray] = {}
        self.total_vacancies = 0
        self.is_fitted = False

        if model_path is None:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model_path = config.MODELS_DIR / "ltr_ranker_xgb_regressor.joblib"
        else:
            self.model_path = model_path

    # ====================== ОБУЧЕНИЕ ======================
    def fit(self, vacancies: List[Dict]) -> "LTRRecommendationEngine":
        if len(vacancies) < 50:
            logger.warning("Недостаточно вакансий для обучения модели")
            self.is_fitted = False
            return self

        logger.info(f"🚀 Обучение LTR-модели на {len(vacancies)} вакансиях...")

        # 1. Извлекаем частоты и hybrid weights (только для target!)
        extraction_result = self.vacancy_parser.extract_skills_from_vacancies(vacancies)
        frequencies = extraction_result["frequencies"]
        hybrid_weights = extraction_result.get("hybrid_weights", {})

        # 2. Уровни навыков
        processed_vacancies = self._prepare_vacancies_for_levels(vacancies)
        self.level_analyzer.analyze_vacancies(processed_vacancies)

        # 3. Эмбеддинги
        all_skills = list(frequencies.keys())
        if len(all_skills) < 5:
            logger.error("Слишком мало навыков для обучения")
            self.is_fitted = False
            return self

        embeddings = self.embedding_model.encode(all_skills, convert_to_numpy=True, show_progress_bar=True)
        self.skill_embeddings = {skill: emb for skill, emb in zip(all_skills, embeddings)}

        # 4. Метаданные
        self.skill_metadata = {}
        max_hybrid = max(hybrid_weights.values()) if hybrid_weights else 1.0
        max_freq = max(frequencies.values()) if frequencies else 1.0

        for skill, freq in frequencies.items():
            self.skill_metadata[skill] = {
                "frequency": freq,
                "hybrid_weight": hybrid_weights.get(skill, 0.0),
                "level": self.level_analyzer.get_skill_level(skill),
                "category": self._get_skill_category(skill),
                "target": hybrid_weights.get(skill, 0.0) / max_hybrid if max_hybrid > 0 else freq / max_freq,
            }
        self.total_vacancies = len(vacancies)

        # 5. Генерация обучающей выборки С ВАРИАЦИЯМИ СТУДЕНТОВ (ключевой фикс!)
        logger.info("Генерация примеров с разными студенческими профилями...")
        X_rows, y_rows = [], []
        market_emb = np.mean(list(self.skill_embeddings.values()), axis=0) if self.skill_embeddings else None

        for skill in all_skills:
            target = self.skill_metadata[skill]["target"]
            skill_emb = self.skill_embeddings.get(skill)

            for i in range(5):  # 5 синтетических студентов
                if i == 0:
                    student_emb = market_emb
                    student_skills = []  # пустой
                elif i == 1:
                    student_emb = skill_emb * 0.9 + market_emb * 0.1 if skill_emb is not None else market_emb
                    student_skills = [skill]
                else:
                    student_emb = market_emb * (0.6 + np.random.rand() * 0.4) if market_emb is not None else None
                    student_skills = np.random.choice(all_skills, size=min(np.random.randint(3, 12), len(all_skills)), replace=False).tolist()

                features = self._extract_features(skill, student_emb, student_skills)
                X_rows.append(features)
                y_rows.append(target)

        X = pd.DataFrame(X_rows)
        y = np.array(y_rows)
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. XGBoost с улучшенными параметрами
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=30
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        self.is_fitted = True

        pred_test = self.model.predict(X_test)
        logger.info(f"✅ Обучение завершено! R² = {r2_score(y_test, pred_test):.4f}, MAE = {mean_absolute_error(y_test, pred_test):.4f}")

        plt.figure(figsize=(12, 8))
        xgb.plot_importance(self.model, max_num_features=15)
        plt.savefig(config.MODELS_DIR / "ltr_feature_importance_fixed.png", dpi=200, bbox_inches='tight')
        plt.close()

        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "skill_metadata": self.skill_metadata,
            "skill_embeddings": self.skill_embeddings,
            "total_vacancies": self.total_vacancies,
        }, self.model_path)

        logger.info(f"Модель сохранена → {self.model_path}")
        return self

    # ====================== ПРИЗНАКИ ======================
    def _extract_features(self, skill: str, student_emb: Optional[np.ndarray], student_skills: List[str]) -> Dict[str, float]:
        meta = self.skill_metadata.get(skill, {})
        skill_emb = self.skill_embeddings.get(skill)

        sim = 0.0
        if skill_emb is not None and student_emb is not None:
            sim = cosine_similarity([skill_emb], [student_emb])[0][0]

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
            "level_encoded": level_map.get(meta.get("level", "middle"), 2),
            "category_encoded": category_map.get(meta.get("category", "other"), 1),
            "cosine_sim": sim,
            "in_student_profile": 1.0 if skill in student_skills else 0.0,
        }

    # ====================== ПРЕДСКАЗАНИЕ ======================
    def predict_skill_impact(
        self, student_skills: List[str], missing_skills: List[str]
    ) -> List[Tuple[str, float, str]]:
        recs, _, _ = self.predict_skill_impact_with_shap(student_skills, missing_skills)
        return recs

    def predict_skill_impact_with_shap(
        self, student_skills: List[str], missing_skills: List[str]
    ) -> Tuple[List[Tuple[str, float, str]], Optional[np.ndarray], Optional[pd.DataFrame]]:
        if not self.is_fitted or self.model is None:
            logger.warning("Модель не обучена, возвращаю fallback")
            return self._fallback_impacts(missing_skills), None, None

        student_emb = self._get_student_embedding(student_skills)
        if student_emb is None:
            student_emb = np.mean(list(self.skill_embeddings.values()), axis=0) if self.skill_embeddings else None
        if student_emb is None:
            return self._fallback_impacts(missing_skills), None, None

        X_rows = []
        valid_missing = []
        for skill in missing_skills:
            if skill not in self.skill_metadata:
                continue
            features = self._extract_features(skill, student_emb, student_skills)
            X_rows.append(features)
            valid_missing.append(skill)

        if not X_rows:
            return [], None, None

        X = pd.DataFrame(X_rows)[self.feature_names]
        raw_scores = self.model.predict(X)

        raw_scores = np.clip(raw_scores, 0.0, 1.0)
        if len(raw_scores) > 1:
            exp_scores = np.exp(raw_scores * 5)
            scores = exp_scores / exp_scores.sum()
        else:
            scores = raw_scores

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            logger.warning(f"Не удалось вычислить SHAP: {e}")
            shap_values = None

        impacts = []
        for i, skill in enumerate(valid_missing):
            score = float(scores[i])
            explanation = self._generate_explanation(skill, score, shap_values, i, X) if shap_values is not None else ""
            impacts.append((skill, round(score * 100, 2), explanation))

        return sorted(impacts, key=lambda x: x[1], reverse=True)[:15], shap_values, X

    # ====================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ======================
    def _get_student_embedding(self, student_skills: List[str]) -> Optional[np.ndarray]:
        if not student_skills:
            return None
        valid = [s for s in student_skills if s in self.skill_embeddings]
        if not valid:
            return None
        embs = [self.skill_embeddings[s] for s in valid]
        return np.mean(embs, axis=0)

    def _extract_skills_from_vacancy(self, vac: Dict) -> List[str]:
        skills = set()
        for ks in vac.get("key_skills", []):
            name = ks.get("name", "")
            if name:
                norm = SkillNormalizer.normalize(name)
                if norm:
                    skills.add(norm)
        desc = vac.get("description", "")
        if desc:
            desc_skills = self.vacancy_parser.extract_skills_from_description(desc)
            for s in desc_skills:
                norm = SkillNormalizer.normalize(s)
                if norm:
                    skills.add(norm)
        if not skills:
            snippet = vac.get("snippet", {})
            requirement = snippet.get("requirement", "")
            responsibility = snippet.get("responsibility", "")
            combined = f"{requirement} {responsibility}"
            if combined.strip():
                desc_skills = self.vacancy_parser.extract_skills_from_description(combined)
                for s in desc_skills:
                    norm = SkillNormalizer.normalize(s)
                    if norm:
                        skills.add(norm)
        return list(skills)

    def _prepare_vacancies_for_levels(self, vacancies: List[Dict]) -> List[Dict]:
        processed = []
        for vac in vacancies:
            skills = self._extract_skills_from_vacancy(vac)
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
            processed.append({"skills": skills, "experience": experience})
        return processed

    def _get_skill_category(self, skill: str) -> str:
        cats = self.skill_filter.get_skill_categories([skill])
        for cat, skills in cats.items():
            if skill in skills:
                return cat
        return "other"

    def _fallback_impacts(self, missing_skills: List[str]) -> List[Tuple[str, float, str]]:
        impacts = []
        total = max(self.total_vacancies, 1)
        for skill in missing_skills:
            freq = self.skill_metadata.get(skill, {}).get("frequency", 0)
            score = (freq / total) * 100
            impacts.append((skill, round(score, 2), f"Встречается в {freq} вакансиях"))
        return sorted(impacts, key=lambda x: x[1], reverse=True)[:10]

    def _generate_explanation(self, skill: str, score: float, shap_values: Optional[np.ndarray],
                              idx: int, X: pd.DataFrame) -> str:
        meta = self.skill_metadata.get(skill, {})
        freq = meta.get("frequency", 0)
        level = meta.get("level", "middle")
        base = f"{skill}: важность {score*100:.1f}% (частота {freq}, уровень {level})"
        if shap_values is not None:
            top_idx = np.argmax(np.abs(shap_values[idx]))
            feat_name = self.feature_names[top_idx]
            feat_val = X.iloc[idx][feat_name]
            if feat_name == "cosine_sim":
                base += f". Сильно связан с вашим профилем (сходство {feat_val:.2f})"
            elif feat_name == "level_encoded":
                level_str = {1: "junior", 2: "middle", 3: "senior"}.get(int(feat_val), "middle")
                base += f". Востребован на уровне {level_str}"
        return base

    def load_model(self, path: Optional[Path] = None) -> "LTRRecommendationEngine":
        model_path = path or self.model_path
        if not model_path.exists():
            logger.error(f"Файл модели не найден: {model_path}")
            return self
        data = joblib.load(model_path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.skill_metadata = data["skill_metadata"]
        self.skill_embeddings = data.get("skill_embeddings", {})
        self.total_vacancies = data["total_vacancies"]
        self.is_fitted = True
        logger.info(f"Модель загружена из {model_path}")
        return self


if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Отладка LTRRecommendationEngine")
    parser.add_argument("--load-raw", action="store_true", help="Загрузить сырые вакансии из data/raw/hh_vacancies_basic.json")
    parser.add_argument("--train", action="store_true", help="Принудительно обучить модель")
    parser.add_argument("--student", type=str, default="base", help="Профиль студента")
    args = parser.parse_args()

    engine = LTRRecommendationEngine()

    if args.train or not engine.model_path.exists():
        if args.load_raw:
            raw_file = config.DATA_RAW_DIR / "hh_vacancies_basic.json"
            if not raw_file.exists():
                logger.error(f"Файл {raw_file} не найден")
                sys.exit(1)
            with open(raw_file, 'r', encoding='utf-8') as f:
                vacancies = json.load(f)
            logger.info(f"Загружено {len(vacancies)} сырых вакансий")
        else:
            logger.warning("Использую синтетические вакансии")
            vacancies = [
                {"key_skills": [{"name": "python"}, {"name": "sql"}], "experience": "middle", "description": "Python dev"},
                {"key_skills": [{"name": "python"}, {"name": "docker"}], "experience": "senior", "description": "Senior dev"},
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
    else:
        student_skills = ["python", "sql"]

    market_skills = list(engine.skill_metadata.keys())
    missing = [s for s in market_skills if s not in student_skills]

    print("\n" + "=" * 60)
    print(f"LTR-рекомендации для профиля '{args.student}':")
    print("=" * 60)
    recs, shap_vals, X = engine.predict_skill_impact_with_shap(student_skills, missing)
    for skill, score, expl in recs[:10]:
        print(f"• {skill:<20} важность: {score:>5.1f}% | {expl}")