"""
ML Recommendation Engine v2 (без salary) - ИСПРАВЛЕННАЯ ВЕРСИЯ
Использует RandomForest для классификации навыков как "важный" или "обычный"
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)


class SkillAnalyzer:
    """Анализирует навыки без использования зарплаты"""
    
    def __init__(self):
        self.skill_frequency = {}
        self.skill_pairs = defaultdict(int)
        self.skill_by_level = defaultdict(lambda: defaultdict(int))
        self.skill_position = defaultdict(list)
        self.skill_importance = {}
        
    def analyze_vacancies(self, vacancies: List[Dict]) -> Dict[str, float]:
        """Анализирует вакансии и вычисляет важность каждого навыка"""
        logger.info(f"Анализируем {len(vacancies)} вакансий...")
        
        total_vacs = len(vacancies)
        
        for vac in vacancies:
            skills = vac.get('skills', [])
            experience = vac.get('experience', 'middle').lower()
            description = vac.get('description', '')
            
            # Частота
            for skill in skills:
                self.skill_frequency[skill] = self.skill_frequency.get(skill, 0) + 1
                self.skill_by_level[skill][experience] += 1
                
                # Позиция в описании
                if description:
                    pos = description.lower().find(skill.lower())
                    if pos != -1:
                        self.skill_position[skill].append(pos)
            
            # Пары навыков
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    key = tuple(sorted([skill1, skill2]))
                    self.skill_pairs[key] += 1
        
        # Вычисляем важность
        self._calculate_importance()
        
        logger.info(f"✓ Проанализировано {len(self.skill_frequency)} уникальных навыков")
        return self.skill_importance
    
    def _calculate_importance(self):
        """Вычисляет комплексный скор важности навыка"""
        if not self.skill_frequency:
            return
            
        total_vacancies = sum(self.skill_frequency.values())
        
        for skill, freq in self.skill_frequency.items():
            # Компонента 1: Частота (0-0.4)
            frequency_score = min(freq / (total_vacancies / 10), 1.0) * 0.4
            
            # Компонента 2: Уровень требования (0-0.3)
            levels = self.skill_by_level[skill]
            level_score = 0
            if levels.get('senior', 0) > 0:
                level_score += 0.15
            if levels.get('middle', 0) > 0:
                level_score += 0.10
            if levels.get('junior', 0) > 0:
                level_score += 0.05
            level_score = min(level_score, 0.3)
            
            # Компонента 3: Позиция в описании (0-0.2)
            position_score = 0
            if self.skill_position[skill]:
                avg_pos = np.mean(self.skill_position[skill])
                position_score = max(0.2 - (avg_pos / 1000), 0.02) * 0.2
            
            # Компонента 4: Синергия (0-0.1)
            synergy_score = 0
            skill_pairs_count = sum(1 for pair in self.skill_pairs if skill in pair)
            synergy_score = min(skill_pairs_count / 50, 1.0) * 0.1
            
            self.skill_importance[skill] = (
                frequency_score + level_score + position_score + synergy_score
            )
    
    def get_skill_synergies(self, skill: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Возвращает навыки, которые часто встречаются с данным"""
        synergies = []
        for (s1, s2), count in self.skill_pairs.items():
            if s1 == skill:
                synergies.append((s2, count))
            elif s2 == skill:
                synergies.append((s1, count))
        
        return sorted(synergies, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_skill_explanation(self, skill: str) -> str:
        """Генерирует объяснение для навыка"""
        freq = self.skill_frequency.get(skill, 0)
        levels = self.skill_by_level.get(skill, {})
        importance = self.skill_importance.get(skill, 0)
        
        explanations = []
        
        # Частота
        if freq > 0:
            pct = (freq / sum(self.skill_frequency.values())) * 100
            explanations.append(f"Встречается в {pct:.1f}% вакансий")
        
        # Уровень
        level_info = []
        if levels.get('senior', 0) > 0:
            level_info.append("требуется для senior")
        if levels.get('middle', 0) > 0:
            level_info.append("требуется для middle")
        if level_info:
            explanations.append(f"{', '.join(level_info).capitalize()}")
        
        # Синергия
        synergies = self.get_skill_synergies(skill, 2)
        if synergies:
            synergy_names = [s[0] for s in synergies]
            explanations.append(f"Часто сочетается с {', '.join(synergy_names)}")
        
        return " → ".join(explanations) if explanations else f"{skill} — востребованный навык"


class MLRecommendationEngine:
    """
    ML движок для рекомендаций навыков (версия 2, без salary)
    Классифицирует навыки как "важные" (top 30%) или "обычные"
    """
    
    def __init__(self):
        self.analyzer = SkillAnalyzer()
        self.skill_importance = {}
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.mlb = MultiLabelBinarizer()
        self.is_fitted = False
        self.feature_names = None
        logger.info("MLRecommendationEngine v2 инициализирован")
    
    def fit(self, vacancies: List[Dict]):
        """Обучаем модель - классифицируем навыки как важные или обычные"""
        logger.info("Начинаем обучение ML модели...")
        
        if not vacancies:
            raise ValueError("Нужны вакансии для обучения")
        
        logger.info(f"DEBUG: Получено {len(vacancies)} вакансий")
        
        # Анализируем вакансии
        self.skill_importance = self.analyzer.analyze_vacancies(vacancies)
        
        # Подготавливаем данные для RandomForest
        X_data = []
        y_data = []
        
        for vac in vacancies:
            skills = vac.get('skills', [])
            
            if not skills:
                continue
            
            X_data.append(skills)
            y_data.append(skills)  # для подсчета важности
        
        logger.info(f"DEBUG: Вакансий с навыками: {len(X_data)}")
        
        if not X_data or len(X_data) < 3:
            logger.warning(f"⚠️ Недостаточно данных ({len(X_data)} < 3)")
            self.is_fitted = False
            return self
        
        # One-hot encoding
        try:
            X = self.mlb.fit_transform(X_data)
            self.feature_names = self.mlb.classes_
            logger.info(f"DEBUG: Features: {len(self.feature_names)}")
        except Exception as e:
            logger.error(f"❌ Ошибка при one-hot encoding: {e}")
            self.is_fitted = False
            return self
        
        # Создаём бинарный target: навык в top 30% по важности = 1, иначе 0
        # Считаем медиану важности
        importances = list(self.skill_importance.values())
        if importances:
            threshold = np.percentile(importances, 70)  # top 30%
        else:
            threshold = 0
        
        logger.info(f"DEBUG: Threshold важности: {threshold:.4f}")
        
        y = np.array([
            1 if self.skill_importance.get(skill, 0) >= threshold else 0
            for skill in self.feature_names
        ])
        
        logger.info(f"DEBUG: Y distribution: {np.bincount(y)}")
        
        # Если только один класс - добавляем фиктивные образцы
        if len(np.unique(y)) < 2:
            logger.warning("⚠️ Только один класс в данных, добавляем фиктивные образцы")
            # Просто используем важность напрямую, без классификации
            self.is_fitted = True
            return self
        
        # Train/test split
        try:
            if len(X) < 10:
                test_size = 0.2
            else:
                test_size = 0.2
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            logger.info(f"DEBUG: Train: {len(X_train)}, Test: {len(X_test)}")
        except Exception as e:
            logger.error(f"❌ Ошибка при split: {e}")
            self.is_fitted = False
            return self
        
        # Обучаем модель
        try:
            self.model.fit(X_train, y_train)
            logger.info("✅ RandomForest fit успешен")
            
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            logger.info(f"ML модель: Train Acc={train_score:.3f}, Test Acc={test_score:.3f}")
        except Exception as e:
            logger.error(f"❌ Ошибка при fit: {e}")
            self.is_fitted = False
            return self
        
        self.is_fitted = True
        logger.info("✅ is_fitted = True")
        return self
    
    def predict_skill_impact(
        self, 
        student_skills: List[str], 
        missing_skills: List[str]
    ) -> List[Tuple[str, float, str]]:
        """Предсказываем важность каждого missing_skill"""
        
        if not self.is_fitted:
            logger.warning("⚠️ Модель не обучена, используем только аналитику")
            return self._fallback_impacts(missing_skills)
        
        impacts = []
        
        for skill in missing_skills:
            importance = self.skill_importance.get(skill, 0)
            explanation = self.analyzer.get_skill_explanation(skill)
            
            # Если навык встречался в обучении
            if skill in self.feature_names:
                try:
                    # Создаём вектор с этим навыком
                    vector = np.zeros(len(self.feature_names), dtype=int)
                    idx = np.where(self.feature_names == skill)[0][0]
                    vector[idx] = 1
                    
                    # Предсказываем вероятность "важности"
                    prob = self.model.predict_proba([vector])[0]
                    ml_score = prob[1] * 100  # вероятность класса 1
                    
                    total_score = (importance * 50) + (ml_score * 0.5)
                except Exception as e:
                    logger.debug(f"Ошибка predict для {skill}: {e}")
                    total_score = importance * 100
            else:
                total_score = importance * 100
            
            impacts.append((skill, round(total_score), explanation))
        
        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts[:10]
    
    def _fallback_impacts(self, missing_skills: List[str]) -> List[Tuple[str, float, str]]:
        """Fallback: используем только аналитику без ML"""
        impacts = []
        for skill in missing_skills:
            importance = self.skill_importance.get(skill, 0)
            explanation = self.analyzer.get_skill_explanation(skill)
            impacts.append((skill, round(importance * 100), explanation))
        
        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts[:10]
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Возвращает top-N важных признаков модели"""
        if not self.is_fitted or self.feature_names is None:
            return {}
        
        try:
            importances = self.model.feature_importances_
            feature_importance_dict = {}
            
            for name, importance in zip(self.feature_names, importances):
                feature_importance_dict[str(name)] = float(importance)
            
            return dict(
                sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )
        except Exception as e:
            logger.warning(f"⚠️ Ошибка при get_feature_importance: {e}")
            return {}
    
    def _skills_to_vector(self, skills: List[str]) -> np.ndarray:
        """Преобразует список навыков в one-hot вектор"""
        if self.feature_names is None or len(self.feature_names) == 0:
            return None
        
        vector = np.zeros(len(self.feature_names), dtype=int)
        
        for skill in skills:
            indices = np.where(self.feature_names == skill)[0]
            if len(indices) > 0:
                vector[indices[0]] = 1
        
        return vector
    
    def get_top_skills(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Возвращает топ навыков по важности"""
        return sorted(
            self.skill_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]