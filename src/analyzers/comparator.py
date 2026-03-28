"""
TF-IDF сравнение навыков студента с рынком вакансий.
С улучшениями: лучшие параметры, детальное логирование, кэширование.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class CompetencyComparator:
    """
    TF-IDF сравнение навыков студента с рынком вакансий.
    
    Улучшения:
    - Лучшие параметры TfidfVectorizer (min_df=2, max_df=0.95)
    - Кэширование средних векторов
    - Детальное логирование
    - Уверенность/confidence в скоре
    """

    def __init__(self, ngram_range=(1, 1), min_df=2, max_df=0.95):
        """
        Args:
            ngram_range: Диапазон n-gramов - ОБЯЗАТЕЛЬНО (1,1) для unigrams только!
            min_df: Минимум документов где встречается слово
            max_df: Максимум % документов где встречается слово
        """
        # === ЖЁСТКИЕ ПАРАМЕТРЫ - ТОЛЬКО UNIGRAMS ===
        if ngram_range != (1, 1):
            logger.warning(f"⚠️ ngram_range должен быть (1,1), получен {ngram_range}")
            ngram_range = (1, 1)
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),         # ← ЖЁСТКИЕ ТРЕБОВАНИЯ: ТОЛЬКО UNIGRAMS!
            min_df=max(2, int(min_df)), # Минимум в 2+ документах
            max_df=max_df,              # Исключаем частые слова
            lowercase=True,
            stop_words=None,
            token_pattern=r'(?u)\b\w+\b',
            norm='l2',
            sublinear_tf=True,
            max_features=300            # ← Уменьшили до 300 для чистоты
        )
        self.matrix = None
        self.feature_names = None
        self.is_fitted = False
        self.market_vector = None
        self.vocab_size = 0
        
        logger.info("✓ TfidfVectorizer инициализирован с ngram_range=(1,1) ТОЛЬКО!")

    def fit_market(self, vacancies_skills: List[List[str]]) -> bool:
        """
        Обучение TF-IDF на данных вакансий с МАКСИМАЛЬНОЙ очисткой.
        """
        if not vacancies_skills:
            logger.warning("Нет данных для обучения TF-IDF")
            return False

        # === АГРЕССИВНАЯ ОЧИСТКА ДОКУМЕНТОВ ===
        documents = []
        for skills in vacancies_skills:
            if not skills:
                continue
            
            # 1. Удаляем дубликаты
            unique_skills = list(set(s.lower().strip() for s in skills if s))
            
            # 2. ТОЛЬКО unigrams (одно слово)
            unigrams_only = [s for s in unique_skills if len(s.split()) == 1]
            
            # 3. Исключаем очень короткие и очень длинные
            filtered = [s for s in unigrams_only if 2 <= len(s) <= 50]
            
            # 4. Исключаем generic слова
            bad_terms = {"api", "core", "net", "frontend", "backend", "fullstack", "crm"}
            filtered = [s for s in filtered if s not in bad_terms]
            
            if filtered:
                documents.append(" ".join(filtered))

        if not documents:
            logger.warning("Нет валидных документов для TF-IDF")
            return False

        logger.info(f"Обучение TF-IDF на {len(documents)} очищенных документах")
        logger.info(f"  - Параметры: ngram_range={self.vectorizer.ngram_range}")
        logger.info(f"  - min_df=2, max_df=0.95, max_features=300")

        try:
            self.matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.is_fitted = True
            
            # Кэшируем средний вектор
            self.market_vector = np.asarray(self.matrix.mean(axis=0)).reshape(1, -1)
            self.vocab_size = len(self.feature_names)
            
            logger.info(f"✓ TF-IDF обучен:")
            logger.info(f"  - Уникальных навыков: {self.vocab_size}")
            logger.info(f"  - Размер матрицы: {self.matrix.shape}")
            
            # Логируем топ-20
            logger.info(f"\n  Топ-20 навыков по TF-IDF:")
            scores = np.asarray(self.matrix.mean(axis=0)).ravel()
            top_indices = np.argsort(scores)[-20:][::-1]
            for i, idx in enumerate(top_indices, 1):
                skill = self.feature_names[idx]
                score = scores[idx]
                logger.info(f"    {i:2d}. {skill:20s} - {score:.4f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при обучении TF-IDF: {e}")
            return False

    def get_skill_weights(self) -> Dict[str, float]:
        """
        Возвращает веса навыков (средний TF-IDF по всем вакансиям).
        Это показывает важность каждого навыка на рынке.
        """
        if not self.is_fitted or self.matrix is None:
            logger.warning("TF-IDF ещё не обучен")
            return {}

        # Вычисляем средний вес для каждого навыка
        scores = np.asarray(self.matrix.mean(axis=0)).ravel()
        weights = dict(zip(self.feature_names, scores))
        
        # Логируем топ-10 навыков
        top_10 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.debug(f"Топ-10 навыков по TF-IDF:")
        for skill, weight in top_10:
            logger.debug(f"  - {skill}: {weight:.4f}")
        
        return weights

    def compare(self, student_skills: List[str]) -> Tuple[float, float]:
        """
        Сравнивает студента с рынком (косинусное сходство).
        
        Args:
            student_skills: Навыки студента
        
        Returns:
            (score, confidence) где score 0-1, confidence - уверенность в оценке
        """
        if not student_skills:
            return 0.0, 0.0

        if not self.is_fitted:
            logger.warning("TF-IDF не обучен → используется простое пересечение")
            return self._simple_overlap(student_skills)

        try:
            # Преобразуем навыки студента в вектор
            student_doc = " ".join(student_skills)
            student_vec = self.vectorizer.transform([student_doc])

            # Вычисляем косинусное сходство
            score = cosine_similarity(student_vec, self.market_vector)[0][0]
            
            # Рассчитываем уверенность (основано на покрытии)
            coverage = self._calculate_coverage(student_skills)
            confidence = min(score * coverage, 1.0)  # Умножаем на покрытие
            
            return float(round(score, 4)), float(round(confidence, 4))
        
        except Exception as e:
            logger.error(f"Ошибка при сравнении: {e}")
            return 0.0, 0.0

    def _calculate_coverage(self, student_skills: List[str]) -> float:
        """
        Вычисляет долю навыков студента, которые есть на рынке (0-1).
        Используется для определения уверенности в оценке.
        """
        if not self.feature_names or not student_skills:
            return 0.0
        
        student_set = set(s.lower().strip() for s in student_skills)
        market_set = set(s.lower().strip() for s in self.feature_names)
        
        if not market_set:
            return 0.0
        
        coverage = len(student_set & market_set) / len(market_set)
        return coverage

    def _simple_overlap(self, student_skills: List[str]) -> Tuple[float, float]:
        """
        Простое пересечение (fallback если TF-IDF не обучен).
        Возвращает (score, confidence).
        """
        if not self.feature_names:
            return 0.0, 0.0

        student_set = set(s.lower().strip() for s in student_skills)
        market_set = set(s.lower().strip() for s in self.feature_names)

        if not market_set:
            return 0.0, 0.0

        intersection = len(student_set & market_set)
        score = round(intersection / len(market_set), 4)
        confidence = score  # Низкая уверенность для fallback
        
        return float(score), float(confidence)

    def get_stats(self) -> Dict:
        """Возвращает статистику модели"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "vocab_size": self.vocab_size,
            "matrix_shape": self.matrix.shape if self.matrix is not None else None,
            "sparsity": 1 - (self.matrix.nnz / (self.matrix.shape[0] * self.matrix.shape[1])) if self.matrix is not None else None
        }