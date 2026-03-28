import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class CompetencyComparator:
    """
    TF-IDF сравнение навыков студента с рынком вакансий.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            stop_words=None
        )
        self.matrix = None
        self.feature_names = None
        self.is_fitted = False

    def fit_market(self, vacancies_skills: List[List[str]]) -> bool:
        """Обучение TF-IDF на данных вакансий."""
        if not vacancies_skills:
            logger.warning("Нет данных для обучения TF-IDF")
            return False

        documents = [" ".join(skills) for skills in vacancies_skills if skills]

        if not documents:
            logger.warning("Нет валидных документов для TF-IDF")
            return False

        self.matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True

        logger.info(f"TF-IDF обучен на {len(documents)} документах")
        return True

    def get_skill_weights(self) -> Dict[str, float]:
        """Возвращает веса навыков (средний TF-IDF)."""
        if not self.is_fitted or self.matrix is None:
            logger.warning("TF-IDF ещё не обучен")
            return {}

        scores = np.asarray(self.matrix.mean(axis=0)).ravel()
        return dict(zip(self.feature_names, scores))

    def compare(self, student_skills: List[str]) -> float:
        """Сходство студента с рынком (0.0 — 1.0)"""
        if not student_skills:
            return 0.0

        if not self.is_fitted:
            logger.warning("TF-IDF не обучен → fallback на простое пересечение")
            return self._simple_overlap(student_skills)

        student_doc = " ".join(student_skills)
        student_vec = self.vectorizer.transform([student_doc])

        # Преобразуем средний вектор рынка в плотный массив
        market_vec = np.asarray(self.matrix.mean(axis=0)).reshape(1, -1)

        score = cosine_similarity(student_vec, market_vec)[0][0]
        return float(round(score, 4))

    def _simple_overlap(self, student_skills: List[str]) -> float:
        """Простое пересечение (fallback)"""
        if not self.feature_names:
            return 0.0

        student_set = set(s.lower().strip() for s in student_skills)
        market_set = set(s.lower().strip() for s in self.feature_names)

        if not market_set:
            return 0.0

        intersection = len(student_set & market_set)
        return round(intersection / len(market_set), 4)