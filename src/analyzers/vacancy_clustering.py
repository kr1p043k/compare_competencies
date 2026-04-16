"""
Кластеризация вакансий на основе эмбеддингов навыков.
Используется для определения профессиональных ролей.
"""
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.parsing.embedding_loader import get_embedding_model

logger = logging.getLogger(__name__)


class VacancyClusterer:
    """Кластеризует вакансии по эмбеддингам навыков."""

    def __init__(self, n_clusters: int = 30, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.cluster_centers: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.vacancy_ids: List[str] = []
        self.is_fitted = False

    def fit(self, vacancies: List[Dict], level: str = "all") -> "VacancyClusterer":
        """
        Обучает KMeans на эмбеддингах вакансий.
        vacancies: список словарей с ключами 'id', 'skills', 'experience'
        """
        if not vacancies:
            logger.warning("Нет вакансий для кластеризации")
            return self

        if level != "all":
            vacancies = [v for v in vacancies if v.get('experience') == level]

        self.vacancy_ids = [v.get('id', str(i)) for i, v in enumerate(vacancies)]

        embedding_model = get_embedding_model()
        vacancy_embs = []
        for vac in vacancies:
            skills = vac.get('skills', [])
            if not skills:
                emb = np.zeros(embedding_model.get_sentence_embedding_dimension())
            else:
                embs = embedding_model.encode(skills, convert_to_numpy=True, show_progress_bar=False)
                emb = np.mean(embs, axis=0)
            vacancy_embs.append(emb)

        X = np.vstack(vacancy_embs)
        logger.info(f"Кластеризация {len(X)} вакансий уровня '{level}'...")
        self.model.fit(X)
        self.labels = self.model.labels_
        self.cluster_centers = self.model.cluster_centers_
        self.is_fitted = True

        self._save_model(level)
        logger.info(f"✅ Кластеризация завершена. Сохранено {self.n_clusters} кластеров.")
        return self

    def _save_model(self, level: str):
        path = config.DATA_PROCESSED_DIR / f"vacancy_clusters_{level}.pkl"
        data = {
            'model': self.model,
            'labels': self.labels,
            'centers': self.cluster_centers,
            'vacancy_ids': self.vacancy_ids,
            'n_clusters': self.n_clusters
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Модель кластеризации сохранена: {path}")

    def load_model(self, level: str = "all") -> bool:
        path = config.DATA_PROCESSED_DIR / f"vacancy_clusters_{level}.pkl"
        if not path.exists():
            logger.warning(f"Файл модели не найден: {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.labels = data['labels']
        self.cluster_centers = data['centers']
        self.vacancy_ids = data['vacancy_ids']
        self.n_clusters = data['n_clusters']
        self.is_fitted = True
        logger.info(f"Модель кластеризации загружена: {path}")
        return True

    def find_closest_clusters(
        self, student_skills: List[str], top_k: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Находит ближайшие кластеры к профилю студента.
        Возвращает список (cluster_id, similarity).
        """
        if not self.is_fitted:
            return []

        embedding_model = get_embedding_model()
        if not student_skills:
            student_emb = np.zeros(embedding_model.get_sentence_embedding_dimension())
        else:
            embs = embedding_model.encode(student_skills, convert_to_numpy=True, show_progress_bar=False)
            student_emb = np.mean(embs, axis=0)

        similarities = cosine_similarity([student_emb], self.cluster_centers)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_cluster_skills(self, cluster_id: int, vacancies: List[Dict]) -> List[str]:
        """Возвращает список навыков, характерных для кластера."""
        if not self.is_fitted or self.labels is None:
            return []
        cluster_vacs = [v for i, v in enumerate(vacancies) if self.labels[i] == cluster_id]
        all_skills = set()
        for vac in cluster_vacs:
            all_skills.update(vac.get('skills', []))
        return list(all_skills)