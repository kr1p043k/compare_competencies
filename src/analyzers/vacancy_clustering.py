"""
Кластеризация вакансий на основе эмбеддингов навыков.
Используется для определения профессиональных ролей.
Поддерживает KMeans с подбором числа кластеров по silhouette_score,
а также HDBSCAN как fallback (с обработкой ошибок метрики).
"""
import logging
import pickle
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from src.parsing.skill_normalizer import SkillNormalizer
from src import config
from src.parsing.embedding_loader import get_embedding_model

logger = logging.getLogger(__name__)

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.info("HDBSCAN не установлен, будет использоваться только KMeans")


class VacancyClusterer:
    """Кластеризует вакансии по эмбеддингам навыков с автоопределением числа кластеров."""

    def __init__(
        self,
        n_clusters: int = 10,
        min_clusters: int = 2,
        max_clusters: int = 40,
        random_state: int = 42,
        use_hdbscan_fallback: bool = True,
        min_cluster_size: int = 5,
    ):
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.use_hdbscan_fallback = use_hdbscan_fallback and HDBSCAN_AVAILABLE
        self.min_cluster_size = min_cluster_size

        self.model = None
        self.clusterer_type = "kmeans"
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.vacancy_ids: List[str] = []
        self.vacancy_skills: List[List[str]] = []
        self.is_fitted = False
        self.label_to_center_idx: Dict[int, int] = {}

    def _compute_embeddings(self, vacancies: List[Dict]) -> np.ndarray:
        embedding_model = get_embedding_model()
        # Собираем тексты: все навыки через запятую
        texts = []
        empty_skills_count = 0
        for skills in self.vacancy_skills:
            if not skills:
                # если навыков нет, используем случайный шум или нулевой вектор
                texts.append("")
                empty_skills_count += 1
            else:
                # Нормализуем и соединяем
                clean_skills = [SkillNormalizer.normalize(s) for s in skills]
                clean_skills = [s for s in clean_skills if s]  # убираем пустые
                if not clean_skills:
                    texts.append("")
                    empty_skills_count += 1
                else:
                    texts.append(", ".join(clean_skills))
        
        # Эмбеддинги для всех текстов разом (быстрее)
        embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Диагностика
        mean_vec = np.mean(embeddings, axis=0)
        std_vec = np.std(embeddings, axis=0)
        logger.info(f"Эмбеддинги: shape={embeddings.shape}, mean={mean_vec[:3].tolist()}..., std={std_vec[:3].tolist()}...")
        logger.info(f"Пустых навыков: {empty_skills_count} из {len(vacancies)}")
        
        # L2-нормализация
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def fit(self, vacancies: List[Dict], level: str = "all") -> "VacancyClusterer":
        if not vacancies:
            logger.warning("Нет вакансий для кластеризации")
            return self

        n_samples = len(vacancies)
        if n_samples < 10:
            logger.warning(
                f"Слишком мало вакансий ({n_samples}) для кластеризации уровня {level}"
            )
            self.is_fitted = False
            return self

        self.vacancy_ids = [v.get('id', str(i)) for i, v in enumerate(vacancies)]
        self.vacancy_skills = [v.get('skills', []) for v in vacancies]

        X = self._compute_embeddings(vacancies)
        logger.info(f"Кластеризация {len(X)} вакансий уровня '{level}'...")

        best_k = self.n_clusters
        best_score = -1
        best_labels = None
        best_model = None

        # Для малого числа вакансий уменьшаем max_k
        max_k = min(self.max_clusters, max(3, int(np.sqrt(n_samples))))

        logger.info(
            f"Параметры KMeans: n_clusters={self.n_clusters}, "
            f"min_clusters={self.min_clusters}, max_clusters={max_k}, "
            f"random_state={self.random_state}"
        )

        best_score = -1
        no_improve = 0
        for k in range(max(2, self.min_clusters), max_k + 1):
            if k >= n_samples:
                continue
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init='auto',
                max_iter=300
            )
            labels = kmeans.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            # Фильтр маленьких кластеров
            counts = np.bincount(labels)
            if np.any(counts < self.min_cluster_size):
                continue
            try:
                score = silhouette_score(X, labels, metric='cosine')
            except Exception:
                continue
            logger.debug(f"K={k}, silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_model = kmeans
                no_improve = 0
            else:
                no_improve += 1
            # early stopping
            if no_improve >= 3:
                break

        if best_score < 0.1:
            logger.warning(
                f"Очень низкий silhouette ({best_score:.3f}) — кластеры могут быть шумными"
            )

        # Если silhouette_score плохой, пробуем HDBSCAN с метрикой cosine
        use_hdbscan = False
        if best_score < 0.2 and self.use_hdbscan_fallback and n_samples >= self.min_cluster_size * 2:
            logger.info(
                f"Silhouette score низкий ({best_score:.3f}), пробуем HDBSCAN с cosine..."
            )
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric='cosine',
                    core_dist_n_jobs=-1
                )
                labels = clusterer.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters >= 2:
                    use_hdbscan = True
                    self.model = clusterer
                    self.clusterer_type = "hdbscan"
                    self.labels_ = labels
                    unique_labels = set(labels)
                    unique_labels.discard(-1)
                    centers = []
                    self.label_to_center_idx = {}
                    for i, lab in enumerate(sorted(unique_labels)):
                        mask = labels == lab
                        if np.any(mask):
                            center = np.mean(X[mask], axis=0)
                            centers.append(center)
                            self.label_to_center_idx[lab] = i
                    if centers:
                        self.cluster_centers = np.vstack(centers)
                    else:
                        self.cluster_centers = None
                    logger.info(
                        f"HDBSCAN создал {n_clusters} кластеров (шум: {sum(labels == -1)})"
                    )
                else:
                    logger.warning("HDBSCAN не смог найти кластеры, используем KMeans")
            except Exception as e:
                logger.warning(
                    f"HDBSCAN не отработал: {e}. Fallback на KMeans."
                )
                use_hdbscan = False   # принудительно переключаем на KMeans

        if not use_hdbscan:
            if best_model is None:
                logger.warning("Не удалось подобрать KMeans, создаём один кластер")
                self.labels_ = np.zeros(n_samples, dtype=int)
                self.cluster_centers = np.mean(X, axis=0).reshape(1, -1)
                self.model = None
                self.label_to_center_idx = {0: 0}
            else:
                self.model = best_model
                self.clusterer_type = "kmeans"
                self.labels_ = best_labels
                self.cluster_centers = best_model.cluster_centers_
                self.label_to_center_idx = {i: i for i in range(best_k)}
            logger.info(
                f"Выбрано K={best_k} с silhouette_score={best_score:.3f}"
            )

        self.is_fitted = True
        self._save_model(level)
        logger.info(
            f"✅ Кластеризация завершена. Тип: {self.clusterer_type}, "
            f"кластеров: {self.n_clusters_}"
        )
        return self

    @property
    def n_clusters_(self) -> int:
        if self.labels_ is None:
            return 0
        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

    def _save_model(self, level: str):
        path = config.DATA_PROCESSED_DIR / f"vacancy_clusters_{level}.pkl"
        data = {
            'model': self.model,
            'clusterer_type': self.clusterer_type,
            'labels': self.labels_,
            'centers': self.cluster_centers,
            'vacancy_ids': self.vacancy_ids,
            'vacancy_skills': self.vacancy_skills,
            'n_clusters': self.n_clusters_,
            'min_cluster_size': self.min_cluster_size,
            'label_to_center_idx': self.label_to_center_idx,
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
        self.clusterer_type = data.get('clusterer_type', 'kmeans')
        self.labels_ = data['labels']
        self.cluster_centers = data.get('centers')
        self.vacancy_ids = data['vacancy_ids']
        self.vacancy_skills = data.get('vacancy_skills', [])
        self.n_clusters = data['n_clusters']
        self.min_cluster_size = data.get('min_cluster_size', 15)
        self.label_to_center_idx = data.get('label_to_center_idx', {})
        self.is_fitted = True
        logger.info(
            f"Модель кластеризации загружена: {path} "
            f"(тип: {self.clusterer_type})"
        )
        return True

    def find_closest_clusters(
        self,
        student_skills_or_embedding: Union[List[str], np.ndarray],
        top_k: int = 3
    ) -> List[Tuple[int, float]]:
        if not self.is_fitted or self.cluster_centers is None:
            return []

        if isinstance(student_skills_or_embedding, np.ndarray):
            student_emb = student_skills_or_embedding
        else:
            embedding_model = get_embedding_model()
            if not student_skills_or_embedding:
                student_emb = np.zeros(embedding_model.get_sentence_embedding_dimension())
            else:
                embs = embedding_model.encode(
                    student_skills_or_embedding, convert_to_numpy=True, show_progress_bar=False
                )
                student_emb = np.mean(embs, axis=0)

        # Нормализуем
        norm = np.linalg.norm(student_emb)
        if norm > 1e-8:
            student_emb = student_emb / norm

        similarities = cosine_similarity([student_emb], self.cluster_centers)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        idx_to_label = {v: k for k, v in self.label_to_center_idx.items()}
        result = []
        for idx in top_indices:
            label = idx_to_label.get(idx, idx)
            result.append((int(label), float(similarities[idx])))
        return result

    def get_cluster_skills(self, cluster_id: int, vacancies: Optional[List[Dict]] = None) -> List[str]:
        if not self.is_fitted or self.labels_ is None:
            return []
        skills_set = set()
        for i, label in enumerate(self.labels_):
            if label == cluster_id and i < len(self.vacancy_skills):
                skills_set.update(self.vacancy_skills[i])
        return list(skills_set)

    def get_cluster_context(
        self,
        profile_embedding: np.ndarray,
        level: str = "middle",
        top_k_clusters: int = 3,
        top_k_skills_per_cluster: int = 20
    ) -> Dict[str, Any]:
        """
        Возвращает контекст: ближайшие кластеры, их навыки и веса.
        """
        if profile_embedding is None:
            return {"clusters": [], "skills": {}, "total_skills_in_context": 0}

        # Нормализуем эмбеддинг
        embedding = profile_embedding / np.linalg.norm(profile_embedding)

        # Находим ближайшие кластеры
        closest = self.find_closest_clusters(embedding, top_k=top_k_clusters)
        if not closest:
            return {"clusters": [], "skills": {}, "total_skills_in_context": 0}

        context_skills = {}
        for cluster_id, sim in closest:
            top_skills = self.get_top_skills_in_cluster(cluster_id, top_n=top_k_skills_per_cluster)
            for skill in top_skills:
                # Вес = сходство кластера (можно также учесть частоту)
                context_skills[skill] = max(context_skills.get(skill, 0.0), sim)

        return {
            "closest_clusters": [{"id": cid, "similarity": sim} for cid, sim in closest],
            "skills": context_skills,
            "total_skills_in_context": len(context_skills)
        }

    def get_top_skills_in_cluster(self, cluster_id: int, top_n: int = 30) -> List[str]:
        """Возвращает топ-N навыков кластера по частоте."""
        if not self.is_fitted or self.labels_ is None:
            return []
        from collections import Counter
        counter = Counter()
        for i, label in enumerate(self.labels_):
            if label == cluster_id and i < len(self.vacancy_skills):
                counter.update(self.vacancy_skills[i])
        return [skill for skill, _ in counter.most_common(top_n)]