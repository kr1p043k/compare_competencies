"""
Кластеризация вакансий на основе эмбеддингов навыков.
Используется для определения профессиональных ролей.
Поддерживает KMeans с подбором числа кластеров по silhouette_score,
а также HDBSCAN как fallback (с обработкой ошибок метрики).
"""

from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from src import Result, Ok, Err, config
from src.errors import DomainError
from src.artifacts import ArtifactManifest
from src.models.enums import ExperienceLevel
from src.parsing.api.embedding_loader import get_embedding_model
from src.parsing.skills.skill_normalizer import SkillNormalizer

logger = structlog.get_logger(__name__)

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.info("hdbscan_unavailable_fallback_to_kmeans")


class VacancyClusterer:
    """Кластеризует вакансии по эмбеддингам навыков с автоопределением числа кластеров."""

    def __init__(
        self,
        n_clusters: int = 10,
        min_clusters: int = 2,
        max_clusters: int = 40,
        random_state: int = 42,
        use_hdbscan_fallback: bool = True,
        min_cluster_size: int = 15,
        skill_weights: dict[str, float] | None = None,
    ):
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.use_hdbscan_fallback = use_hdbscan_fallback and HDBSCAN_AVAILABLE
        self.min_cluster_size = min_cluster_size
        self.skill_weights = skill_weights or {}

        self.model = None
        self.clusterer_type = "kmeans"
        self.labels_: np.ndarray | None = None
        self.cluster_centers: np.ndarray | None = None
        self.vacancy_ids: list[str] = []
        self.vacancy_skills: list[list[str]] = []
        self.is_fitted = False
        self.label_to_center_idx: dict[int, int] = {}

    def _compute_embeddings(self, vacancies: list[dict]) -> np.ndarray:
        embedding_model = get_embedding_model()
        dim = embedding_model.get_sentence_embedding_dimension()

        has_weights = bool(self.skill_weights)

        # Step 1: normalise all skills
        cleaned_per_vacancy = []
        all_skill_set: set[str] = set()
        for skills in self.vacancy_skills:
            def _norm(s):
                match SkillNormalizer.normalize(s):
                    case Ok(n):
                        return n
                    case _:
                        logger.warning("skill_normalization_failed_in_clustering", skill=s)
                        return None
            clean = [n for s in skills if (n := _norm(s))]
            cleaned_per_vacancy.append(clean)
            all_skill_set.update(clean)

        # Step 2: batch-embed all unique skills at once
        unique_skills = sorted(all_skill_set)
        if unique_skills:
            skill_embs = embedding_model.encode(unique_skills, convert_to_numpy=True, show_progress_bar=False)
            skill_to_emb = dict(zip(unique_skills, skill_embs, strict=False))
        else:
            skill_to_emb = {}

        # Step 3: weighted mean-pool per vacancy (BM25/hybrid weights or uniform)
        vacancy_embs = []
        empty_skills_count = 0
        for clean_skills in cleaned_per_vacancy:
            if not clean_skills:
                vacancy_embs.append(np.zeros(dim))
                empty_skills_count += 1
                continue

            embs_list = [skill_to_emb[s] for s in clean_skills if s in skill_to_emb]
            if not embs_list:
                vacancy_embs.append(np.zeros(dim))
                empty_skills_count += 1
                continue

            if has_weights:
                weights = np.array([self.skill_weights.get(s, 0.01) for s in clean_skills if s in skill_to_emb])
            else:
                weights = np.ones(len(embs_list))
            weighted = np.average(embs_list, axis=0, weights=weights)
            vacancy_embs.append(weighted)

        embeddings = np.vstack(vacancy_embs)

        logger.info(
            "embeddings_computed",
            shape=embeddings.shape,
            unique_skills=len(unique_skills),
            empty_skills=empty_skills_count,
            total=len(vacancies),
            weighted=has_weights,
        )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def fit(self, vacancies: list[dict], level: str = "all") -> "VacancyClusterer":
        if not vacancies:
            logger.warning("no_vacancies_for_clustering")
            return self

        n_samples = len(vacancies)
        if n_samples < 10:
            logger.warning("too_few_vacancies_for_clustering", samples=n_samples, level=level)
            self.is_fitted = False
            return self

        self.vacancy_ids = [v.get("id", str(i)) for i, v in enumerate(vacancies)]
        self.vacancy_skills = [v.get("skills", []) for v in vacancies]

        x = self._compute_embeddings(vacancies)
        logger.info("clustering_started", samples=len(x), level=level)

        best_k = self.n_clusters
        best_score = -1.0
        best_labels = None
        best_model = None

        # Для малого числа вакансий уменьшаем max_k
        max_k = min(self.max_clusters, max(3, int(np.sqrt(n_samples))))

        logger.info(
            "kmeans_params",
            n_clusters=self.n_clusters,
            min_clusters=self.min_clusters,
            max_clusters=max_k,
            random_state=self.random_state,
        )

        no_improve = 0
        for k in range(max(2, self.min_clusters), max_k + 1):
            if k >= n_samples:
                continue
            kmeans = KMeans(
                n_clusters=k,
                random_state=config.GLOBAL_RANDOM_SEED if hasattr(config, "GLOBAL_RANDOM_SEED") else 42,
                n_init="auto",
                max_iter=300,
            )
            labels = kmeans.fit_predict(x)
            if len(set(labels)) < 2:
                continue
            try:
                if n_samples > 500:
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n_samples, 500, replace=False)
                    score = silhouette_score(x[idx], labels[idx], metric="cosine")
                else:
                    score = silhouette_score(x, labels, metric="cosine")
            except Exception:
                continue
            logger.debug("kmeans_silhouette", k=k, silhouette=round(score, 4))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_model = kmeans
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 5:
                logger.debug("early_stop_silhouette", k=k, no_improve=no_improve)
                break

        if best_score < 0.1:
            logger.warning("low_silhouette_score", score=round(best_score, 3))

        # Если silhouette_score плохой, пробуем HDBSCAN (cosine)
        use_hdbscan = False
        if best_score < 0.2 and self.use_hdbscan_fallback and n_samples >= self.min_cluster_size * 2:
            logger.info("trying_hdbscan_fallback", silhouette=round(best_score, 3))
            try:
                # euclidean на L2-нормализованных векторах эквивалентен cosine
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size, metric="euclidean", core_dist_n_jobs=-1
                )
                labels = clusterer.fit_predict(x)
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
                            center = np.mean(x[mask], axis=0)
                            centers.append(center)
                            self.label_to_center_idx[lab] = i
                    if centers:
                        self.cluster_centers = np.vstack(centers)
                    else:
                        self.cluster_centers = None
                    logger.info(
                        "hdbscan_clusters_created",
                        clusters=n_clusters,
                        noise=int(sum(labels == -1)),
                    )
                else:
                    logger.warning("hdbscan_no_clusters_found")
            except Exception as e:
                logger.warning("hdbscan_failed", error=str(e))
                use_hdbscan = False  # принудительно переключаем на KMeans

        if not use_hdbscan:
            if best_model is None:
                logger.warning("no_kmeans_model_creating_single_cluster")
                self.labels_ = np.zeros(n_samples, dtype=int)
                self.cluster_centers = np.mean(x, axis=0).reshape(1, -1)
                self.model = None
                self.label_to_center_idx = {0: 0}
            else:
                self.model = best_model
                self.clusterer_type = "kmeans"
                self.labels_ = best_labels
                self.cluster_centers = best_model.cluster_centers_
                self.label_to_center_idx = {i: i for i in range(best_k)}
            logger.info(
                "kmeans_selected",
                k=best_k,
                silhouette=round(best_score, 3),
            )

        self.is_fitted = True
        self._save_model(level)
        logger.info(
            "clustering_completed",
            type=self.clusterer_type,
            clusters=self.n_clusters_,
        )
        return self

    @property
    def n_clusters_(self) -> int:
        if self.labels_ is None:
            return 0
        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

    def _save_model(self, level: ExperienceLevel):
        path = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{level}.joblib"
        data = {
            "model": self.model,
            "clusterer_type": self.clusterer_type,
            "labels": self.labels_,
            "centers": self.cluster_centers,
            "vacancy_ids": self.vacancy_ids,
            "vacancy_skills": self.vacancy_skills,
            "n_clusters": self.n_clusters_,
            "min_cluster_size": self.min_cluster_size,
            "label_to_center_idx": self.label_to_center_idx,
            "skill_weights": self.skill_weights,
        }
        joblib.dump(data, path)
        logger.info(f"Модель кластеризации сохранена: {path}")

        manifest = ArtifactManifest(
            artifact_path=path,
            metrics={"clusters": self.n_clusters_, "samples": len(self.vacancy_ids)},
        )
        if manifest.save().is_err():
            logger.warning("cluster_manifest_save_failed")

    def _migrate_pkl_to_joblib(self, level: str) -> None:
        path_pkl = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{level}.pkl"
        path_joblib = path_pkl.with_suffix(".joblib")
        try:
            data = joblib.load(path_pkl)
            joblib.dump(data, path_joblib)
            path_pkl.unlink()
            logger.info("migrated_pkl_to_joblib", level=level, path=str(path_joblib))
        except Exception as e:
            logger.error("pkl_migration_failed", level=level, error=str(e))

    def load_model(self, level: str = "all") -> bool:
        path = config.VACANCY_CLUSTERS_CACHE_DIR / f"vacancy_clusters_{level}.joblib"
        if not path.exists():
            path_old = path.with_suffix(".pkl")
            if path_old.exists():
                logger.warning("detected_old_pkl_format_migrating", path=str(path_old))
                self._migrate_pkl_to_joblib(level)
                if not path.exists():
                    return False
            else:
                logger.warning("cluster_model_file_not_found", path=str(path))
                return False
        data = joblib.load(path)
        self.model = data["model"]
        self.clusterer_type = data.get("clusterer_type", "kmeans")
        self.labels_ = data["labels"]
        self.cluster_centers = data.get("centers")
        self.vacancy_ids = data["vacancy_ids"]
        self.vacancy_skills = data.get("vacancy_skills", [])
        self.n_clusters = data["n_clusters"]
        self.min_cluster_size = data.get("min_cluster_size", 15)
        self.label_to_center_idx = data.get("label_to_center_idx", {})
        self.skill_weights = data.get("skill_weights", {})
        self.is_fitted = True
        logger.info(
            "cluster_model_loaded",
            path=str(path),
            type=self.clusterer_type,
        )
        return True

    def find_closest_clusters(
        self, student_skills_or_embedding: list[str] | np.ndarray, top_k: int = 3
    ) -> list[tuple[int, float]]:
        if not self.is_fitted or self.cluster_centers is None:
            logger.warning("clusterer_not_fitted_for_find_closest")
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

    def get_cluster_skills(self, cluster_id: int, vacancies: list[dict] | None = None) -> list[str]:
        if not self.is_fitted or self.labels_ is None:
            logger.warning("clusterer_not_fitted_for_get_skills")
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
        top_k_skills_per_cluster: int = 20,
    ) -> Result[dict[str, Any], DomainError]:
        if profile_embedding is None:
            return Ok({"clusters": [], "skills": {}, "total_skills_in_context": 0})

        embedding = profile_embedding / np.linalg.norm(profile_embedding)

        closest = self.find_closest_clusters(embedding, top_k=top_k_clusters)
        if not closest:
            return Ok({"clusters": [], "skills": {}, "total_skills_in_context": 0})

        context_skills = {}
        result_clusters = []

        for cluster_id, sim in closest:
            top_skills = self.get_top_skills_in_cluster(cluster_id, top_n=top_k_skills_per_cluster)

            cluster_name = self._generate_cluster_name(cluster_id)

            result_clusters.append({"id": int(cluster_id), "similarity": float(sim), "name": cluster_name})

            for skill in top_skills:
                context_skills[skill] = max(context_skills.get(skill, 0.0), sim)

        return Ok({
            "closest_clusters": result_clusters,
            "skills": context_skills,
            "total_skills_in_context": len(context_skills),
        })

    def _generate_cluster_name(self, cluster_id: int) -> str:
        """
        Генерирует человекочитаемое имя кластера.
        Только доминирующая категория + вторичная (без перечисления навыков).
        """
        from src.analyzers.skills.skill_taxonomy import SkillTaxonomy

        taxonomy = SkillTaxonomy()
        top_skills = self.get_top_skills_in_cluster(cluster_id, top_n=15)

        if not top_skills:
            return f"Empty Cluster {cluster_id}"

        # Определяем доминирующую категорию (по топ-15 навыкам)
        dominant = taxonomy.get_dominant_category(top_skills)

        # Проверяем, что доминирующая категория осмысленна
        # (если other — ищем следующую по величине)
        if dominant == "other":
            stats = taxonomy.get_category_stats(top_skills)
            match stats:
                case Ok(s):
                    for cat, count in s.items():
                        if cat != "other" and count >= 2:
                            dominant = cat
                            break

        dominant_label = taxonomy.get_category_label_by_id(dominant)
        dominant_icon = taxonomy.get_category_icon_by_id(dominant)

        # Определяем вторичную категорию (если есть)
        stats = taxonomy.get_category_stats(top_skills)
        secondary = None
        match stats:
            case Ok(s):
                for cat, count in s.items():
                    if cat != dominant and cat != "other" and count >= 3:
                        secondary = taxonomy.get_category_label_by_id(cat)
                        break

        # Формируем имя: иконка + категория
        from src.visualization._config import EMOJI_TO_TEXT

        text_icon = EMOJI_TO_TEXT.get(dominant_icon, "")
        name = f"{text_icon} {dominant_label}" if text_icon else dominant_label

        if secondary:
            name += f" + {secondary}"

        return name

    def get_top_skills_in_cluster(self, cluster_id: int, top_n: int = 30) -> list[str]:
        """Возвращает топ-N навыков кластера по частоте."""
        if not self.is_fitted or self.labels_ is None:
            logger.warning("clusterer_not_fitted_for_top_skills")
            return []
        from collections import Counter

        counter = Counter()
        for i, label in enumerate(self.labels_):
            if label == cluster_id and i < len(self.vacancy_skills):
                counter.update(self.vacancy_skills[i])
        return [skill for skill, _ in counter.most_common(top_n)]
