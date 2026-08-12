"""
Embedding Comparator — семантическое сравнение навыков через эмбеддинги.
Атомарная запись кэша эмбеддингов + восстановление при порче.
"""

import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import structlog
from sklearn.metrics.pairwise import cosine_similarity

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.analyzers.comparison.engines import (
    ComparisonResult,
    EnsembleEngine,
    JaccardEngine,
    SimilarityEngine,
)
from src.artifacts import ArtifactManifest
from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory

if TYPE_CHECKING:
    from src.analyzers.clustering.vacancy_clustering import VacancyClusterer

logger = structlog.get_logger(__name__)


#: Вес cosine similarity при бленде с extra_engines. Остаток делится между extra.
COSINE_WEIGHT = 0.7


class EmbeddingComparator:
    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
        similarity_threshold: float = 0.75,
    ):
        self.provider = EmbeddingProviderFactory.get(model_name)
        self.model = self.provider
        if cache_dir is None:
            self.cache_dir = config.EMBEDDINGS_CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.market_embeddings = None
        self.market_skills = None
        self.skill_weights: dict[str, float] = {}
        self.clusterer: VacancyClusterer | None = None
        self.vacancies_data: list[dict] = []
        self._skill_cache_loaded = False
        self._skill_cache_lock = threading.Lock()
        self._skill_cache: dict[str, np.ndarray] = {}

    def _get_cache_path(self, name: str, level: str = "middle") -> Path:
        return self.cache_dir / f"{name}_{level}.joblib"

    def _load_skill_cache(self) -> None:
        """Load persisted per-skill embeddings (shared JSON cache)."""
        if self._skill_cache_loaded:
            return
        self._skill_cache_loaded = True
        try:
            cache_file = self.cache_dir / "skill_embeddings.json"
            if not cache_file.exists():
                return
            import json

            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            with self._skill_cache_lock:
                self._skill_cache = {
                    s: np.asarray(e, dtype=np.float32)
                    for s, e in data.get("embeddings", {}).items()
                }
        except Exception as exc:
            logger.warning("skill_cache_load_failed", error=str(exc))

    def _save_skill_cache(self) -> None:
        try:
            import json
            import os
            import tempfile

            cache_file = self.cache_dir / "skill_embeddings.json"
            with self._skill_cache_lock:
                snapshot = dict(self._skill_cache)
            data = {
                "model_version": self.model.model_version(),
                "embeddings": {
                    s: [float(x) for x in e] for s, e in snapshot.items()
                },
            }
            fd, tmp_path = tempfile.mkstemp(dir=cache_file.parent, suffix=".json.tmp")
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, cache_file)
        except Exception as exc:
            logger.warning("skill_cache_save_failed", error=str(exc))

    def embed_skills(self, skills: list[str]) -> np.ndarray:
        if not skills:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim))
        self._load_skill_cache()
        todo = [s for s in skills if s not in self._skill_cache]
        if todo:
            embs = self.model.encode(todo, convert_to_numpy=True, show_progress_bar=False)
            with self._skill_cache_lock:
                for s, e in zip(todo, embs, strict=False):
                    self._skill_cache[s] = e
            self._save_skill_cache()
        return np.stack([self._skill_cache[s] for s in skills])

    def build_market_index(self, all_market_skills: list[str], level: str = "middle"):
        cache_path = self._get_cache_path("market_embeddings", level)

        if cache_path.exists():
            manifest_path = cache_path.with_suffix(".manifest.json")
            manifest_ok = True
            if manifest_path.exists():
                match ArtifactManifest.load(cache_path):
                    case Ok(manifest):
                        match manifest.is_compatible():
                            case Ok(True):
                                pass
                            case _:
                                logger.info("market_cache_invalidated_by_model",
                                    level=level,
                                    manifest_version=manifest.model_version,
                                    current_version=ArtifactManifest._get_embedding_model_version())
                                manifest_ok = False
                    case Err(err):
                        logger.warning("market_cache_manifest_load_failed", error=str(err))
                        manifest_ok = False
            if manifest_ok:
                try:
                    loaded = joblib.load(cache_path)
                    if isinstance(loaded, dict):
                        self.market_embeddings = loaded["embeddings"]
                        self.market_skills = loaded["skills"]
                    else:
                        self.market_embeddings, self.market_skills = loaded
                    logger.info("embeddings_cache_loaded", level=level)
                    return
                except Exception as e:
                    logger.warning("market_cache_load_failed", level=level, error=str(e))
                    with suppress(Exception):
                        cache_path.unlink()
                    with suppress(Exception):
                        manifest_path.unlink()

        self.market_skills = all_market_skills
        self.market_embeddings = self.embed_skills(self.market_skills)

        try:
            fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".joblib.tmp")
            os.close(fd)
            joblib.dump(
                {"embeddings": self.market_embeddings, "skills": self.market_skills},
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
            logger.info("market_embeddings_saved_atomically", level=level, path=str(cache_path))
        except Exception as e:
            logger.error("failed_to_save_market_embeddings", error=str(e))
            with suppress(Exception):
                os.unlink(tmp_path)
            raise

        try:
            manifest = ArtifactManifest(
                artifact_path=cache_path,
                metrics={"num_skills": len(self.market_skills)},
            )
            if manifest.save().is_err():
                logger.warning("market_cache_manifest_save_failed")
        except Exception as e:
            logger.warning("market_cache_manifest_save_failed", error=str(e))

    def compare_student_to_market(self, student_skills: list[str]) -> Result[dict, DomainError]:
        if self.market_embeddings is None:
            return Err(DomainError(message="Сначала вызови build_market_index()"))

        student_embs = self.embed_skills(student_skills)

        if len(student_embs) == 0:
            return Ok({"score": 0.0, "weighted_coverage": 0.0, "matches": [], "missing": [], "avg_similarity": 0.0})

        similarities = cosine_similarity(student_embs, self.market_embeddings)
        best_sims = similarities.max(axis=0)

        return Ok(self._result_from_sims(best_sims))

    def compare_student_to_market_ensemble(
        self,
        student_skills: list[str],
        extra_engines: dict[str, tuple[SimilarityEngine, float]] | None = None,
    ) -> ComparisonResult:
        """Blends cosine similarity with extra_engines via EnsembleEngine.

        Usage:
            comp.compare_student_to_market_ensemble(
                student_skills,
                extra_engines={"jaccard": (JaccardEngine(), 0.3)},
            )
        """
        base = self.compare_student_to_market(student_skills).unwrap_or(
            {"score": 0.0, "weighted_coverage": 0.0, "avg_similarity": 0.0, "matches": [], "missing": []}
        )
        if not extra_engines or not self.market_skills:
            return base

        class _CosineProxy:
            def __init__(self, outer):
                self._outer = outer
            def compare(self, ss, ms):
                if self._outer.market_embeddings is None:
                    return Ok({"score": 0.0, "matches": []})
                student_embs = self._outer.embed_skills(ss)
                if len(student_embs) == 0:
                    return Ok({"score": 0.0, "matches": []})
                sims = cosine_similarity(student_embs, self._outer.market_embeddings)
                best = sims.max(axis=0)
                avg = float(best.mean()) if best.size else 0.0
                top_idx = np.argsort(best)[-15:][::-1]
                matches = [
                    {"skill": self._outer.market_skills[i], "similarity": float(best[i])}
                    for i in top_idx
                ]
                return Ok(dict(score=round(avg, 4), weighted_coverage=round(avg, 4),
                            avg_similarity=round(avg, 4), matches=matches, missing=[]))

        total_extra = sum(w for _, w in extra_engines.values()) or 1.0
        engines: dict[str, tuple[SimilarityEngine, float]] = {
            "cosine": (_CosineProxy(self), COSINE_WEIGHT),
        }
        for name, (engine, weight) in extra_engines.items():
            engines[name] = (engine, (1 - COSINE_WEIGHT) * weight / total_extra)

        ensemble = EnsembleEngine(engines)
        return ensemble.compare(student_skills, self.market_skills).unwrap_or(
            {"score": 0.0, "weighted_coverage": 0.0, "avg_similarity": 0.0, "matches": [], "missing": []}
        )

    def _result_from_sims(self, best_sims: np.ndarray) -> ComparisonResult:
        """Build result from a (n_market,) array of best cosine similarities.

        Accepts either a numpy array (vectorized path) or a dict {skill: sim}
        (legacy path). Kept numpy-vectorized so the hot loop over 35k market
        skills is a single array operation instead of a Python loop.
        """
        if isinstance(best_sims, dict):
            values = np.fromiter(best_sims.values(), dtype=float)
            skills_list = list(best_sims.keys())
        else:
            values = np.asarray(best_sims, dtype=float)
            skills_list = None

        if values.size == 0:
            return dict(
                score=0.0, weighted_coverage=0.0, avg_similarity=0.0,
                matches=[], missing=[],
            )

        effective = values ** 2
        if self.skill_weights and skills_list is not None:
            weights = np.array([self.skill_weights.get(s, 0.0) for s in skills_list], dtype=float)
            total_w = float(weights.sum())
            weighted_coverage = float(np.dot(effective, weights) / total_w) if total_w > 0 else float(effective.mean())
        else:
            weighted_coverage = float(effective.mean())

        avg_similarity = float(values.mean())

        if skills_list is None:
            skills_list = [self.market_skills[i] for i in range(len(values))]

        top_idx = np.argsort(values)[-15:][::-1]
        matches = [
            {"skill": skills_list[i], "similarity": float(values[i])}
            for i in top_idx
        ]

        return dict(
            score=round(weighted_coverage, 4),
            weighted_coverage=round(weighted_coverage, 4),
            avg_similarity=round(avg_similarity, 4),
            matches=matches,
            missing=[],
        )

    def get_vacancy_embedding(self, skills: list[str]) -> np.ndarray:
        if not skills:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        embs = self.embed_skills(skills)
        return np.mean(embs, axis=0)

    def find_closest_vacancies(
        self, student_skills: list[str], vacancies: list[dict], level: str = "middle", top_k: int = 50
    ) -> list[dict]:
        student_emb = self.embed_skills(student_skills)
        if len(student_emb) == 0:
            student_emb = np.zeros((1, self.model.get_sentence_embedding_dimension()))
        else:
            student_emb = np.mean(student_emb, axis=0).reshape(1, -1)

        level_vacancies = [v for v in vacancies if v.get("experience") == level]
        if not level_vacancies:
            level_vacancies = vacancies
        if not level_vacancies:
            logger.warning("find_closest_vacancies_no_vacancies", level=level)
            return []

        # Batched embedding: embed all unique skills once, then mean-pool per vacancy
        vac_skill_lists = [(i, v.get("skills", [])) for i, v in enumerate(level_vacancies)]
        all_skills = list({s for _, sk in vac_skill_lists for s in sk})
        if not all_skills:
            logger.warning("find_closest_vacancies_no_skills", level=level)
            return []

        skill_to_emb = dict(zip(all_skills, self.embed_skills(all_skills), strict=False))

        vac_embs = np.zeros((len(level_vacancies), self.model.get_sentence_embedding_dimension()))
        for i, skills in vac_skill_lists:
            embs = [skill_to_emb[s] for s in skills if s in skill_to_emb]
            vac_embs[i] = np.mean(embs, axis=0) if embs else np.zeros(self.model.get_sentence_embedding_dimension())

        similarities = cosine_similarity(student_emb, vac_embs)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [level_vacancies[i] for i in top_indices]

    def set_clusterer(self, clusterer: "VacancyClusterer", vacancies_data: list[dict]):
        self.clusterer = clusterer
        self.vacancies_data = vacancies_data

    def compare_to_clusters(self, student_skills: list[str], top_k: int = 3) -> dict[str, Any]:
        if self.clusterer is None or not self.clusterer.is_fitted:
            return {"clusters": [], "error": "Clusterer not available"}

        closest = self.clusterer.find_closest_clusters(student_skills, top_k)
        result = []
        for cluster_id, sim in closest:
            cluster_skills = self.clusterer.get_cluster_skills(cluster_id, self.vacancies_data)
            covered = len(set(student_skills) & set(cluster_skills))
            coverage = covered / len(cluster_skills) if cluster_skills else 0.0
            result.append(
                {
                    "cluster_id": cluster_id,
                    "similarity": round(sim, 4),
                    "coverage": round(coverage, 4),
                    "top_skills": cluster_skills[:10],
                }
            )
        return {"clusters": result}

    def hybrid_compare(
        self, student_skills: list[str], global_weights: dict[str, float], cluster_weight: float = 0.6
    ) -> dict[str, Any]:
        global_result = self.compare_student_to_market(student_skills).unwrap_or(
            {"avg_similarity": 0.0, "weighted_coverage": 0.0}
        )
        global_score = global_result["avg_similarity"]

        cluster_result = self.compare_to_clusters(student_skills, top_k=3)
        clusters = cluster_result.get("clusters", [])
        if clusters:
            best_cluster = clusters[0]
            cluster_score = best_cluster["coverage"]
        else:
            best_cluster = None
            cluster_score = global_score

        if best_cluster is not None:
            hybrid_score = (1 - cluster_weight) * global_score + cluster_weight * cluster_score
        else:
            hybrid_score = global_score

        return {
            "global_score": round(global_score, 4),
            "cluster_score": round(cluster_score, 4) if best_cluster else None,
            "hybrid_score": round(hybrid_score, 4),
            "best_cluster": best_cluster,
            "all_clusters": clusters,
        }
