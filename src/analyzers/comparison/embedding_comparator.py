"""
Embedding Comparator — семантическое сравнение навыков через эмбеддинги.
Атомарная запись кэша эмбеддингов + восстановление при порче.
"""

import os
import tempfile
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
from src.parsing.api.embedding_loader import get_embedding_model

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
        self.model = get_embedding_model(model_name)
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

    def _get_cache_path(self, name: str, level: str = "middle") -> Path:
        return self.cache_dir / f"{name}_{level}.joblib"

    def embed_skills(self, skills: list[str]) -> np.ndarray:
        if not skills:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim))
        return self.model.encode(skills, convert_to_numpy=True, show_progress_bar=False)

    def build_market_index(self, all_market_skills: list[str], level: str = "middle"):
        cache_path = self._get_cache_path("market_embeddings", level)

        if cache_path.exists():
            manifest_path = cache_path.with_suffix(".manifest.json")
            manifest_ok = True
            if manifest_path.exists():
                match ArtifactManifest.load(cache_path):
                    case Ok(manifest) if not manifest.is_compatible():
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

        manifest = ArtifactManifest(
            artifact_path=cache_path,
            metrics={"num_skills": len(self.market_skills)},
        )
        if manifest.save().is_err():
            logger.warning("market_cache_manifest_save_failed")

    def compare_student_to_market(self, student_skills: list[str]) -> Result[dict, DomainError]:
        if self.market_embeddings is None:
            return Err(DomainError(message="Сначала вызови build_market_index()"))

        student_embs = self.embed_skills(student_skills)

        if len(student_embs) == 0:
            return Ok({"score": 0.0, "weighted_coverage": 0.0, "matches": [], "missing": [], "avg_similarity": 0.0})

        best_sims_per_market = {}
        similarities = cosine_similarity(student_embs, self.market_embeddings)
        for i in range(len(student_skills)):
            for j, sim in enumerate(similarities[i]):
                mskill = self.market_skills[j]
                best_sims_per_market[mskill] = max(best_sims_per_market.get(mskill, 0.0), float(sim))

        return Ok(self._result_from_sims(best_sims_per_market))

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
                    return {"score": 0.0, "matches": []}
                student_embs = self._outer.embed_skills(ss)
                if len(student_embs) == 0:
                    return {"score": 0.0, "matches": []}
                sims = cosine_similarity(student_embs, self._outer.market_embeddings)
                best = {}
                for i in range(len(ss)):
                    for j, s in enumerate(sims[i]):
                        best[ms[j]] = max(best.get(ms[j], 0.0), float(s))
                avg = float(np.mean(list(best.values()))) if best else 0.0
                matches = sorted(
                    [{"skill": k, "similarity": v} for k, v in best.items()],
                    key=lambda x: x["similarity"], reverse=True,
                )[:15]
                return dict(score=round(avg, 4), weighted_coverage=round(avg, 4),
                            avg_similarity=round(avg, 4), matches=matches, missing=[])

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

    def _result_from_sims(self, best_sims_per_market: dict[str, float]) -> ComparisonResult:
        total_weighted = 0.0
        total_weight = 0.0

        if not self.skill_weights:
            logger.warning("skill_weights_empty")
        else:
            logger.debug("skill_weights_count", count=len(self.skill_weights))

        if self.skill_weights:
            for mskill, weight in self.skill_weights.items():
                raw_sim = best_sims_per_market.get(mskill, 0.0)
                effective_sim = raw_sim**2
                total_weighted += effective_sim * weight
                total_weight += weight
        else:
            for _mskill, raw_sim in best_sims_per_market.items():
                effective_sim = raw_sim**2
                total_weighted += effective_sim
                total_weight += 1.0

        weighted_coverage = total_weighted / total_weight if total_weight > 0 else 0.0
        avg_similarity = float(np.mean(list(best_sims_per_market.values()))) if best_sims_per_market else 0.0

        sorted_matches = sorted(
            [{"skill": k, "similarity": v} for k, v in best_sims_per_market.items()],
            key=lambda x: x["similarity"],
            reverse=True,
        )[:15]

        return dict(
            score=round(weighted_coverage, 4),
            weighted_coverage=round(weighted_coverage, 4),
            avg_similarity=round(avg_similarity, 4),
            matches=sorted_matches,
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
