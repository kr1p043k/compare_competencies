"""Гибридные веса BM25 + Embeddings (с PCA, graceful degradation, merge BM25-only)."""

import math
import numpy as np
import structlog
import torch

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.parsing.skills.skill_embedding_cache import SkillEmbeddingCache

logger = structlog.get_logger(__name__)


class HybridWeightCalculator:
    def __init__(self, bm25_ranker, embedding_cache=None):
        self.bm25 = bm25_ranker
        self.cache = embedding_cache or SkillEmbeddingCache()

    def calculate(self, vacancies: list) -> Result[dict[str, float], DomainError]:
        bm25_weights: dict[str, float] = {}
        match self.bm25.calculate_weights(vacancies):
            case Ok(w):
                bm25_weights = w
            case Err(err):
                return Err(DomainError(message="BM25 weight calculation failed", detail=str(err)))

        if not bm25_weights:
            return Ok({})

        if self.cache.model is None:
            logger.warning("Эмбеддинги недоступны — только BM25")
            return Ok(self._norm(bm25_weights))

        try:
            skills = list(bm25_weights.keys())
            emb_dict = self.cache.get_embeddings(skills)
        except Exception as e:
            logger.warning("embedding_failed_fallback_to_bm25", error=str(e))
            return Ok(self._norm(bm25_weights))

        if len(emb_dict) < 10:
            return Ok(self._norm(bm25_weights))

        skill_list = list(emb_dict.keys())
        embs = np.array([emb_dict[s] for s in skill_list], dtype=np.float32)

        if config.PCA_ENABLED and len(embs) > config.PCA_MIN_SAMPLES and embs.shape[1] > config.PCA_MIN_FEATURES:
            from sklearn.decomposition import PCA

            n_comp = min(config.PCA_TARGET_DIM, len(embs) - 1, embs.shape[1])
            if n_comp < embs.shape[1]:
                embs = PCA(n_components=n_comp, random_state=config.GLOBAL_RANDOM_SEED).fit_transform(embs)
                logger.info(f"PCA: {emb_dict[skill_list[0]].shape[0]} → {n_comp}")

        embs_t = torch.tensor(embs)
        embs_t = torch.nn.functional.normalize(embs_t, p=2, dim=1)

        # Centroid of all skills = "average skill"
        centroid = embs_t.mean(dim=0, keepdim=True)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)

        # Cosine similarity of each skill to centroid
        # High = general/common skill, Low = specific/unique skill
        sim_to_center = (embs_t @ centroid.T).squeeze().cpu().numpy()

        # Inverse: specific skills get higher semantic weight
        sem_raw = 1.0 - sim_to_center  # [0, 2], higher = more specific
        sem_raw = np.clip(sem_raw, 0.0, 2.0)

        # IDF bonus: rare skills get higher weight
        try:
            import json
            from src import config as cfg
            freq_path = cfg.DATA_PROCESSED_DIR / "competency_frequency.json"
            if freq_path.exists():
                freq_data = json.loads(freq_path.read_text(encoding="utf-8"))
                total_skills = len(skill_list)
                idf_bonus = np.array([
                    math.log(max(total_skills / (freq_data.get(s, 0) + 1), 1.0))
                    for s in skill_list
                ])
                idf_bonus = idf_bonus / max(idf_bonus.max(), 1.0)
                sem_raw = sem_raw * (0.5 + 0.5 * idf_bonus)
        except Exception:
            pass

        sem_norm = (sem_raw - sem_raw.min()) / (sem_raw.max() - sem_raw.min() + 1e-8)

        alpha = getattr(config, "HYBRID_BM25_WEIGHT", 0.65)
        beta = getattr(config, "HYBRID_SEM_WEIGHT", 0.35)
        total = alpha + beta
        alpha, beta = alpha / total, beta / total

        hybrid = {}
        for i, skill in enumerate(skill_list):
            hybrid[skill] = round(float(alpha * bm25_weights[skill] + beta * sem_norm[i]), 4)

        bm25_only = 0
        for skill, w in bm25_weights.items():
            if skill not in hybrid:
                hybrid[skill] = w
                bm25_only += 1

        logger.info(f"Гибридные веса готовы: {len(hybrid)} навыков (bm25_only={bm25_only})")
        return Ok(hybrid)

    @staticmethod
    def _norm(w: dict[str, float]) -> dict[str, float]:
        if not w:
            return w
        vals = np.array(list(w.values()))
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            return {s: round((w[s] - vmin) / (vmax - vmin), 4) for s in w}
        return w
