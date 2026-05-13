"""Гибридные веса BM25 + Embeddings (с PCA, graceful degradation)."""

import numpy as np
import structlog
import torch

from src import config
from src.parsing.skills.skill_embedding_cache import SkillEmbeddingCache

logger = structlog.get_logger(__name__)


class HybridWeightCalculator:
    def __init__(self, bm25_ranker, embedding_cache=None):
        self.bm25 = bm25_ranker
        self.cache = embedding_cache or SkillEmbeddingCache()

    def calculate(self, vacancies: list) -> dict[str, float]:
        bm25_weights = self.bm25.calculate_weights(vacancies)
        if not bm25_weights:
            return {}

        if self.cache.model is None:
            logger.warning("Эмбеддинги недоступны — только BM25")
            return self._minmax(bm25_weights)

        try:
            skills = list(bm25_weights.keys())
            emb_dict = self.cache.get_embeddings(skills)
        except Exception:
            return self._minmax(bm25_weights)

        if len(emb_dict) < 10:
            return bm25_weights

        skill_list = list(emb_dict.keys())
        embs = np.array([emb_dict[s] for s in skill_list], dtype=np.float32)

        # PCA
        if config.PCA_ENABLED and len(embs) > config.PCA_MIN_SAMPLES and embs.shape[1] > config.PCA_MIN_FEATURES:
            from sklearn.decomposition import PCA

            n_comp = min(config.PCA_TARGET_DIM, len(embs) - 1, embs.shape[1])
            if n_comp < embs.shape[1]:
                embs = PCA(n_components=n_comp, random_state=config.GLOBAL_RANDOM_SEED).fit_transform(embs)
                logger.info(f"PCA: {emb_dict[skill_list[0]].shape[0]} → {n_comp}")

        embs = torch.tensor(embs)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        sim = torch.matmul(embs, embs.T).mean(dim=1).cpu().numpy()

        bm25_vals = np.array([bm25_weights.get(s, 0.0) for s in skill_list])
        bm25_norm = (bm25_vals - bm25_vals.min()) / (bm25_vals.max() - bm25_vals.min() + 1e-8)
        sem_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        hybrid = {}
        alpha, beta = 0.65, 0.35
        for i, skill in enumerate(skill_list):
            hybrid[skill] = round(float(alpha * bm25_norm[i] + beta * sem_norm[i]), 4)

        logger.info(f"Гибридные веса готовы: {len(hybrid)} навыков")
        return hybrid

    def _minmax(self, w):
        vals = np.array(list(w.values()))
        vmin, vmax = vals.min(), vals.max()
        return {s: round((w[s] - vmin) / (vmax - vmin), 4) for s in w} if vmax > vmin else w
