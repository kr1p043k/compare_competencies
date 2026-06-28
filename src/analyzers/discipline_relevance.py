"""Discipline-aware relevance scoring for teacher analysis recommendations.

Computes semantic relevance between a skill and an academic discipline
using KRM KSA texts and embedding cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from src import config

logger = structlog.get_logger(__name__)

CORE_THRESHOLD = 0.6
RELATED_THRESHOLD = 0.35
ADJACENT_THRESHOLD = 0.15


class DisciplineRelevance:
    level: str
    score: float
    combined: float

    def __init__(self, score: float):
        self.score = float(score)
        self.combined = max(0.0, float(score))
        if self.combined >= CORE_THRESHOLD:
            self.level = "CORE"
        elif self.combined >= RELATED_THRESHOLD:
            self.level = "RELATED"
        elif self.combined >= ADJACENT_THRESHOLD:
            self.level = "ADJACENT"
        else:
            self.level = "UNRELATED"


class DisciplineAwareScorer:
    def __init__(self, embedding_model: Any | None = None):
        self._model = embedding_model
        self._loaded = False
        self._disciplines: dict[str, dict] = {}
        self._discipline_texts: dict[str, list[str]] = {}
        self._discipline_embeddings: dict[str, np.ndarray] = {}

    def load(self, disciplines_path: Path | None = None) -> None:
        disciplines_path = disciplines_path or config.REFERENCE_DIR / "krm_disciplines_09.03.02.json"
        if not disciplines_path.exists():
            logger.warning("discipline_file_not_found", path=str(disciplines_path))
            return
        raw = json.loads(disciplines_path.read_text(encoding="utf-8"))
        direction = next(iter(raw.values()))
        for disc_name, disc_data in direction.get("disciplines", {}).items():
            texts: list[str] = []
            for comp_code in disc_data.get("competencies", []):
                texts.extend(disc_data.get("skills", {}).get(comp_code, []))
            if texts:
                self._disciplines[disc_name] = {"competencies": disc_data.get("competencies", []), "text_count": len(texts)}
                self._discipline_texts[disc_name] = texts
        self._loaded = True
        logger.info("discipline_relevance_loaded", disciplines=len(self._disciplines))

    def _ensure_embeddings(self) -> None:
        if self._discipline_embeddings or not self._loaded:
            return
        if self._model is None:
            from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
            self._model = EmbeddingProviderFactory.get()
        for disc_name, texts in self._discipline_texts.items():
            try:
                embs = self._model.encode(texts, show_progress_bar=False)
                mean_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                self._discipline_embeddings[disc_name] = mean_emb
            except Exception as exc:
                logger.warning("discipline_embedding_failed", discipline=disc_name, error=str(exc))
        logger.info("discipline_embeddings_computed", count=len(self._discipline_embeddings))

    def compute_relevance(self, skill_name: str, discipline_name: str | None = None) -> DisciplineRelevance:
        if not self._loaded:
            self.load()
        self._ensure_embeddings()
        if not self._discipline_texts:
            return DisciplineRelevance(0.0)
        if discipline_name and discipline_name in self._discipline_embeddings:
            disc_emb = self._discipline_embeddings[discipline_name]
        else:
            return DisciplineRelevance(0.0)
        if self._model is None:
            return DisciplineRelevance(0.0)
        try:
            skill_emb = self._model.encode([skill_name], show_progress_bar=False)[0]
            skill_norm = np.linalg.norm(skill_emb)
            if skill_norm > 0:
                skill_emb = skill_emb / skill_norm
            else:
                return DisciplineRelevance(0.0)
            sim = float(np.dot(skill_emb, disc_emb))
            return DisciplineRelevance(sim)
        except Exception as exc:
            logger.warning("relevance_compute_failed", skill=skill_name, error=str(exc))
            return DisciplineRelevance(0.0)

    def get_discipline_names(self) -> list[str]:
        return list(self._disciplines.keys())

    def get_discipline_embedding(self, name: str) -> np.ndarray | None:
        return self._discipline_embeddings.get(name)
