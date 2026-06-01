"""Protocol-based similarity engines.

Usage:
    jaccard = JaccardEngine()
    result = jaccard.compare(["python dev"], ["python developer"])

    # Wire into EmbeddingComparator:
    comp = EmbeddingComparator(extra_engines={"jaccard": (JaccardEngine(), 0.3)})
    result = comp.compare_student_to_market(student_skills, market_skills)
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import structlog
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

from src import Result, Ok, Err
from src.errors import DomainError

logger = structlog.get_logger(__name__)


ComparisonResult = dict[str, Any]


class SimilarityEngine(Protocol):
    def compare(
        self,
        student_skills: list[str],
        market_skills: list[str],
    ) -> Result[ComparisonResult, DomainError]: ...


class JaccardEngine:
    """Fuzzy string similarity via rapidfuzz token_sort_ratio.

    No embedding model needed — pure string matching.
    Catches typos and different word order ("python dev" vs "dev python").
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def compare(
        self,
        student_skills: list[str],
        market_skills: list[str],
    ) -> Result[ComparisonResult, DomainError]:
        best_sims: dict[str, float] = {}
        for ms in market_skills:
            best = max(
                (fuzz.token_sort_ratio(ss.lower(), ms.lower()) / 100.0 for ss in student_skills),
                default=0.0,
            )
            if best >= self.threshold:
                best_sims[ms] = best

        avg_sim = float(np.mean(list(best_sims.values()))) if best_sims else 0.0
        sorted_matches = sorted(
            [{"skill": k, "similarity": round(v, 4)} for k, v in best_sims.items()],
            key=lambda x: x["similarity"],
            reverse=True,
        )[:15]

        return Ok(dict(
            score=round(avg_sim, 4),
            weighted_coverage=round(avg_sim, 4),
            avg_similarity=round(avg_sim, 4),
            matches=sorted_matches,
            missing=[],
        ))


class EnsembleEngine:
    """Weighted average of multiple SimilarityEngine instances.

    Example:
        ensemble = EnsembleEngine({
            "cosine": (embedding_comparator, 0.6),
            "jaccard": (JaccardEngine(), 0.4),
        })
    """

    def __init__(self, engines: dict[str, tuple[SimilarityEngine, float]]):
        if not engines:
            raise ValueError("At least one engine required")
        self.engines = engines

    def compare(
        self,
        student_skills: list[str],
        market_skills: list[str],
    ) -> Result[dict, DomainError]:
        results: dict[str, ComparisonResult] = {}
        for name, (engine, _weight) in self.engines.items():
            match engine.compare(student_skills, market_skills):
                case Ok(res):
                    results[name] = res
                case Err(err):
                    logger.warning("engine_compare_failed", engine=name, error=str(err))
                    results[name] = dict(score=0.0, weighted_coverage=0.0, avg_similarity=0.0, matches=[], missing=[])

        total_weight = sum(w for _, w in self.engines.values())
        if total_weight <= 0:
            return Ok(dict(score=0.0, weighted_coverage=0.0, avg_similarity=0.0, matches=[], missing=[]))

        combined_score = sum(
            results[name].get("score", 0.0) * w
            for name, (_, w) in self.engines.items()
        ) / total_weight

        all_matches: dict[str, float] = {}
        for name, (_, _w) in self.engines.items():
            for m in results[name].get("matches", []):
                skill = m["skill"]
                existing = all_matches.get(skill, 0.0)
                all_matches[skill] = max(existing, m["similarity"])

        sorted_matches = sorted(
            [{"skill": k, "similarity": round(v, 4)} for k, v in all_matches.items()],
            key=lambda x: x["similarity"],
            reverse=True,
        )[:15]

        avg_sim = float(np.mean([m["similarity"] for m in sorted_matches])) if sorted_matches else 0.0
        engine_details = {name: dict(res) for name, res in results.items()}

        return Ok(dict(
            score=round(combined_score, 4),
            weighted_coverage=round(combined_score, 4),
            avg_similarity=round(avg_sim, 4),
            matches=sorted_matches,
            missing=[],
            details=engine_details,
        ))


class BM25Engine:
    """BM25 term-frequency similarity.

    Builds a BM25Okapi index over all market skill names.
    On compare(), scores each market skill by BM25 relevance
    to the student's skill set as a query.

    Orthogonal signal vs cosine (semantic) and Jaccard (fuzzy string):
    BM25 captures shared vocabulary / term overlap at the word level.
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus: list[str] = []
        self._corpus_lower: list[str] = []

    def fit(self, corpus: list[str]) -> None:
        self._corpus = corpus
        self._corpus_lower = [s.lower() for s in corpus]
        tokenized = [s.split() for s in self._corpus_lower]
        self._bm25 = BM25Okapi(tokenized)

    def compare(
        self,
        student_skills: list[str],
        market_skills: list[str],
    ) -> Result[ComparisonResult, DomainError]:
        if not student_skills or self._bm25 is None:
            return Ok(dict(score=0.0, weighted_coverage=0.0, avg_similarity=0.0, matches=[], missing=[]))

        query = " ".join(student_skills).lower().split()
        if not query:
            return Ok(dict(score=0.0, weighted_coverage=0.0, avg_similarity=0.0, matches=[], missing=[]))

        doc_scores = self._bm25.get_scores(query)
        idx_map = {s: i for i, s in enumerate(self._corpus_lower)}

        best: dict[str, float] = {}
        for ms in market_skills:
            ms_lower = ms.lower()
            idx = idx_map.get(ms_lower)
            if idx is not None:
                s = float(doc_scores[idx])
                if s > 0:
                    best[ms] = s

        if not best:
            return Ok(dict(score=0.0, weighted_coverage=0.0, avg_similarity=0.0, matches=[], missing=[]))

        vals = np.array(list(best.values()))
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            for k in best:
                best[k] = round((best[k] - vmin) / (vmax - vmin), 4)
        else:
            best = {k: 1.0 for k in best}

        avg = float(np.mean(list(best.values())))
        sorted_matches = sorted(
            [{"skill": k, "similarity": round(v, 4)} for k, v in best.items()],
            key=lambda x: x["similarity"],
            reverse=True,
        )[:15]

        return Ok(dict(
            score=round(avg, 4),
            weighted_coverage=round(avg, 4),
            avg_similarity=round(avg, 4),
            matches=sorted_matches,
            missing=[],
        ))
