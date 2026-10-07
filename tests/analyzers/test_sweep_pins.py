"""Замеры, зафиксированные тестами (Exp-sweep, decomposition, rest-canonical)."""

from src.analyzers.comparison.comparator import BM25_WEIGHT, JACCARD_WEIGHT
from src.analyzers.comparison.embedding_comparator import (
    COSINE_WEIGHT,
    EmbeddingComparator,
)
from src.parsing.skills.skill_normalizer import SkillNormalizer


def test_ensemble_weights_from_sweep():
    """Decomposition: new weights + thr 0.7 → F1 0.9307 vs 0.9135 base,
    precision flat 0.9985. Порог 0.6: zero-overlap synonym-пары дают
    P 0.955 / R 0.651 против 0.961 / 0.566 на 0.7 — принято."""
    assert COSINE_WEIGHT == 0.5
    assert (JACCARD_WEIGHT, BM25_WEIGHT) == (0.1, 0.25)
    assert EmbeddingComparator().similarity_threshold == 0.6


def test_rest_api_canonical_not_truncated():
    """SYNONYM_MAP был инвертирован: 'rest api' -> 'rest' (2931 упоминание)."""
    assert SkillNormalizer.normalize("REST API").unwrap() == "rest api"
    assert SkillNormalizer.normalize("rest").unwrap() == "rest api"


def test_homoglyph_fold_lookup():
    """Кириллические lookalikes сворачиваются перед canon-lookup."""
    assert SkillNormalizer.normalize("с++").unwrap() == "c++"
