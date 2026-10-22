"""Замеры, зафиксированные тестами (Exp-sweep, rest-canonical).

Ensemble blend удалён 2026-10-11 (prodval: ни один безопасный дизайн не
выжил — full-replace добавляет мусор без порога, lexical veto убивает
истинные переводы 19/24, rescue-коридор пуст). Evidence весов сохранена
в git-истории и MEMORY.md. Здесь пинится только дефолтный порог 0.6.
"""

from src.analyzers.comparison.embedding_comparator import EmbeddingComparator
from src.parsing.skills.skill_normalizer import SkillNormalizer


def test_comparator_default_threshold():
    """Порог 0.6: zero-overlap synonym-пары дают P 0.955 / R 0.651 против
    0.961 / 0.566 на 0.7; расширенный набор 197 пар: F1 0.70 vs 0.65.
    Прод-workload 28 дисков: 406 vs 412 матчей, [0.5,0.6) — мусор."""
    assert EmbeddingComparator().similarity_threshold == 0.6


def test_rest_api_canonical_not_truncated():
    """SYNONYM_MAP был инвертирован: 'rest api' -> 'rest' (2931 упоминание)."""
    assert SkillNormalizer.normalize("REST API").unwrap() == "rest api"
    assert SkillNormalizer.normalize("rest").unwrap() == "rest api"


def test_homoglyph_fold_lookup():
    """Кириллические lookalikes сворачиваются перед canon-lookup."""
    assert SkillNormalizer.normalize("с++").unwrap() == "c++"
