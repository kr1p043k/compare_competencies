"""Настройки RAG: читаются из src.config с безопасными дефолтами.

Отдельный слой нужен, чтобы подсистему можно было тестировать без .env и
чтобы retrieval-профили были данными, а не магическими константами в коде.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

#: Профили retrieval: источники, веса веток, реранк, размер выдачи.
PROFILES: dict[str, dict[str, Any]] = {
    "chat_qa": {
        "sources": None,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "entity_boost": 0.15,
        "top_n": 8,
        "rerank": False,
    },
    "rpd_gap_explain": {
        "sources": ("rpd", "krm"),
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "entity_boost": 0.25,
        "top_n": 10,
        "rerank": False,
    },
    "market_evidence": {
        "sources": ("vacancies", "publications"),
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "entity_boost": 0.2,
        "top_n": 12,
        "rerank": False,
    },
    "curriculum_design": {
        "sources": ("rpd", "publications", "docs"),
        "dense_weight": 0.4,
        "sparse_weight": 0.6,
        "entity_boost": 0.1,
        "top_n": 10,
        "rerank": False,
    },
    "competency_lookup": {
        "sources": ("krm",),
        "dense_weight": 0.0,
        "sparse_weight": 1.0,
        "entity_boost": 0.5,
        "top_n": 5,
        "rerank": False,
    },
}


@dataclass(frozen=True, slots=True)
class RagSettings:
    enabled: bool = False
    profile_default: str = "chat_qa"
    dense_k: int = 50
    sparse_k: int = 50
    entity_k: int = 20
    top_n: int = 8
    min_score: float = 0.35
    context_budget: int = 8000
    timeout_s: float = 1.5
    index_batch: int = 64
    mmr_lambda: float = 0.7
    max_chunk_tokens: int = 700
    min_chunk_tokens: int = 40
    acl_default: str = "internal"
    #: L1-кэш ответов retrieval (секунды); 0 — без кэша.
    cache_ttl_s: float = 60.0
    #: RRF-константа: 60 — стандарт из статьи Cormack et al.
    rrf_k: int = 60
    #: Максимум чанков одного документа в выдаче.
    max_per_document: int = 3
    #: Доля бюджета контекста под доказательства.
    evidence_share: float = 0.45

    def profile(self, name: str | None = None) -> dict[str, Any]:
        key = name or self.profile_default
        return PROFILES.get(key, PROFILES["chat_qa"])

    @property
    def profile_names(self) -> list[str]:
        return sorted(PROFILES)


def _cfg(name: str, default: Any) -> Any:
    try:
        from src import config

        return getattr(config, name, default)
    except Exception:
        return default


@lru_cache(maxsize=1)
def get_settings() -> RagSettings:
    return RagSettings(
        enabled=bool(_cfg("RAG_ENABLED", False)),
        profile_default=str(_cfg("RAG_PROFILE_DEFAULT", "chat_qa")),
        dense_k=int(_cfg("RAG_DENSE_K", 50)),
        sparse_k=int(_cfg("RAG_SPARSE_K", 50)),
        entity_k=int(_cfg("RAG_ENTITY_K", 20)),
        top_n=int(_cfg("RAG_TOP_N", 8)),
        min_score=float(_cfg("RAG_MIN_SCORE", 0.35)),
        context_budget=int(_cfg("RAG_CONTEXT_BUDGET", 8000)),
        timeout_s=float(_cfg("RAG_TIMEOUT_S", 1.5)),
        index_batch=int(_cfg("RAG_INDEX_BATCH", 64)),
        mmr_lambda=float(_cfg("RAG_MMR_LAMBDA", 0.7)),
        max_chunk_tokens=int(_cfg("RAG_MAX_CHUNK_TOKENS", 700)),
        min_chunk_tokens=int(_cfg("RAG_MIN_CHUNK_TOKENS", 40)),
        acl_default=str(_cfg("RAG_ACL_DEFAULT", "internal")),
        cache_ttl_s=float(_cfg("RAG_CACHE_TTL_S", 60.0)),
    )


def reset_settings_cache() -> None:
    """Сброс кэша настроек (тесты, смена env в рантайме)."""
    get_settings.cache_clear()


@dataclass(slots=True)
class AclScope:
    """Иерархия видимости: student ⊂ teacher ⊂ admin."""

    level: str = "internal"
    allowed: tuple[str, ...] = field(default_factory=lambda: ("public", "internal"))

    @classmethod
    def for_role(cls, role: str | None) -> AclScope:
        role = (role or "").lower()
        if role in ("admin", "rop"):
            return cls(level="admin", allowed=("public", "internal", "teacher", "admin"))
        if role in ("teacher",):
            return cls(level="teacher", allowed=("public", "internal", "teacher"))
        if role in ("student",):
            return cls(level="student", allowed=("public", "internal"))
        return cls(level="anon", allowed=("public",))
