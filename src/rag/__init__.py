"""RAG-подсистема: индексация корпуса и grounded-retrieval для LLM.

Точки входа:
  * ``src.rag.service.get_service()`` — фасад для API/чата;
  * ``src.rag.indexer.index_documents`` — индексация;
  * ``python -m src.cli rag-index`` — CLI.

Фича за флагом ``RAG_ENABLED`` (по умолчанию False): без него ничего не
индексируется и не ищется, поведение llm_chat не меняется.
"""

from src.rag.context import assemble_context, fallback_fact_block, sanitize_evidence
from src.rag.embedder import Embedder, EmbeddingUnavailableError, get_embedder
from src.rag.indexer import SPEC_VERSION, IndexStats, index_documents
from src.rag.retriever import (
    HybridRetriever,
    aggregate_by_document,
    dedupe,
    hit_confidence,
    mmr_select,
    rrf_fuse,
)
from src.rag.service import RagResult, RagService, clear_cache, get_service, reset_service
from src.rag.settings import PROFILES, AclScope, RagSettings, get_settings
from src.rag.store import RagStore
from src.rag.types import (
    Chunk,
    Citation,
    Document,
    RagContext,
    RagQuery,
    RetrievalHit,
    content_hash,
)

__all__ = [
    "assemble_context",
    "fallback_fact_block",
    "sanitize_evidence",
    "Embedder",
    "EmbeddingUnavailableError",
    "get_embedder",
    "SPEC_VERSION",
    "IndexStats",
    "index_documents",
    "HybridRetriever",
    "aggregate_by_document",
    "dedupe",
    "hit_confidence",
    "mmr_select",
    "rrf_fuse",
    "RagResult",
    "RagService",
    "clear_cache",
    "get_service",
    "reset_service",
    "PROFILES",
    "AclScope",
    "RagSettings",
    "get_settings",
    "RagStore",
    "Chunk",
    "Citation",
    "Document",
    "RagContext",
    "RagQuery",
    "RetrievalHit",
    "content_hash",
]
