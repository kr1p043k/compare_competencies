"""Загрузчики источников RAG (сеть -> нормализованные Document).

Отдельно от ``chunking``: здесь только получение и нормализация данных,
чанкование — в ``src.rag.chunking``. Так один и тот же чанкер можно кормить
из API, из БД или из фикстуры в тестах.
"""

from src.rag.loaders.publications import (
    load_publications_documents,
    publications_to_documents,
)

__all__ = ["load_publications_documents", "publications_to_documents"]
