# src/parsing/embedding_loader.py
import logging
from sentence_transformers import SentenceTransformer
from src import config

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        _embedding_model.eval()
    return _embedding_model