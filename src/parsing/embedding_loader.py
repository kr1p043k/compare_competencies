# src/parsing/embedding_loader.py
import logging
from sentence_transformers import SentenceTransformer
from src import config

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model(model_name: str = None):
    global _embedding_model
    if _embedding_model is None:
        model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Загрузка модели эмбеддингов: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model.eval()
    return _embedding_model