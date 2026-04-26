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
        _embedding_model = SentenceTransformer(
            model_name,
            use_auth_token=config.HF_TOKEN if config.HF_TOKEN else None
        )
        _embedding_model.eval()
        logger.info("✅ Модель эмбеддингов успешно загружена")
    return _embedding_model