import structlog
from sentence_transformers import SentenceTransformer

from src import config

logger = structlog.get_logger(__name__)

_embedding_model = None


def get_embedding_model(model_name: str = None):
    global _embedding_model
    if _embedding_model is None:
        model_name = model_name or config.EMBEDDING_MODEL
        logger.info("loading_embedding_model", model=model_name)
        _embedding_model = SentenceTransformer(model_name, use_auth_token=config.HF_TOKEN if config.HF_TOKEN else None)
        _embedding_model.eval()
        logger.info("embedding_model_loaded")
    return _embedding_model
