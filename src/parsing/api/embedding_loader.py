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
        kwargs = {}
        if config.HF_TOKEN:
            kwargs["token"] = config.HF_TOKEN.get_secret_value()
        _embedding_model = SentenceTransformer(
            model_name,
            device="cpu",
            model_kwargs={"low_cpu_mem_usage": True},
            **kwargs,
        )
        _embedding_model.eval()
        logger.info("embedding_model_loaded")
    return _embedding_model
