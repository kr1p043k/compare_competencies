import os
import structlog
from sentence_transformers import SentenceTransformer

from src import config

logger = structlog.get_logger(__name__)

_embedding_model = None

HF_MIRROR = "https://hf-mirror.com"


def get_embedding_model(model_name: str = None):
    global _embedding_model
    if _embedding_model is None:
        model_name = model_name or config.EMBEDDING_MODEL
        kwargs = {}
        if config.HF_TOKEN:
            os.environ["HF_TOKEN"] = config.HF_TOKEN.get_secret_value()
            kwargs["token"] = config.HF_TOKEN.get_secret_value()

        for attempt, endpoint in enumerate([None, HF_MIRROR]):
            try:
                if endpoint:
                    os.environ["HF_ENDPOINT"] = endpoint
                    logger.info("loading_embedding_model_via_mirror", model=model_name, mirror=endpoint)
                else:
                    logger.info("loading_embedding_model", model=model_name)

                _embedding_model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    model_kwargs={"low_cpu_mem_usage": True},
                    **kwargs,
                )
                _embedding_model.eval()
                logger.info("embedding_model_loaded", source="mirror" if endpoint else "huggingface")
                return _embedding_model
            except Exception as e:
                if endpoint:
                    logger.error("embedding_model_failed_mirror", model=model_name, error=str(e))
                    raise
                logger.warning("embedding_model_failed_try_mirror", model=model_name, error=str(e))

        raise RuntimeError(f"Failed to load embedding model {model_name} from both sources")
    return _embedding_model
