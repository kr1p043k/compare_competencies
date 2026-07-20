import time
import structlog
from typing import Any

from src import config
from src.monitoring.metrics import llm_requests_total, llm_request_duration_seconds, llm_token_usage, llm_response_length

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for LLMClient. Install with: pip install openai>=1.68.0"
            )
        self.base_url = base_url or config.OLLAMA_URL
        self.model = model or config.OLLAMA_MODEL
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key or "ollama",
            timeout=60.0,
            max_retries=2,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> Any:
        temperature = temperature if temperature is not None else config.QWEN_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else config.QWEN_MAX_TOKENS

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            llm_requests_total.labels(model=self.model, status="ok").inc()
            if hasattr(response, "usage") and response.usage:
                if response.usage.prompt_tokens:
                    llm_token_usage.labels(model=self.model, type="prompt").inc(response.usage.prompt_tokens)
                if response.usage.completion_tokens:
                    llm_token_usage.labels(model=self.model, type="completion").inc(response.usage.completion_tokens)
            if not stream and hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content or ""
                llm_response_length.labels(model=self.model).observe(len(content))
            return response
        except Exception:
            llm_requests_total.labels(model=self.model, status="error").inc()
            raise
        finally:
            duration = time.time() - start
            llm_request_duration_seconds.labels(model=self.model).observe(duration)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, base_url={self.base_url!r})"
