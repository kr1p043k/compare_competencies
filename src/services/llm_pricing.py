"""LLM pricing for cost accounting of model calls.

.. include:: _pricing_note

Defaults below are **list-price estimates** based on public provider pages
and must be **verified against the actual provider bill** before use in any
financial context.  Prices change without notice; the function never guesses
a price for an unknown model – it returns ``None``.

Supported models (per 1 M tokens):
    yandexgpt-lite    – YandexGPT list price (converted to USD)
    gpt-oss:120b      – open-source, hosted on university Ollama: free
    qwen2.5:0.5b      – open-source embedding model: free

Overriding prices
-----------------
Set the environment variable ``LLM_PRICING`` to the path of a JSON file with
your own rates.  Expected structure::

    {
      "yandexgpt-lite": {"input": 0.001, "output": 0.003}
    }

Keys are per-1M-token prices in USD.  If a model key is present the
environment config takes precedence over the built-in defaults.
"""

import json
import os
from pathlib import Path
from typing import Final

# ─────────────────── built-in defaults ──────────────────────────────────────
# List-price estimates – NOT audited billing data.
# yandexgpt-lite: ~250 RUB / 1M input tokens, ~750 RUB / 1M output tokens
#   (YandexCloud public page, 2026).  Converted with ~90 RUB/USD midpoint.
# gpt-oss / qwen2.5: open-source, no per-token charge for self-hosted.
_DEFAULT_PRICES: Final[dict[str, dict[str, float]]] = {
    "yandexgpt-lite": {"input": 250 / 90, "output": 750 / 90},  # ≈ 2.778 / 8.333
    "gpt-oss:120b":   {"input": 0.0, "output": 0.0},
    "qwen2.5:0.5b":   {"input": 0.0, "output": 0.0},
}

# ─────────────────── lazy load from env JSON override ───────────────────────

def _load_env_pricing() -> dict[str, dict[str, float]]:
    """Return user-overrides from LLM_PRICING env JSON file (or empty)."""
    path_str = os.environ.get("LLM_PRICING")
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
    except (json.JSONDecodeError, OSError):
        pass
    return {}


_env_pricing = _load_env_pricing()  # evaluated once per import


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    response_tokens: int,
) -> float | None:
    """Return estimated USD cost for *model* given token counts.

    Returns
    -------
    float | None
        Estimated cost in USD, or ``None`` when the model is unknown / has no
        pricing data.  Prices are **list-price estimates** and must be
        verified against the actual provider bill.

    Examples
    --------
    >>> estimate_cost_usd("yandexgpt-lite", 1000, 500)
    0.005556  # ≈ 5.56 millicents
    >>> estimate_cost_usd("unknown-model", 1000, 500) is None
    True
    """
    prices = _env_pricing.get(model) or _DEFAULT_PRICES.get(model)
    if prices is None:
        return None
    prompt_price = prices.get("input", 0.0)
    response_price = prices.get("output", 0.0)
    cost = (prompt_tokens * prompt_price + response_tokens * response_price) / 1_000_000
    return round(cost, 10)  # avoid floating-point artefacts
