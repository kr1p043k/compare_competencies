"""LLM skill extraction (opt-in, fallback-first).

RU description -> JSON list of skill names STRICTLY from a provided market
vocabulary slice (top-200 by frequency + latin-token exacts, prompt <3000
chars). temperature 0.0, max_tokens 300. Validator drops any name not in the
full vocab (case-insensitive) and dedupes preserving order.

Any exception/timeout -> return [] (never raises), so callers keep current
behavior when the LLM endpoint is unreachable.

Flag: src.config.LLM_EXTRACT (default False).
Cache task 'extract' via DB-backed src.services.llm_cache
(get_cached/put_cached with model+prompt) plus an L1 in-memory cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:  # DB-backed shared cache (parallel worker): get_cached(task, model, prompt)
    from src.services.llm_cache import (  # type: ignore
        get_cached as _db_get_cached,
    )
    from src.services.llm_cache import (
        put_cached as _db_put_cached,
    )
    _DB_CACHE = True
except Exception:
    _DB_CACHE = False

# L1 in-memory cache (spec fallback names: get_cached/put_cached).
_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()
_TTL_S = 3600.0
_MAX_ENTRIES = 2000


def _evict_locked(now: float) -> None:
    dead = [k for k, (exp, _) in _CACHE.items() if exp <= now]
    for k in dead:
        _CACHE.pop(k, None)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.pop(next(iter(_CACHE)), None)


def get_cached(task: str, key: str) -> Any | None:
    """Return L1 cached value for (task, key) or None on miss/expiry."""
    now = time.time()
    ckey = "%s:%s" % (task, key)
    with _LOCK:
        hit = _CACHE.get(ckey)
        if hit is None:
            return None
        exp, val = hit
        if exp <= now:
            _CACHE.pop(ckey, None)
            return None
        return val


def put_cached(task: str, key: str, value: Any, ttl_s: float = _TTL_S) -> None:
    """Store L1 value for (task, key) with TTL (seconds)."""
    now = time.time()
    ckey = "%s:%s" % (task, key)
    with _LOCK:
        _evict_locked(now)
        _CACHE[ckey] = (now + max(1.0, float(ttl_s)), value)


def _model_of(client: Any) -> str:
    try:
        m = getattr(client, "model", None)
        if m:
            return str(m)
    except Exception:
        pass
    try:
        from src import config as _cfg
        return str(getattr(_cfg, "OLLAMA_MODEL", "") or "qwen3.6:latest")
    except Exception:
        return "qwen3.6:latest"


_LATIN_RE = re.compile(r"[A-Za-z]")
_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)

_PROMPT_BUDGET = 3000
_DESC_BUDGET = 800
_EXTRACT_MAX_TOKENS = 300
_EXTRACT_TEMPERATURE = 0.0


def _norm(name: str) -> str:
    return " ".join((name or "").lower().split())


def _vocab_slice(vocab_names: list[str], limit: int = 200) -> list[str]:
    """Top-<limit> names (input order = frequency order) + latin-token exacts."""
    out: list[str] = []
    seen: set[str] = set()
    for n in vocab_names[:limit]:
        s = (n or "").strip()
        if s and _norm(s) not in seen:
            seen.add(_norm(s))
            out.append(s)
    for n in vocab_names[limit:]:
        s = (n or "").strip()
        if s and _LATIN_RE.search(s) and _norm(s) not in seen:
            seen.add(_norm(s))
            out.append(s)
    return out


def _build_prompt(description: str, vocab_slice: list[str]) -> str:
    desc = " ".join((description or "").split())[:_DESC_BUDGET]
    names = list(vocab_slice)
    while names and len(", ".join(names)) + len(desc) + 400 > _PROMPT_BUDGET:
        names.pop()
    prompt = (
        "Izvleki navyki iz opisaniya. Otvet TOLKO JSON-spiskom strok. "
        "Kazhdoe imya - STROGO iz slovarya nizhe, doslovno, bez izmeneniy. "
        "Nichego ne vydumyvay. Esli sovpadeniy net - verni [].\n"
        "Slovar: %s\n"
        "Opisanie: %s\n"
        "JSON:"
    ) % (", ".join(names), desc)
    return prompt[:_PROMPT_BUDGET]


def _parse_json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except Exception:
        m = _JSON_ARRAY_RE.search(text or "")
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if isinstance(x, (str, int, float))]


def _validate(candidates: list[str], vocab_names: list[str]) -> list[str]:
    canon = {_norm(n): (n or "").strip() for n in vocab_names if (n or "").strip()}
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = _norm(str(c))
        if key in canon and key not in seen:
            seen.add(key)
            out.append(canon[key])
    return out


def _chat_guarded(client: Any, messages: list[dict[str, str]], timeout_s: float) -> str:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(client.chat, messages, _EXTRACT_TEMPERATURE, _EXTRACT_MAX_TOKENS)
        resp = fut.result(timeout=timeout_s)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def extract_skills(
    description: str,
    vocab_names: list[str],
    client: Any | None = None,
    use_cache: bool = True,
) -> list[str]:
    """Extract market skill names from a RU description. Never raises."""
    try:
        if not (description or "").strip() or not vocab_names:
            return []
        from src import config as _cfg
        timeout_s = float(getattr(_cfg, "LLM_TIMEOUT_S", 20) or 20)

        vocab = [(v or "").strip() for v in vocab_names if (v or "").strip()]
        if not vocab:
            return []
        prompt = _build_prompt(description, _vocab_slice(vocab))
        cache_key = ""
        if use_cache:
            sig = hashlib.sha256(
                "|".join(sorted(_norm(v) for v in vocab)).encode("utf-8")
            ).hexdigest()[:16]
            cache_key = hashlib.sha256(
                (_norm(description) + "::" + sig).encode("utf-8")
            ).hexdigest()
            hit = get_cached("extract", cache_key)
            if isinstance(hit, list):
                return _validate([str(x) for x in hit], vocab)

        if client is None:
            from src.services.llm_client import LLMClient
            client = LLMClient()
        model = _model_of(client)
        if use_cache and _DB_CACHE:  # shared DB cache keyed by (task, model, prompt)
            try:
                raw_hit = _db_get_cached("extract", model, prompt)
            except Exception:
                raw_hit = None
            if isinstance(raw_hit, str) and raw_hit.strip():
                result = _validate(_parse_json_list(raw_hit), vocab)
                put_cached("extract", cache_key, result,
                           ttl_s=3600.0 if result else 120.0)
                return result
        messages = [
            {"role": "system", "content": "Ty izvlekaesh navyki iz teksta. Otvechaesh tolko JSON."},
            {"role": "user", "content": prompt},
        ]
        text = _chat_guarded(client, messages, timeout_s)
        result = _validate(_parse_json_list(text), vocab)
        if use_cache:
            put_cached("extract", cache_key, result, ttl_s=3600.0 if result else 120.0)
            if _DB_CACHE and text:
                try:
                    _db_put_cached("extract", model, prompt, text)
                except Exception:
                    pass
        return result
    except Exception:
        return []
