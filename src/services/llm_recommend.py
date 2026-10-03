"""LLM recommendation enhancement (opt-in, fallback-first).

Builds ON TOP of base recommendations: the LLM re-ranks / explains, never
invents skill names (validator enforces names strictly subset of base names).
Any failure -> return the base input unchanged, so endpoints never 500
because of the LLM.

Flags: src.config.LLM_ENHANCE_STUDENT / LLM_ENHANCE_TEACHER (default False).
Cache task 'recommend' via DB-backed src.services.llm_cache
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
        return str(getattr(_cfg, "OLLAMA_MODEL", "") or "gpt-oss:120b")
    except Exception:
        return "gpt-oss:120b"


def _rest_copy(rest: list) -> list:
    return [dict(r) if isinstance(r, dict) else r for r in rest]


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT_BUDGET = 3000
_CONTEXT_BUDGET = 600
_RECOMMEND_MAX_TOKENS = 800
_RECOMMEND_TEMPERATURE = 0.2
_MAX_BASE_ITEMS = 30


def _norm(name: str) -> str:
    return " ".join((name or "").lower().split())


def _fit(text: Any, budget: int) -> str:
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False, default=str)
        except Exception:
            text = str(text)
    return " ".join(text.split())[:budget]


def _skill_of(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("skill") or "").strip()
    return ""


def _score_of(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("score", item.get("weight", ""))
    return ""


def _build_prompt(context_txt: str, items: list[dict]) -> str:
    ctx = _fit(context_txt, _CONTEXT_BUDGET)
    shown = items[:_MAX_BASE_ITEMS]
    lines: list[str] = []
    for i, r in enumerate(shown):
        lines.append("%d. %s (score=%s)" % (i + 1, _skill_of(r), _score_of(r)))
    base_txt = "\n".join(lines)
    while base_txt and len(base_txt) + len(ctx) + 700 > _PROMPT_BUDGET:
        shown = shown[:-1]
        lines = lines[:-1]
        base_txt = "\n".join(lines)
    prompt = (
        "Pere-ranzhiruy i obyasni rekomendacii. Otvet TOLKO JSON-obektom "
        '{"recommendations": [{"skill": ..., "reason": ...}]}. '
        "Imya skill - STROGO iz spiska nizhe, doslovno. Novye navyki zaprescheny. "
        "Poryadok - po prioritetu. Reason - 1 fraza.\n"
        "Kontekst: %s\n"
        "Spisok:\n%s\n"
        "JSON:"
    ) % (ctx, base_txt)
    return prompt[:_PROMPT_BUDGET]


def _parse_llm_items(text: str) -> list[dict]:
    data: Any = None
    try:
        data = json.loads(text or "")
    except Exception:
        m = _JSON_OBJ_RE.search(text or "")
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(data, dict):
        data = data.get("recommendations", [])
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for r in data:
        if isinstance(r, dict) and str(r.get("skill") or "").strip():
            out.append({"skill": str(r["skill"]).strip(), "reason": str(r.get("reason") or "")[:500]})
    return out


def _apply_rerank(valid: list[dict], llm_items: list[dict]) -> list[dict] | None:
    """Merge LLM order/explanations over base items. None = no valid signal."""
    by_norm = {_norm(_skill_of(r)): r for r in valid}
    if not by_norm:
        return None
    merged: list[dict] = []
    seen: set[str] = set()
    for li in llm_items:
        key = _norm(li["skill"])
        if key in by_norm and key not in seen:
            seen.add(key)
            item = dict(by_norm[key])
            if li.get("reason"):
                item["llm_reason"] = li["reason"]
            merged.append(item)
    if not merged:
        return None
    for r in valid:  # base items the LLM dropped keep original relative order
        if _norm(_skill_of(r)) not in seen:
            merged.append(dict(r))
    return merged


def _chat_guarded(
    client: Any, messages: list[dict[str, str]], timeout_s: float
) -> str:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(
            client.chat, messages, _RECOMMEND_TEMPERATURE, _RECOMMEND_MAX_TOKENS
        )
        resp = fut.result(timeout=timeout_s)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def _timeout_s() -> float:
    try:
        from src import config as _cfg
        return float(getattr(_cfg, "LLM_TIMEOUT_S", 20) or 20)
    except Exception:
        return 20.0


def _cache_key(ns: str, context_txt: str, valid: list[dict]) -> str:
    sig = "|".join(
        "%s=%s" % (_norm(_skill_of(r)), _score_of(r)) for r in valid[:_MAX_BASE_ITEMS]
    )
    return hashlib.sha256((ns + "::" + _fit(context_txt, 400) + "::" + sig).encode("utf-8")).hexdigest()


def _enhance_core(
    ns: str,
    context_txt: str,
    base_recs: Any,
    client: Any | None,
    use_cache: bool,
) -> Any:
    """Shared dict|list enhancement. Returns base input on any failure."""
    if isinstance(base_recs, dict):
        items = base_recs.get("recommendations")
        wrap = True
    elif isinstance(base_recs, list):
        items = base_recs
        wrap = False
    else:
        return base_recs
    if not isinstance(items, list) or not items:
        return base_recs
    valid = [r for r in items if isinstance(r, dict) and _skill_of(r)]
    rest = [r for r in items if not (isinstance(r, dict) and _skill_of(r))]
    if not valid:
        return base_recs

    timeout = _timeout_s()
    ckey = _cache_key(ns, context_txt, valid)
    prompt = _build_prompt(context_txt, valid)
    if use_cache:
        hit = get_cached("recommend", ckey)
        if isinstance(hit, list) and hit:
            merged = _apply_rerank(valid, hit)
            if merged is not None:
                return _wrap(base_recs, wrap, merged + _rest_copy(rest))
    if client is None:
        from src.services.llm_client import LLMClient
        client = LLMClient()
    model = _model_of(client)
    if use_cache and _DB_CACHE:  # shared DB cache keyed by (task, model, prompt)
        try:
            raw_hit = _db_get_cached("recommend", model, prompt)
        except Exception:
            raw_hit = None
        if isinstance(raw_hit, str) and raw_hit.strip():
            merged = _apply_rerank(valid, _parse_llm_items(raw_hit))
            if merged is not None:
                full = merged + _rest_copy(rest)
                put_cached("recommend", ckey,
                           [{"skill": _skill_of(r),
                             "reason": r.get("llm_reason", "")} for r in merged],
                           ttl_s=3600.0)
                return _wrap(base_recs, wrap, full)
    messages = [
        {"role": "system", "content": "Ty pomogaesh s rekomendaciyami navykov. Otvechaesh tolko JSON."},
        {"role": "user", "content": prompt},
    ]
    text = _chat_guarded(client, messages, timeout)
    llm_items = _parse_llm_items(text)
    merged = _apply_rerank(valid, llm_items)
    if merged is None:
        return base_recs
    full = merged + _rest_copy(rest)
    if use_cache:
        put_cached("recommend", ckey, llm_items, ttl_s=3600.0)
        if _DB_CACHE and text:
            try:
                _db_put_cached("recommend", model, prompt, text)
            except Exception:
                pass
    return _wrap(base_recs, wrap, full)


def _wrap(base_recs: Any, wrap: bool, items: list) -> Any:
    if wrap:
        out = dict(base_recs)
        out["recommendations"] = items
        out["llm_enhanced"] = True
        return out
    return items


def enhance_student_recs(
    profile_summary: str,
    base_recs: dict,
    client: Any | None = None,
    use_cache: bool = True,
) -> dict:
    """Re-rank/explain student base recs via LLM. Base unchanged on failure."""
    try:
        out = _enhance_core("student", profile_summary, base_recs, client, use_cache)
        return out if isinstance(out, dict) else base_recs
    except Exception:
        return base_recs


def enhance_teacher_recs(
    discipline: str,
    gaps: Any,
    base_recs: Any = None,
    client: Any | None = None,
    use_cache: bool = True,
) -> Any:
    """Re-rank/explain teacher base recs via LLM. Base unchanged on failure."""
    try:
        if base_recs is None:
            return {}
        ctx = "Disciplina: %s. Probely: %s" % (_fit(discipline, 200), _fit(gaps, 800))
        return _enhance_core("teacher", ctx, base_recs, client, use_cache)
    except Exception:
        return base_recs if base_recs is not None else {}
