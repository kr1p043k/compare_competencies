"""Hub-gate: вето для чисто-семантических матчей серой зоны к хаб-навыкам.

Мотивация (prodval 2026-10-11, n=128 размеченных пар):
- в зоне [0.6, 0.75) косинус инвертирован (AUC 0.000): чем выше скор, тем вернее мусор;
- hubness (сколько запросов матчится к навыку с cos>=0.5) даёт AUC 0.868;
- правило kill if hub>=200 AND morph<0.2: P=0.971, R=0.680 (66 мусора
  убито ценой 2 истинных; см. C:/forCode/temp/fp_tmp.py разбор);
- exact/fuzzy совпадения не трогаем: вето только для sim в [lo, hi),
  лексически очевидное (morph>=M) выживает.

Хаб-таблица: data/reference/market_hubs.json (пересборка — см. _meta).
Нет таблицы — вето не работает (warning, не падение).
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from src import config

logger = structlog.get_logger(__name__)

_HUBS: dict[str, int] | None = None
_MORPH = None


def _hub_table() -> dict[str, int]:
    global _HUBS
    if _HUBS is not None:
        return _HUBS
    _HUBS = {}
    try:
        path = Path(config.HUB_TABLE_PATH)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            _HUBS = {str(k): int(v) for k, v in doc.get("hubs", {}).items()}
        else:
            logger.warning("hub_table_missing", path=str(path))
    except Exception as exc:
        logger.warning("hub_table_load_failed", error=str(exc))
        _HUBS = {}
    return _HUBS


def _morph_overlap(a: str, b: str) -> float:
    global _MORPH
    try:
        if _MORPH is None:
            import pymorphy3

            _MORPH = pymorphy3.MorphAnalyzer()
        from src.analyzers.skill_matcher import fold_script
        # Токен живёт если есть хоть один alnum-символ: '1c'/'c++'/'c#' —
        # легитимные символьные навыки (2026-10-12: isalpha() их убивал,
        # morph=0.0 ложно выполнял оба условия вето).
        # Fold обеих сторон: кросс-скрипт 'разработка 1с' vs '1c' (2026-10-12).
        def _toks(s: str) -> set[str]:
            return {_MORPH.parse(w)[0].normal_form
                    for w in fold_script(s).lower().split()
                    if any(ch.isalnum() for ch in w)}
        ta, tb = _toks(a), _toks(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)
    except Exception:
        return 0.0


def veto_match(query: str, market_skill: str, similarity: float) -> bool:
    """True — отбросить совпадение (мусор с P~97%).

    Два непересекающихся правила (prodval 2026-10-11, n=128):
    - gray: sim в [lo, hi), hub>=T, morph<M (P=0.971, 2 известных FP);
    - ultra: hub>=ULTRA_T AND morph==0 строго, любой sim>=lo (26 мусора,
      НОЛЬ истинных на fit-сете; max hub истинных <1000).
    """
    sim = float(similarity)
    if sim < config.HUB_GATE_SIM_LO:
        return False
    hub = _hub_table().get(market_skill, _hub_table().get(market_skill.lower(), 0))
    if hub >= config.HUB_GATE_ULTRA_HUB:
        return _morph_overlap(query, market_skill) == 0.0
    if not (sim < config.HUB_GATE_SIM_HI):
        return False
    if hub < config.HUB_GATE_HUB_T:
        return False
    return _morph_overlap(query, market_skill) < config.HUB_GATE_MORPH_M


def filter_matches(
    disc_skills: list[str],
    matches: list[dict],
    comp=None,
) -> tuple[list[dict], list[dict]]:
    """Разделить matches на (kept, vetoed). Нужны запросы: восстанавливаем
    argmax по эмбеддингам через comp (как semantic_judge); без comp —
    запрос неизвестен, вето не применяем (возвращаем всё как kept)."""
    if not config.HUB_GATE_ENABLED or comp is None:
        return matches, []
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        qembs = comp.embed_skills(disc_skills).astype(float)
    except Exception as exc:
        logger.warning("hub_gate_encode_failed", error=str(exc))
        return matches, []
    kept, vetoed = [], []
    for m in matches:
        try:
            me = comp.embed_skills([m["skill"]]).astype(float)
            bi = int(np.argmax(cosine_similarity(me, qembs)[0]))
            if veto_match(disc_skills[bi], m["skill"], float(m.get("similarity", 0))):
                vetoed.append(m)
                continue
        except Exception as exc:
            logger.warning("hub_gate_pair_failed", error=str(exc))
        kept.append(m)
    return kept, vetoed
