"""LLM-судья семантического хвоста (заготовка, default off).

Проверяет пары (запрос -> совпадение) из косинусной серой зоны [0.6, 0.75):
истинный перевод/синоним или эмбеддинг-хаб. Вердикты только записываются,
совпадения НЕ фильтруются (судья не валидирован — фильтровать рано).

Подключение: env LLM_JUDGE_URL / LLM_JUDGE_API_KEY / LLM_JUDGE_MODEL,
затем LLM_JUDGE_ENABLED=true. Без подключения — тихий skip без падения.
"""

from __future__ import annotations

import json

import structlog

from src import config

logger = structlog.get_logger(__name__)

GRAY_LO = 0.6
GRAY_HI = 0.75


def gray_pairs_for_judge(comp, disc_skills: list[str], matches: list[dict]) -> list[tuple[str, str, float]]:
    """Пары серой зоны с восстановленным запросом (argmax по эмбеддингам).

    `matches` — из compare_student_to_market (навык маркета + similarity,
    запрос потерян). Возвращает [(query, market, sim)] для sim в [0.6, 0.75),
    но не более config.LLM_JUDGE_MAX_PAIRS. Ошибки кодирования — пусто.
    """
    gray = [m for m in matches
            if GRAY_LO <= float(m.get("similarity", 0)) < GRAY_HI]
    if not gray or not disc_skills:
        return []
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        qembs = comp.embed_skills(disc_skills).astype(float)
        out = []
        for m in gray[: config.LLM_JUDGE_MAX_PAIRS]:
            me = comp.embed_skills([m["skill"]]).astype(float)
            sims = cosine_similarity(me, qembs)[0]
            out.append((disc_skills[int(np.argmax(sims))], m["skill"], float(m["similarity"])))
        return out
    except Exception as exc:
        logger.warning("judge_pairs_recovery_failed", error=str(exc))
        return []


def judge_pairs(
    pairs: list[tuple[str, str, float]],
    client=None,
) -> dict:
    """Вердикты LLM по парам. Возврат: {"verdicts": {(q, m): bool|None},
    "skipped": reason|None}. None-вердикт = воздержался (не veto)."""
    if not config.LLM_JUDGE_ENABLED:
        return {"verdicts": {}, "skipped": "disabled"}
    if not pairs:
        return {"verdicts": {}, "skipped": "no_pairs"}
    try:
        from src.services.llm_client import LLMClient

        api_key = None
        if config.LLM_JUDGE_API_KEY is not None:
            api_key = config.LLM_JUDGE_API_KEY.get_secret_value()
        client = client or LLMClient(
            base_url=config.LLM_JUDGE_URL or config.OLLAMA_URL,
            api_key=api_key,
            model=config.LLM_JUDGE_MODEL or config.OLLAMA_MODEL,
        )
        numbered = "\n".join(
            f'{i}. "{q}" -> "{m}" (cos={s})' for i, (q, m, s) in enumerate(pairs))
        prompt = (
            "Ты проверяешь совпадения навыков: запрос из РПД дисциплины "
            "и найденный навык рынка. Для каждой пары ответь true "
            "(истинный синоним, перевод или прямое подмножество) или false "
            "(ложное совпадение). Отвечай ТОЛЬКО JSON-объектом вида "
            '{"0": true, "1": false} без пояснений.\n' + numbered)
        text = client.generate(prompt, temperature=0.0, max_tokens=500)
        verdicts = _parse_verdicts(text, pairs)
        return {"verdicts": verdicts, "skipped": None}
    except Exception as exc:
        logger.warning("judge_unavailable", error=str(exc))
        return {"verdicts": {}, "skipped": f"unavailable: {exc}"}


def _parse_verdicts(text: str, pairs: list[tuple[str, str, float]]) -> dict[tuple[str, str], bool | None]:
    """Строгий парсинг JSON-вердиктов; нераспознанное — None (воздержался)."""
    verdicts: dict[tuple[str, str], bool | None] = {}
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    except (ValueError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    for i, (q, m, _s) in enumerate(pairs):
        v = data.get(str(i), data.get(i))
        verdicts[(q, m)] = bool(v) if isinstance(v, bool) else None
    return verdicts
