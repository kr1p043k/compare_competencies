"""Веса навыков из научных публикаций: артефакт для смешивания источников по ролям.

Рынок даёт частоты навыков в вакансиях, наука - в публикациях (OpenAlex/arXiv).
Этот CLI собирает вторую половину: по каждой профессии из
``profession_taxonomy.json`` берёт документы по её hh-запросам, извлекает навыки
тем же ``SkillParser``, что и для вакансий (иначе шкалы несопоставимы), и пишет
``data/processed/article_weights.json``:

    {"_meta": {...}, "roles": {"<Профессия>": {"python": 1.0, ...}}, "global": {...}}

Веса нормированы на максимум внутри роли (0..1]. Потребитель -
``src/analyzers/skills/science_weights.py``.

Сеть нужна только в ``fetch_*``: разбор ответов и сборка payload - чистые
функции, поэтому их можно проверять офлайн.

Запуск:
    python -m src.cli build-article-weights --per-query 25 --source openalex
    python -m src.cli build-article-weights --professions "Data Scientist" --dry-run
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import httpx
import structlog

from src import config
from src.parsing.skills.skill_parser import SkillParser, SkillSource
from src.utils import atomic_write_json

logger = structlog.get_logger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
METHODOLOGY_VERSION = "v1"
DEFAULT_SINCE = "2023-01-01"
MAX_QUERIES_PER_PROFESSION = 2
REQUEST_PAUSE_S = 1.0
# OpenAlex отвечает 429 на пачку запросов: ждём и повторяем, иначе роль
# получает ноль документов и выпадает из артефакта.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRY_BACKOFF_S = (5.0, 15.0, 30.0)
# Абстракты - это проза, а не список технологий: английские модальные глаголы
# и служебные слова из it_skills ("can" = CAN bus) в тексте статьи срабатывают
# как ложные навыки. Слова из этого набора в артефакт не попадают никогда.
ARTICLE_SKILL_STOPWORDS = frozenset({
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "does", "did", "has", "have", "had", "was", "were", "been", "being", "are",
})
# Абстракт OpenAlex обычно до 2000 символов; дефолтный срез парсера (1000)
# терял половину текста вместе с навыками из выводов.
ARTICLE_TEXT_LIMIT = 4000


# ── разбор ответов ────────────────────────────────────────────────────────────


def _reconstruct_abstract(inverted: object) -> str:
    """OpenAlex отдаёт абстракт как инвертированный индекс: слово -> позиции."""
    if not isinstance(inverted, dict):
        return ""
    placed: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                placed.append((pos, word))
    placed.sort(key=lambda p: p[0])
    return " ".join(word for _, word in placed)


def _openalex_keywords(work: dict) -> list[str]:
    out: list[str] = []
    for key in work.get("keywords") or []:
        if isinstance(key, dict):
            name = key.get("display_name") or key.get("keyword")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
        elif isinstance(key, str) and key.strip():
            out.append(key.strip())
    for concept in work.get("concepts") or []:
        if isinstance(concept, dict):
            name = concept.get("display_name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return out


def parse_openalex_works(payload: dict) -> list[dict]:
    """Нормализовать ответ OpenAlex. Пустые/битые поля не роняют разбор."""
    docs: list[dict] = []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return docs
    for work in results:
        if not isinstance(work, dict):
            continue
        year = work.get("publication_year")
        docs.append(
            {
                "title": str(work.get("title") or work.get("display_name") or ""),
                "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
                "keywords": _openalex_keywords(work),
                "year": year if isinstance(year, int) else None,
                "source": "openalex",
            }
        )
    return docs


def _arxiv_year(published: str | None) -> int | None:
    if not published:
        return None
    try:
        return int(published[:4])
    except (TypeError, ValueError):
        return None


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    """Нормализовать Atom-ленту arXiv. Битый XML -> пустой список."""
    docs: list[dict] = []
    if not xml_text:
        return docs
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("arxiv_feed_parse_failed", error=str(exc))
        return docs
    for entry in root.findall(".//atom:entry", ARXIV_NS):
        title = (entry.findtext("./atom:title", default="", namespaces=ARXIV_NS) or "").strip()
        summary = (entry.findtext("./atom:summary", default="", namespaces=ARXIV_NS) or "").strip()
        published = entry.findtext("./atom:published", default="", namespaces=ARXIV_NS)
        keywords = [
            cat.get("term", "").strip()
            for cat in entry.findall("./atom:category", ARXIV_NS)
            if cat.get("term")
        ]
        docs.append(
            {
                "title": title,
                "abstract": summary,
                "keywords": keywords,
                "year": _arxiv_year(published),
                "source": "arxiv",
            }
        )
    return docs


# ── извлечение навыков и нормировка ──────────────────────────────────────────


def extract_document_skills(
    docs: list[dict], parser: SkillParser, vocab: set[str] | None = None
) -> dict[str, int]:
    """Сколько документов упоминают навык. Навык считается один раз на документ."""
    counts: dict[str, int] = {}
    allowed = {v.strip().casefold() for v in vocab} if vocab else None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        parts = [
            str(doc.get("title") or ""),
            str(doc.get("abstract") or ""),
            " ".join(str(k) for k in (doc.get("keywords") or [])),
        ]
        text = " ".join(p for p in parts if p).strip()
        if not text:
            continue
        result = parser._extract_from_text(text, SkillSource.DESCRIPTION, max_text_length=ARTICLE_TEXT_LIMIT)
        if not result.is_ok():
            logger.warning("article_skill_extract_failed", error=str(result.err()))
            continue
        seen: set[str] = set()
        for item in result.unwrap():
            name = (getattr(item, "text", "") or "").strip().casefold()
            if not name or name in seen or name in ARTICLE_SKILL_STOPWORDS:
                continue
            if allowed is not None and name not in allowed:
                continue
            seen.add(name)
            counts[name] = counts.get(name, 0) + 1
    return counts


def normalize_weights(counts: dict[str, int], min_freq: int = 2) -> dict[str, float]:
    """Срезать редкое, поделить на максимум, отсортировать по убыванию."""
    kept = {s: int(c) for s, c in (counts or {}).items() if isinstance(c, (int, float)) and c >= min_freq}
    if not kept:
        return {}
    peak = max(kept.values())
    if peak <= 0:
        return {}
    ordered = sorted(kept.items(), key=lambda kv: (-kv[1], kv[0]))
    return {skill: round(min(1.0, value / peak), 4) for skill, value in ordered}


def build_payload(roles_counts: dict[str, dict[str, int]], meta: dict, min_freq: int = 2) -> dict:
    """Собрать артефакт: роли нормированы по отдельности, global - по сумме."""
    roles: dict[str, dict[str, float]] = {}
    totals: dict[str, int] = {}
    for role, counts in (roles_counts or {}).items():
        normalized = normalize_weights(counts, min_freq=min_freq)
        if not normalized:
            continue
        roles[str(role)] = normalized
        for skill, count in counts.items():
            key = str(skill).strip().casefold()
            if key:
                totals[key] = totals.get(key, 0) + int(count)
    payload_meta = dict(meta or {})
    payload_meta["methodology_version"] = payload_meta.get("methodology_version", METHODOLOGY_VERSION)
    payload_meta.setdefault("generated_at", datetime.now(UTC).isoformat())
    payload_meta["normalization"] = "per_role_max"
    payload_meta["professions"] = len(roles)
    return {
        "_meta": payload_meta,
        "roles": roles,
        "global": normalize_weights(totals, min_freq=min_freq),
    }


# ── сеть ─────────────────────────────────────────────────────────────────────


def _openalex_mailto() -> str | None:
    """Контакт для polite pool OpenAlex (даёт более высокий лимит запросов).

    Берётся из ``OPENALEX_MAILTO``: адрес - персональные данные, в коде его нет.
    Без переменной запросы идут в общий пул, и на длинном прогоне прилетает 429.
    """
    return (os.environ.get("OPENALEX_MAILTO") or "").strip() or None


def _get_json(url: str, params: dict, timeout: float) -> dict | None:
    """GET с экспоненциальным бэкоффом на 429/5xx.

    OpenAlex режет пачку запросов (429), и без ожидания роль молча получает
    ноль документов. Паузы 5/15/30 с - как в рекомендациях OpenAlex.
    """
    last_error: object = "no attempt"
    for wait in (0.0, *RETRY_BACKOFF_S):
        if wait:
            time.sleep(wait)
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            if response.status_code in RETRY_STATUSES:
                last_error = f"status {response.status_code}"
                logger.warning("http_retry", url=url, status=response.status_code, wait=wait)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            logger.warning("http_retry", url=url, error=str(exc), wait=wait)
    logger.warning("http_get_failed", url=url, error=str(last_error))
    return None


def _get_text(url: str, params: dict, timeout: float) -> str | None:
    last_error: object = "no attempt"
    for wait in (0.0, *RETRY_BACKOFF_S):
        if wait:
            time.sleep(wait)
        try:
            response = httpx.get(url, params=params, timeout=timeout)
            if response.status_code in RETRY_STATUSES:
                last_error = f"status {response.status_code}"
                logger.warning("http_retry", url=url, status=response.status_code, wait=wait)
                continue
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            logger.warning("http_retry", url=url, error=str(exc), wait=wait)
    logger.warning("http_get_failed", url=url, error=str(last_error))
    return None


def fetch_openalex(query: str, per_page: int, since: str, timeout: float = 30.0) -> list[dict]:
    """Свежие, но релевантные работы: сортировка по релевантности, окно - фильтр.

    Сортировка по дате отдаёт случайный поток свежих статей (ботаника вместо ML),
    поэтому дата задаётся фильтром, а порядок - релевантностью запроса.
    """
    params = {
        "search": query,
        "per_page": max(1, int(per_page)),
        "sort": "relevance_score:desc",
        "filter": f"from_publication_date:{since}",
    }
    mailto = _openalex_mailto()
    if mailto:
        params["mailto"] = mailto
    payload = _get_json(OPENALEX_URL, params, timeout)
    if payload is None:
        logger.warning("openalex_fetch_failed", query=query)
        return []
    return parse_openalex_works(payload)


def fetch_arxiv(query: str, max_results: int, timeout: float = 30.0) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max(1, int(max_results)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    text = _get_text(ARXIV_URL, params, timeout)
    if text is None:
        logger.warning("arxiv_fetch_failed", query=query)
        return []
    return parse_arxiv_feed(text)


# ── оркестрация ──────────────────────────────────────────────────────────────


def _load_it_skills() -> set[str]:
    try:
        raw = json.loads(Path(config.IT_SKILLS_PATH).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("article_vocab_load_failed", error=str(exc))
        return set()
    return {str(s).strip() for s in raw if isinstance(s, str) and s.strip()}


def _load_vocab() -> set[str]:
    """Словарь навыков для фильтра извлечённого.

    Приоритет - словарь рыночных весов: смесь всё равно применяется только к
    навыкам, которые есть в рыночной модели, а научные частоты остальных -
    мёртвый груз в артефакте. Нет весов (свежая установка) - полный it_skills.
    """
    weights_path = Path(config.DATA_PROCESSED_DIR) / "skill_weights.json"
    try:
        raw = json.loads(weights_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw:
            logger.info("article_vocab_from_market_weights", count=len(raw))
            return {str(k).strip() for k in raw if str(k).strip()}
    except FileNotFoundError:
        logger.info("article_vocab_market_weights_missing", path=str(weights_path))
    except Exception as exc:
        logger.warning("article_vocab_market_weights_failed", error=str(exc))
    return _load_it_skills()


def _collect_docs(
    queries: list[str], per_query: int, source: str, since: str, pause: float | None = None
) -> list[dict]:
    delay = REQUEST_PAUSE_S if pause is None else max(0.0, float(pause))
    docs: list[dict] = []
    for query in queries:
        if source in ("openalex", "both"):
            docs.extend(fetch_openalex(query, per_query, since))
            time.sleep(delay)
        if source in ("arxiv", "both"):
            docs.extend(fetch_arxiv(query, per_query))
            time.sleep(delay)
    return docs


def _previous_role_weights(target: Path) -> dict[str, dict[str, float]]:
    """Сырые веса прошлого артефакта (для переноса ролей без новых данных)."""
    try:
        raw = json.loads(Path(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    roles = raw.get("roles")
    if not isinstance(roles, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for role, weights in roles.items():
        if isinstance(weights, dict) and weights:
            out[str(role)] = {str(k): float(v) for k, v in weights.items()}
    return out


def _weights_to_counts(weights: dict[str, float], min_freq: int) -> dict[str, int]:
    """Вернуть нормированные веса в псевдо-частоты 0..100.

    Нормировка на максимум инвариантна к масштабу, поэтому перенос даёт те же
    доли внутри роли, но проходит через общий порог ``min_freq`` и участвует в
    сборке ``global`` как обычные частоты.
    """
    peak = max((float(v) for v in weights.values()), default=0.0)
    if peak <= 0:
        return {}
    return {str(k): max(int(min_freq), int(round(float(v) / peak * 100))) for k, v in weights.items()}


def main(
    professions: list[str] | None = None,
    per_query: int = 25,
    source: str = "openalex",
    since: str = DEFAULT_SINCE,
    out: str | None = None,
    min_freq: int = 2,
    dry_run: bool = False,
    pause: float | None = None,
) -> None:
    taxonomy_path = Path(config.REFERENCE_DIR) / "profession_taxonomy.json"
    try:
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Не удалось прочитать {taxonomy_path}: {exc}")
        return
    all_professions = taxonomy.get("professions") or {}
    if not isinstance(all_professions, dict) or not all_professions:
        print("profession_taxonomy.json без профессий")
        return

    selected = [p for p in (professions or list(all_professions)) if p in all_professions]
    missing = [p for p in (professions or []) if p not in all_professions]
    if missing:
        print(f"Пропущены неизвестные профессии: {', '.join(missing)}")

    parser = SkillParser()
    vocab = _load_vocab()
    roles_counts: dict[str, dict[str, int]] = {}
    documents = 0
    target = Path(out) if out else Path(config.ARTICLE_WEIGHTS_PATH)
    # Прошлый артефакт: роль, у которой сбор не дал ни одного навыка (сеть
    # отвалилась или запрос пуст), не должна исчезнуть из файла - иначе один
    # плохой прогон обнуляет научную половину для целой роли.
    previous_roles = _previous_role_weights(target)
    carried: list[str] = []

    for name in selected:
        data = all_professions.get(name) or {}
        queries = [str(q) for q in (data.get("hh_queries") or [name])][:MAX_QUERIES_PER_PROFESSION]
        try:
            docs = _collect_docs(queries, per_query, source, since, pause=pause)
            counts = extract_document_skills(docs, parser, vocab or None)
        except Exception as exc:
            logger.warning("article_role_failed", profession=name, error=str(exc))
            print(f"  {name:35s} ошибка: {exc}")
            counts, docs = {}, []
        documents += len(docs)
        if not counts and name in previous_roles:
            roles_counts[name] = _weights_to_counts(previous_roles[name], min_freq)
            carried.append(name)
            print(f"  {name:35s} документов: {len(docs):4d}  навыков: перенос из прошлого артефакта")
            continue
        roles_counts[name] = counts
        print(f"  {name:35s} документов: {len(docs):4d}  навыков: {len(counts):4d}")

    # Частичный прогон (--professions X) не должен затирать остальные роли:
    # берём их из прошлого артефакта, иначе файл молча теряет 40 ролей из 43.
    for role, weights in previous_roles.items():
        if role not in roles_counts:
            roles_counts[role] = _weights_to_counts(weights, min_freq)
            carried.append(role)

    meta = {
        "sources": [source],
        "per_query": per_query,
        "since": since,
        # Документы именно этого прогона: carried_over-роли сюда не входят,
        # поэтому цифра не выдаёт себя за полный объём корпуса.
        "documents_last_run": documents,
        "roles_fresh_last_run": len(roles_counts) - len(carried),
        "skill_vocab": len(vocab),
    }
    if carried:
        meta["carried_over"] = sorted(carried)
    payload = build_payload(roles_counts, meta, min_freq=min_freq)

    if dry_run:
        print(
            json.dumps(
                {
                    "professions": payload["_meta"]["professions"],
                    "documents": documents,
                    "global_skills": len(payload["global"]),
                    "carried_over": sorted(carried),
                },
                ensure_ascii=False,
            )
        )
        return

    atomic_write_json(payload, target)
    print(f"Записано: {target} (ролей: {len(payload['roles'])}, навыков: {len(payload['global'])})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Веса навыков из научных публикаций")
    ap.add_argument("--professions", nargs="*", default=None)
    ap.add_argument("--per-query", type=int, default=25)
    ap.add_argument("--source", default="openalex", choices=["openalex", "arxiv", "both"])
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--pause", type=float, default=None,
                    help=f"Пауза между запросами, с (по умолчанию {REQUEST_PAUSE_S})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    main(
        professions=args.professions,
        per_query=args.per_query,
        source=args.source,
        since=args.since,
        out=args.out,
        min_freq=args.min_freq,
        dry_run=args.dry_run,
        pause=args.pause,
    )
