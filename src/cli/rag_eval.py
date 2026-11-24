"""CLI офлайн-оценки RAG: recall@k, MRR, nDCG@k по golden-set.

Usage:
    python -m src.cli rag-eval --template > golden.json
    python -m src.cli rag-eval --golden golden.json --dry-run
    python -m src.cli rag-eval --golden golden.json --k 5 --fail-under 0.6
    python -m src.cli rag-eval --golden golden.json --min-score 0.2
    python -m src.cli rag-eval --golden golden.json --sweep 0.2,0.3,0.35
    python -m src.cli rag-eval --golden golden.json --json reports/rag_eval.json

Golden-set - JSON вида ``{"items": [{"query": ..., "expect": [...]}]}``; ключ
ожидания - ``external_id`` документа, его заголовок или ``source:title``.
Поиск идёт через тот же ``RagService``, что и чат, но без записи в
``rag_query_log`` и без LLM: оценивается только retrieval.

``--sweep`` прогоняет набор на нескольких порогах ``min_score`` и печатает
сводку по каждому: так подбирается ``RAG_MIN_SCORE`` без правки .env и
перезапуска сервиса.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

import structlog

from src.rag import eval as rag_eval
from src.rag.service import RagService, clear_cache
from src.rag.settings import RagSettings, get_settings

logger = structlog.get_logger(__name__)


def _build_settings(min_score: float | None) -> RagSettings:
    settings = get_settings()
    if min_score is None:
        return settings
    return replace(settings, min_score=float(min_score))


async def _warmup() -> None:
    """Прогреть модель эмбеддингов до замера.

    Первый ``encode`` грузит sentence-transformers ~10-20 с и не укладывается в
    ``RAG_TIMEOUT_S``: без прогрева первый вопрос всегда деградирует до FTS и
    метрики зависят от порядка вопросов, а не от индекса.
    """
    try:
        from src.rag.embedder import get_embedder

        await get_embedder().encode_one("прогрев модели эмбеддингов")
    except Exception as exc:
        logger.warning("rag_eval_warmup_failed", error=str(exc))


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'recall':>6}  {'mrr':>5}  {'ndcg':>5}  {'hits':>4}  query"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['recall_at_k']:>6.2f}  {row['mrr']:>5.2f}  "
            f"{row['ndcg_at_k']:>5.2f}  {row['hits']:>4}  {row['query']}"
        )
        if row["missing"]:
            print(f"        не найдено: {', '.join(row['missing'][:5])}")


def _print_summary(summary: dict[str, Any], k: int, min_score: float) -> None:
    print()
    print(
        f"queries={summary['queries']} skipped={summary['skipped']} k={k} "
        f"min_score={min_score:g} "
        f"recall@{k}={summary['recall_at_k']:.3f} "
        f"mrr={summary['mrr']:.3f} "
        f"ndcg@{k}={summary['ndcg_at_k']:.3f} "
        f"empty={summary['empty_result_queries']}"
    )


async def _evaluate_all(
    service: RagService,
    items: list[rag_eval.GoldenItem],
    *,
    k: int,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        try:
            result = await service.search(
                item.query,
                role=role,
                profile=item.profile,
                sources=item.sources,
                direction=item.direction,
                year=item.year,
                top_n=k,
                log=False,
            )
        except Exception as exc:  # pragma: no cover - search не бросает
            logger.warning("rag_eval_query_failed", query=item.query, error=str(exc))
            rows.append(
                {
                    "query": item.query,
                    "profile": item.profile or "",
                    "expected": len(item.keys()),
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "ndcg_at_k": 0.0,
                    "hits": 0,
                    "missing": sorted(item.keys()),
                    "top": [],
                }
            )
            continue
        rows.append(rag_eval.evaluate_item(result.context.evidence, item, k=k))
    return rows


async def run(args: Any) -> int:
    items = rag_eval.load_golden(args.golden)
    k = int(args.k)

    if args.dry_run:
        print(f"golden-set OK: {len(items)} вопросов, k={k}")
        for item in items:
            expect = ", ".join(item.expect) if item.expect else "<нет expect: пропуск>"
            print(f"  - {item.query}  ->  {expect}")
        return 0

    if not items:
        print("golden-set пуст: нечего оценивать", file=sys.stderr)
        return 2

    if not _build_settings(None).enabled:
        print(
            "RAG_ENABLED=false: поиск всё равно выполнится по текущему индексу "
            "(флаг влияет только на чат и пайплайн)",
            file=sys.stderr,
        )

    if getattr(args, "warmup", True):
        await _warmup()

    if args.sweep:
        thresholds = _parse_sweep(args.sweep)
        sweeps: list[dict[str, Any]] = []
        print(f"{'min_score':>9}  {'recall':>6}  {'mrr':>5}  {'ndcg':>5}  {'empty':>5}")
        print("-" * 44)
        for threshold in thresholds:
            # Ключ L1-кэша не содержит min_score: между порогами кэш обязан быть пуст.
            clear_cache()
            service = RagService(settings=_build_settings(threshold))
            rows = await _evaluate_all(service, items, k=k, role=args.role)
            summary = rag_eval.aggregate(rows)
            sweeps.append({"min_score": threshold, "summary": summary, "items": rows})
            print(
                f"{threshold:>9.2f}  {summary['recall_at_k']:>6.2f}  "
                f"{summary['mrr']:>5.2f}  {summary['ndcg_at_k']:>5.2f}  "
                f"{summary['empty_result_queries']:>5}"
            )
        if args.json:
            _write_report(
                Path(args.json),
                {"k": k, "role": args.role, "sweeps": sweeps},
            )
        return 0

    service = RagService(settings=_build_settings(args.min_score))
    # Оценка не должна читать залипший L1-кэш прошлых прогонов.
    clear_cache()
    rows = await _evaluate_all(service, items, k=k, role=args.role)
    summary = rag_eval.aggregate(rows)
    _print_table(rows)
    _print_summary(summary, k, service.settings.min_score)

    if args.json:
        _write_report(
            Path(args.json),
            {
                "k": k,
                "role": args.role,
                "min_score": service.settings.min_score,
                "summary": summary,
                "items": rows,
            },
        )

    if args.fail_under is not None and summary["recall_at_k"] < args.fail_under:
        print(
            f"\nFAIL: recall@{k}={summary['recall_at_k']:.3f} < {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {path}")


def _parse_sweep(raw: str) -> list[float]:
    try:
        values = sorted({float(part) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit(f"--sweep: нужен список чисел через запятую ({raw!r})") from exc
    if not values:
        raise SystemExit("--sweep: пустой список порогов")
    return values


def main(args: Any) -> None:
    if getattr(args, "template", False):
        print(json.dumps(rag_eval.golden_template(), ensure_ascii=False, indent=2))
        return
    if not getattr(args, "golden", None):
        raise SystemExit("нужен --golden PATH (или --template, чтобы получить схему)")
    raise SystemExit(asyncio.run(run(args)))


def build_parser(sub) -> None:
    parser = sub.add_parser(
        "rag-eval", help="Оценить retrieval RAG по golden-set (recall@k, MRR, nDCG@k)"
    )
    parser.add_argument("--golden", default=None, help="Путь к golden-set JSON")
    parser.add_argument("--k", type=int, default=5, help="Глубина оценки (по умолчанию 5)")
    parser.add_argument("--role", default="admin", help="Роль для ACL при поиске")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Порог уверенности для evidence (по умолчанию из RAG_MIN_SCORE)",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="Прогнать набор на порогах min_score через запятую (0.2,0.3,0.35)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Код возврата 1, если средний recall@k ниже порога",
    )
    parser.add_argument("--json", default=None, help="Сохранить полный отчёт в файл")
    parser.add_argument("--dry-run", action="store_true", help="Только проверить golden-set")
    parser.add_argument("--template", action="store_true", help="Напечатать схему golden-set")
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Не прогревать модель эмбеддингов (метрики станут зависимы от порядка)",
    )
    parser.set_defaults(func=main)
