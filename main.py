#!/usr/bin/env python
"""
main.py — CLI entry point. Thin wrapper around src.pipeline.runner.
"""

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from src import Err, Ok, Result, config
from src.logging_config import setup_structlog
from src.pipeline.runner import (
    rebuild,
    run_full_pipeline,
    run_status,
    run_train_model,
)
from src.pipeline.teacher_analysis_runner import run_teacher_analysis


def parse_arguments():
    parser = argparse.ArgumentParser(description="Полный пайплайн: сбор вакансий + gap-анализ + рекомендации")
    parser.add_argument("--query", "-q", type=str, default="1с разработчик")
    parser.add_argument("--area-id", "-a", type=int, default=2)
    parser.add_argument("--max-pages", "-p", type=int, default=1)
    parser.add_argument("--period", "-d", type=int, default=30)
    parser.add_argument("--show-vacancies", "-v", action="store_true")
    parser.add_argument("--skip-details", "-s", action="store_true")
    parser.add_argument("--excel", "-e", action="store_true")
    parser.add_argument("--no-filter", "-nf", action="store_true")
    parser.add_argument("--queries-file", "-qf", type=str)
    parser.add_argument("--regions", "-r", type=str)
    parser.add_argument("--industry", "-i", type=int)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-vacancies-per-query", type=int, default=2000)
    parser.add_argument("--it-sector", action="store_true")
    parser.add_argument("--use-async", action="store_true", default=True)
    parser.add_argument("--async-workers", type=int, default=3)
    parser.add_argument("--async-threshold", type=int, default=10000)
    parser.add_argument("--run-gap-analysis", action="store_true", default=False,
        help="Устарел. Используйте --skip-gap-analysis для пропуска")
    parser.add_argument("--run-notebooks", action="store_true")
    parser.add_argument("--skip-gap-analysis", action="store_true", default=False)
    parser.add_argument("--status", action="store_true", help="Показать состояние файлов и моделей")
    parser.add_argument("--train-model", action="store_true", help="Обучить LTR-модель на текущих данных и выйти")
    parser.add_argument("--force", action="store_true", help="Принудительное переобучение (пропускает проверки кэша)")
    parser.add_argument("--use-llm", action="store_true", default=False,
        help="Использовать LLM (YandexGPT) для живых объяснений рекомендаций")
    parser.add_argument("--skip-collection", action="store_true",
        help="Пропустить сбор вакансий, использовать существующие файлы")
    parser.add_argument("--teacher-analysis", "-ta", nargs="?", const="all", default=None,
        help="Запустить teacher analysis. Укажите код направления (09.03.02) или all")
    parser.add_argument("--discipline", type=str, default=None,
        help="Фильтр по дисциплине (часть названия). Работает с --teacher-analysis")
    args = parser.parse_args()
    if args.query and args.query.startswith("b64:"):
        import base64
        args.query = base64.b64decode(args.query[4:]).decode("utf-8")
    return args


def validate_args(args) -> None:
    errors = []
    if args.train_model:
        detailed_ok = (config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json").exists()
        basic_ok = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_ok and not basic_ok:
            errors.append(
                "Для --train-model нужен файл вакансий "
                "(hh_vacancies_detailed.json или hh_vacancies_basic.json). Сначала выполните сбор данных."
            )
    if args.skip_collection and not args.train_model:
        detailed_exists = (config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json").exists()
        basic_exists = (config.DATA_RAW_DIR / "hh_vacancies_basic.json").exists()
        if not detailed_exists and not basic_exists:
            errors.append(
                "--skip-collection указан, но нет файлов вакансий. Сначала выполните сбор или уберите этот флаг."
            )
    if args.use_llm and (not config.YC_API_KEY or not config.YC_FOLDER_ID):
        errors.append("Для --use-llm необходимо задать YC_API_KEY и YC_FOLDER_ID в .env или переменных окружения.")
    if errors:
        for msg in errors:
            print(f"❌ Ошибка: {msg}")
        sys.exit(1)


def main() -> Result[None, str]:
    config.settings.ensure_dirs()
    setup_structlog()
    args = parse_arguments()
    validate_args(args)

    if args.status:
        match run_status(args):
            case Ok(_):
                return Ok(None)
            case Err(e):
                return Err(e)

    if args.teacher_analysis:
        print(f"📊 Запуск teacher analysis (направление: {args.teacher_analysis})...")
        import asyncio
        result = asyncio.run(run_teacher_analysis(
            direction_code=None if args.teacher_analysis == "all" else args.teacher_analysis,
            discipline_filter=args.discipline,
        ))
        match result:
            case Ok(summary):
                print(f"✅ Teacher analysis завершён. Покрытие: {summary.get('average_coverage', 0)*100:.1f}%")
                print(f"   Дисциплин: {summary.get('total_disciplines', 0)}")
                print(f"   Всего пробелов: {summary.get('total_gaps_across_all', 0)}")
                print(f"   Результаты: {summary.get('generated_at', '')}")
                return Ok(None)
            case Err(e):
                return Err(e)

    if args.train_model:
        match run_train_model(args):
            case Ok(_):
                return Ok(None)
            case Err(e):
                return Err(e)

    match run_full_pipeline(args):
        case Ok(_):
            return Ok(None)
        case Err(e):
            return Err(e)


if __name__ == "__main__":
    match main():
        case Ok(_):
            sys.exit(0)
        case Err(e):
            print(f"❌ {e}")
            sys.exit(1)
