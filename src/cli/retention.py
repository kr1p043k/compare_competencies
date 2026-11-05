"""Retention: чистка устаревших данных БД.

Зачем: вакансии, request_logs и pipeline_runs копились бесконечно (57k
вакансий, сотни тысяч логов запросов). Без чистки таблицы растут, а запросы
со временем падают по времени и по диску.

По умолчанию - dry-run: показывает, что БУДЕТ удалено. Удаление только с
--apply и всегда в одной транзакции, чтобы не оставить полуудалённое состояние.

Запуск:
    python -m src.cli retention                 # dry-run, показать план
    python -m src.cli retention --apply         # удалить
    python -m src.cli retention --vacancies-days 730 --logs-days 90 --apply
"""

from __future__ import annotations

import argparse

import structlog

from src import config

logger = structlog.get_logger(__name__)

#: По умолчанию: вакансии живут 2 года (нужны для прогнозов и трендов),
#: логи запросов - 90 дней, прогоны пайплайна - год.
DEFAULT_VACANCY_DAYS = 730
DEFAULT_LOG_DAYS = 90
DEFAULT_RUN_DAYS = 365

#: Границы допустимых значений дня для всех day-based аргументов.
#: Интерполяция в `INTERVAL '<n> days'` без границ позволяла передать
#: `--vacancies-days 0` или `-1`, и условие `published_at < NOW() - INTERVAL
#: '0 days'` (или отрицательный интервал) собирало буквально все строки.
MIN_DAYS = 1
MAX_DAYS = 36500


def positive_days(value: str) -> int:
    """Argparse type: принимает только целое число дней от 1 до 36500.

    Поднимает argparse.ArgumentTypeError с русским сообщением, чтобы
    argparse показал ошибку и завершился с ненулевым кодом.
    """
    try:
        days = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"'{value}' - не число. Укажите целое число дней "
            f"от {MIN_DAYS} до {MAX_DAYS}, например: --vacancies-days 730"
        ) from err
    if days < MIN_DAYS or days > MAX_DAYS:
        raise argparse.ArgumentTypeError(
            f"{days} - вне допустимого диапазона: число дней должно быть "
            f"от {MIN_DAYS} до {MAX_DAYS} (получено {value})"
        )
    return days


def _to_sync_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


def main(args: argparse.Namespace) -> int:
    import psycopg2

    dsn = _to_sync_dsn(config.settings.DATABASE_URL)
    apply = bool(getattr(args, "apply", False))
    vac_days = int(args.vacancies_days)
    log_days = int(args.logs_days)
    run_days = int(args.run_days)

    plan = [
        (
            "vacancies",
            # позиции, у которых истёк срок давности. published_at бывает
            # NULL - для таких строк используем created_at, иначе они никогда
            # не попадали бы под чистку и копились вечно.
            "SELECT COUNT(*) FROM vacancies "
            f"WHERE COALESCE(published_at, created_at) < NOW() - INTERVAL '{vac_days} days'",
            "DELETE FROM vacancies "
            f"WHERE COALESCE(published_at, created_at) < NOW() - INTERVAL '{vac_days} days'",
        ),
        (
            "request_logs",
            "SELECT COUNT(*) FROM request_logs "
            f"WHERE created_at < NOW() - INTERVAL '{log_days} days'",
            "DELETE FROM request_logs "
            f"WHERE created_at < NOW() - INTERVAL '{log_days} days'",
        ),
        (
            "pipeline_runs",
            "SELECT COUNT(*) FROM pipeline_runs "
            f"WHERE started_at < NOW() - INTERVAL '{run_days} days'",
            "DELETE FROM pipeline_runs "
            f"WHERE started_at < NOW() - INTERVAL '{run_days} days'",
        ),
    ]

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        total = 0
        for table, count_sql, delete_sql in plan:
            cur.execute(count_sql)
            n = int(cur.fetchone()[0])
            total += n
            print(f"{table:16} к удалению: {n}")
            if apply and n:
                cur.execute(delete_sql)
                print(f"{table:16} удалено: {cur.rowcount}")
        if apply:
            conn.commit()
            logger.info("retention_applied", rows=total)
            print(f"Готово, удалено строк: {total}")
        else:
            conn.rollback()
            print(f"dry-run: всего кандидатов {total}. Повторите с --apply.")
        return 0
    finally:
        conn.close()


def build_parser(sub) -> None:
    p = sub.add_parser("retention", help="Удалить устаревшие данные (по умолчанию dry-run)")
    p.add_argument("--vacancies-days", type=positive_days, default=DEFAULT_VACANCY_DAYS, metavar="N")
    p.add_argument("--logs-days", type=positive_days, default=DEFAULT_LOG_DAYS, metavar="N")
    p.add_argument("--run-days", type=positive_days, default=DEFAULT_RUN_DAYS, metavar="N")
    p.add_argument("--apply", action="store_true", help="Реально удалить (без флага - dry-run)")
    p.set_defaults(func=lambda a: main(a))
