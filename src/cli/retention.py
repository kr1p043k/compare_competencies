"""Retention: чистка устаревших данных БД.

Зачем: вакансии, request_logs, pipeline_runs и sessions копились бесконечно
(57k вакансий, сотни тысяч логов запросов). Без чистки таблицы растут, а
запросы со временем падают по времени и по диску.

Правила чистки:
  * sessions - только закрытые сессии (logged_out_at IS NOT NULL) старше
    --session-days; активные сессии (logged_out_at IS NULL) не трогаются.
  * pipeline_runs - только завершённые прогоны (status in completed/failed/
    cancelled) старше --run-days; незавершённые (status='started') не
    удаляются ни при каких сроках.

По умолчанию - dry-run: показывает, что БУДЕТ удалено. Удаление только с
--apply и всегда в одной транзакции, чтобы не оставить полуудалённое состояние.

Запуск:
    python -m src.cli retention                 # dry-run, показать план
    python -m src.cli retention --apply         # удалить
    python -m src.cli retention --vacancies-days 730 --logs-days 90 --apply
    python -m src.cli retention --session-days 30 --run-days 365 --apply
"""

from __future__ import annotations

import argparse

import structlog

from src import config

logger = structlog.get_logger(__name__)

#: По умолчанию: вакансии живут 2 года (нужны для прогнозов и трендов),
#: логи запросов - 90 дней, закрытые сессии - месяц, прогоны пайплайна - год.
DEFAULT_VACANCY_DAYS = 730
DEFAULT_LOG_DAYS = 90
DEFAULT_SESSION_DAYS = 30
DEFAULT_RUN_DAYS = 365
#: Аудит действий живёт дольше логов: он нужен для разбора инцидентов и
#: проверок, но и он не должен расти вечно (в записях есть email и IP).
DEFAULT_AUDIT_DAYS = 730

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
    session_days = int(getattr(args, "session_days", DEFAULT_SESSION_DAYS))
    # Отрицательные значения: --logs-days=-1 удаляло бы всё, включая свежее.
    audit_days = int(getattr(args, "audit_days", DEFAULT_AUDIT_DAYS))
    # Защита от опечатки: отрицательные значения удалили бы почти всю таблицу.
    for name, value in (("vacancies-days", vac_days), ("logs-days", log_days),
                        ("run-days", run_days), ("session-days", session_days),
                        ("audit-days", audit_days)):
        if value < 1:
            print(f"Ошибка: --{name} должно быть >= 1 (получено {value})")
            return 2

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
            "sessions",
            # Только закрытые сессии (logged_out_at задан) старше срока. Активные
            # сессии (logged_out_at IS NULL) не трогаем - иначе разлогиним живых
            # пользователей, чьи токены ещё валидны.
            "SELECT COUNT(*) FROM sessions "
            f"WHERE logged_out_at IS NOT NULL "
            f"AND logged_out_at < NOW() - INTERVAL '{session_days} days'",
            "DELETE FROM sessions "
            f"WHERE logged_out_at IS NOT NULL "
            f"AND logged_out_at < NOW() - INTERVAL '{session_days} days'",
        ),
        (
            "pipeline_runs",
            # Только завершённые прогоны: status='started' - активный, его нельзя
            # удалять, даже если он висит дольше срока (иначе сломаем идущий прогон).
            "SELECT COUNT(*) FROM pipeline_runs "
            f"WHERE status IN ('completed', 'failed', 'cancelled') "
            f"AND started_at < NOW() - INTERVAL '{run_days} days'",
            "DELETE FROM pipeline_runs "
            f"WHERE status IN ('completed', 'failed', 'cancelled') "
            f"AND started_at < NOW() - INTERVAL '{run_days} days'",
        ),
        (
            # Таблицы может не быть до миграции add_audit_logs: тогда COUNT упадёт,
            # поэтому шаг пропускается при отсутствии relation (см. ниже).
            "audit_logs",
            "SELECT COUNT(*) FROM audit_logs "
            f"WHERE created_at < NOW() - INTERVAL '{audit_days} days'",
            "DELETE FROM audit_logs "
            f"WHERE created_at < NOW() - INTERVAL '{audit_days} days'",
        ),
    ]

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        total = 0
        for table, count_sql, delete_sql in plan:
            try:
                cur.execute(count_sql)
            except psycopg2.errors.UndefinedTable:
                # Миграция ещё не применена (например, audit_logs на старой базе) -
                # шаг пропускаем, а не роняем всю чистку.
                conn.rollback()
                print(f"{table:16} пропущено: таблицы нет")
                continue
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
    p.add_argument("--session-days", type=positive_days, default=DEFAULT_SESSION_DAYS, metavar="N",
                   help="Удалять закрытые сессии (logged_out_at) старше N дней")
    p.add_argument("--run-days", type=positive_days, default=DEFAULT_RUN_DAYS, metavar="N",
                   help="Удалять завершённые прогоны (completed/failed/cancelled) старше N дней")
    p.add_argument("--audit-days", type=positive_days, default=DEFAULT_AUDIT_DAYS, metavar="N")
    p.add_argument("--apply", action="store_true", help="Реально удалить (без флага - dry-run)")
    p.set_defaults(func=lambda a: main(a))
