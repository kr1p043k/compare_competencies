"""
Вспомогательные функции для загрузки детальных вакансий.
Используются DataSource и main.py.
"""

import json
import time

from tqdm import tqdm

import structlog

from src import Err, Ok, Result, config
from src.errors import DomainError
from src.models.vacancy import Vacancy

logger = structlog.get_logger("helpers")


def get_load_mode(total_vacancies: int, args, log) -> tuple:
    threshold = args.async_threshold
    if not args.use_async:
        return False, 0, "async_disabled_by_user"
    if total_vacancies > threshold:
        msg = f"⚠️  Ожидается {total_vacancies} вакансий — переключение на синхронную загрузку"
        print(f"  {msg}")
        log.warning("rate_limit_protection_activated", expected=total_vacancies, threshold=threshold)
        return False, 0, "sync_mode_large_volume"
    print(f"  ✓ Асинхронная загрузка ({args.async_workers} воркеров)")
    log.info("async_mode_activated", workers=args.async_workers)
    return True, args.async_workers, "async_mode"


def _load_details_progress(current: int, total: int, phase: str):
    from src.pipeline.progress import write as pwrite
    pct = 10 + int(current / total * 10) if total else 10
    pwrite(pct, f"Загрузка деталей: {current}/{total} ({phase})")


def load_vacancies_details(basic_vacancies: list, hh_api, use_async: bool, async_workers: int, parser, log) -> Result[list, DomainError]:
    print("  Загрузка детальной информации по вакансиям...")
    log.info("loading_vacancy_details_started")
    _load_details_progress(0, len(basic_vacancies), "старт")

    if use_async:
        try:
            from src.parsing.api.hh_api_async import HeadHunterAPIAsync

            api_async = HeadHunterAPIAsync(
                max_concurrent=async_workers,
                request_delay=config.REQUEST_DELAY,
                token=hh_api._token,
                token_expires_at=hh_api._token_expires_at,
            )
            vacancy_ids = [v.get("id") if isinstance(v, dict) else v.id for v in basic_vacancies]
            start = time.time()
            _load_details_progress(0, len(vacancy_ids), "асинхронно")
            if config.PYDANTIC_VALIDATION_ENABLED:
                match api_async.get_vacancies_details_sync_validated(vacancy_ids):
                    case Ok(results):
                        detailed = [Vacancy.from_api(r.model_dump()) for r in results]
                    case Err(e):
                        logger.error("async_validated_loading_failed", error=str(e))
                        return Err(DomainError(message=f"Async validated loading failed: {e}", detail=str(e)))
            else:
                match api_async.get_vacancies_details_sync(vacancy_ids):
                    case Ok(results):
                        detailed = [Vacancy.from_api(r) for r in results]
                    case Err(e):
                        logger.error("async_loading_failed", error=str(e))
                        return Err(DomainError(message=f"Async loading failed: {e}", detail=str(e)))
            elapsed = time.time() - start
            _load_details_progress(len(detailed), len(vacancy_ids), "готово")
            print(f"  ✓ Загружено {len(detailed)}/{len(vacancy_ids)} вакансий за {elapsed:.1f} сек")
            log.info("async_loading_completed", elapsed=round(elapsed, 1), loaded=len(detailed), total=len(vacancy_ids))
            return Ok(detailed)
        except Exception as e:
            print(f"  ⚠️  Ошибка асинхронной загрузки: {e}")
            log.warning("async_loading_failed_fallback_to_sync", error=str(e))

    # Синхронная загрузка
    detailed = []
    total = len(basic_vacancies)
    start = time.time()
    for idx, vac in tqdm(enumerate(basic_vacancies, 1), total=total, desc="Загрузка вакансий"):
        if idx % 10 == 0:
            _load_details_progress(idx, total, "синхронно")
        vac_id = vac.get("id") if isinstance(vac, dict) else vac.id
        if config.PYDANTIC_VALIDATION_ENABLED:
            match hh_api.get_vacancy_details_validated(vac_id):
                case Ok(validated):
                    det = Vacancy.from_api(validated.model_dump())
                case Err(e):
                    logger.warning("vacancy_validation_failed", vac_id=vac_id, error=str(e))
                    det = None
        else:
            match hh_api.get_vacancy_details_as_object(vac_id):
                case Ok(v):
                    det = v
                case Err(e):
                    logger.warning("vacancy_details_failed", vac_id=vac_id, error=str(e))
                    det = None
        if det:
            detailed.append(det)
        time.sleep(config.REQUEST_DELAY)
    elapsed = time.time() - start
    _load_details_progress(len(detailed), total, "готово")
    print(f"  ✓ Загружено {len(detailed)}/{total} вакансий за {elapsed / 60:.1f} мин")
    log.info("sync_loading_completed", elapsed=round(elapsed / 60, 1), loaded=len(detailed), total=total)
    return Ok(detailed)


def save_detailed_vacancies(vacancies, log) -> Result[None, DomainError]:
    file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        data = [v.raw_data if isinstance(v, Vacancy) else v for v in vacancies]
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("detailed_vacancies_saved", path=str(file), count=len(vacancies))
        return Ok(None)
    except Exception as e:
        return Err(DomainError(message="Failed to save detailed vacancies", detail=str(e)))


def console_info(msg: str):
    print(f"  {msg}")


def console_header(msg: str):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")
