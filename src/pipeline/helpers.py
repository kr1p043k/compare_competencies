"""
Вспомогательные функции для загрузки детальных вакансий.
Используются DataSource и main.py.
"""

import json
import time

from tqdm import tqdm

from src import config
from src.models.vacancy import Vacancy


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


def load_vacancies_details(basic_vacancies: list, hh_api, use_async: bool, async_workers: int, parser, log) -> list:
    print("  Загрузка детальной информации по вакансиям...")
    log.info("loading_vacancy_details_started")

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
            if config.PYDANTIC_VALIDATION_ENABLED:
                detailed = [
                    Vacancy.from_api(r.model_dump())
                    for r in api_async.get_vacancies_details_sync_validated(vacancy_ids)
                ]
            else:
                detailed = [
                    Vacancy.from_api(r)
                    for r in api_async.get_vacancies_details_sync(vacancy_ids)
                    if not isinstance(r, Exception)
                ]
            elapsed = time.time() - start
            print(f"  ✓ Загружено {len(detailed)}/{len(vacancy_ids)} вакансий за {elapsed:.1f} сек")
            log.info("async_loading_completed", elapsed=round(elapsed, 1), loaded=len(detailed), total=len(vacancy_ids))
            return detailed
        except Exception as e:
            print(f"  ⚠️  Ошибка асинхронной загрузки: {e}")
            log.warning("async_loading_failed_fallback_to_sync", error=str(e))

    # Синхронная загрузка
    detailed = []
    total = len(basic_vacancies)
    start = time.time()
    for _, vac in tqdm(enumerate(basic_vacancies, 1), total=total, desc="Загрузка вакансий"):
        vac_id = vac.get("id") if isinstance(vac, dict) else vac.id
        if config.PYDANTIC_VALIDATION_ENABLED:
            try:
                validated = hh_api.get_vacancy_details_validated(vac_id)
                det = Vacancy.from_api(validated.model_dump())
            except Exception:
                det = None
        else:
            det = hh_api.get_vacancy_details_as_object(vac_id)
        if det:
            detailed.append(det)
        time.sleep(config.REQUEST_DELAY)
    elapsed = time.time() - start
    print(f"  ✓ Загружено {len(detailed)}/{total} вакансий за {elapsed / 60:.1f} мин")
    log.info("sync_loading_completed", elapsed=round(elapsed / 60, 1), loaded=len(detailed), total=total)
    return detailed


def save_detailed_vacancies(vacancies, log):
    file = config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    data = [v.raw_data if isinstance(v, Vacancy) else v for v in vacancies]
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("detailed_vacancies_saved", path=str(file), count=len(vacancies))


def console_info(msg: str):
    print(f"  {msg}")


def console_header(msg: str):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")
