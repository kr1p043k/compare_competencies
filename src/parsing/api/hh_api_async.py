# src/parsing/hh_api_async.py

import asyncio
import random
import time
from typing import cast

import aiohttp
import structlog

from src import config
from src.errors import ApiError, RateLimitError, VacancyNotFoundError
from src.models.hh_responses import (
    VacancyDetailResponse,
    parse_response,
)
from src.parsing.api.hh_api import HeadHunterAPI
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)


class HeadHunterAPIAsync:
    BASE_URL = "https://api.hh.ru/"

    def __init__(
        self,
        max_concurrent: int = 2,
        request_delay: float | None = None,
        batch_size: int = 50,
        token: str | None = None,
        token_expires_at: float = 0,
    ):
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay or 0.5
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0
        self.stats = {"success": 0, "403_errors": 0, "404_errors": 0, "429_errors": 0, "timeouts": 0, "other_errors": 0}
        self._progress_file = config.BASE_DIR / "data" / "cache" / "pipeline_progress.json"

        self._progress_file = (
            config.BASE_DIR / "data" / "cache" / "pipeline_progress.json"
        ) if config.BASE_DIR else None

        # === АВТОРИЗАЦИЯ ===
        self._token = token
        self._token_expires_at = token_expires_at
        self._token_lock = asyncio.Lock()

    async def get_vacancy_details_validated(
        self, session: aiohttp.ClientSession, vacancy_id: str
    ) -> Result[VacancyDetailResponse, ApiError]:
        """Асинхронное получение деталей с валидацией."""
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        match await self._request(session, url):
            case Ok(raw):
                return Ok(cast(VacancyDetailResponse, parse_response(raw, VacancyDetailResponse)))
            case Err(e):
                return Err(ApiError(message=f"Vacancy {vacancy_id} not found or API error", endpoint=f"vacancies/{vacancy_id}", detail=str(e)))

    def _write_progress(self, loaded: int, total: int):
        if not self._progress_file:
            return
        try:
            import json
            pct = int(loaded / total * 10) if total else 10
            self._progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._progress_file, "w", encoding="utf-8") as f:
                json.dump({"pct": pct, "message": f"Загрузка деталей: {loaded}/{total}"}, f, ensure_ascii=False)
        except Exception:
            pass

    async def _ensure_token(self) -> Result[bool, ApiError]:
        if self._token and time.time() < self._token_expires_at:
            return Ok(True)

        async with self._token_lock:
            if self._token and time.time() < self._token_expires_at:
                return Ok(True)

            if not config.HH_CLIENT_ID or not config.HH_CLIENT_SECRET:
                return Err(ApiError(message="HH credentials not set for async", endpoint="token"))

            sync_api = HeadHunterAPI()
            if (
                hasattr(sync_api, "_token")
                and sync_api._token
                and hasattr(sync_api, "_token_expires_at")
                and time.time() < sync_api._token_expires_at
            ):
                self._token = sync_api._token
                self._token_expires_at = sync_api._token_expires_at
                logger.info("token_reused_from_sync_api")
                return Ok(True)

            return await self._get_app_token()

    async def _get_app_token(self) -> Result[bool, ApiError]:
        url = "https://api.hh.ru/token"
        client_id = config.HH_CLIENT_ID.get_secret_value() if config.HH_CLIENT_ID else None
        client_secret = config.HH_CLIENT_SECRET.get_secret_value() if config.HH_CLIENT_SECRET else None
        if not client_id or not client_secret:
            return Err(ApiError(message="HH credentials not set for async token", endpoint="token"))

        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        headers = {"User-Agent": config.HH_USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"}

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, data=payload, headers=headers) as resp,
            ):
                if resp.status == 200:
                    token_data = await resp.json()
                    self._token = token_data.get("access_token")
                    expires_in = token_data.get("expires_in", 86400)
                    self._token_expires_at = time.time() + expires_in - 60
                    logger.info("app_token_obtained_async")
                    return Ok(True)
                else:
                    text = await resp.text()
                    return Err(ApiError(message=f"Token request failed: {resp.status}", status_code=resp.status, endpoint="token", detail=text[:200]))
        except Exception as e:
            return Err(ApiError(message=f"Token request exception: {e}", endpoint="token"))

    def _get_headers(self) -> dict[str, str]:
        """Формирует заголовки с авторизацией если токен есть."""
        headers = {"User-Agent": config.HH_USER_AGENT, "Accept": "application/json; charset=utf-8"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _throttle(self):
        """Соблюдает rate limit между запросами."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    async def _request_result(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict | None = None,
        retries: int = 0,
        max_retries: int = 5,
    ) -> Result[dict, ApiError]:
        """Асинхронный GET с Result[T, E] — все ошибки типизированы."""
        async with self.semaphore:
            await self._throttle()

            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30), headers=self._get_headers()
                ) as resp:
                    if resp.status == 200:
                        self.stats["success"] += 1
                        return Ok(await resp.json())

                    if resp.status == 429:
                        self.stats["429_errors"] += 1
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning("rate_limited_async", retry_after=retry_after, attempt=retries + 1)
                        await asyncio.sleep(retry_after)
                        if retries < max_retries:
                            return await self._request_result(session, url, params, retries + 1, max_retries)
                        logger.error("max_retries_exceeded_async", url=url)
                        return Err(RateLimitError(message="Max retries exceeded", endpoint=url, retry_after=retry_after))

                    if resp.status == 401:
                        if retries < 1:
                            logger.warning("http_401_refreshing_token_async")
                            match await self._ensure_token():
                                case Err(e):
                                    logger.error("token_refresh_failed_async", error=str(e))
                                case _:
                                    pass
                            return await self._request_result(session, url, params, retries + 1, max_retries)
                        logger.error("http_401_after_token_refresh_async")
                        return Err(ApiError(message="Unauthorized after token refresh", status_code=401, endpoint=url))

                    if resp.status == 403:
                        self.stats["403_errors"] += 1
                        if retries < 1:
                            logger.warning("http_403_refreshing_token_async")
                            self._token = None
                            self._token_expires_at = 0
                            match await self._ensure_token():
                                case Err(e):
                                    logger.error("token_refresh_failed_async", error=str(e))
                                case _:
                                    pass
                            return await self._request_result(session, url, params, retries + 1, max_retries)
                        logger.error("http_403_after_token_refresh_async")
                        return Err(ApiError(message="Forbidden after token refresh", status_code=403, endpoint=url))

                    if resp.status == 404:
                        self.stats["404_errors"] += 1
                        logger.debug("http_404_async", url=url)
                        return Err(VacancyNotFoundError(message=f"Not found: {url}", detail=url))

                    if resp.status >= 500:
                        self.stats["other_errors"] += 1
                        if retries < max_retries:
                            backoff = min(2**retries, 30)
                            jitter = random.uniform(0, 0.5)
                            wait_time = backoff + jitter
                            logger.warning("http_server_error_async", status=resp.status, attempt=retries + 1, wait=round(wait_time, 2))
                            await asyncio.sleep(wait_time)
                            return await self._request_result(session, url, params, retries + 1, max_retries)
                        logger.error("max_retries_exceeded_async", url=url, status=resp.status)
                        return Err(ApiError(message=f"Server error {resp.status} max retries", status_code=resp.status, endpoint=url))

                    self.stats["other_errors"] += 1
                    logger.warning("http_unexpected_status_async", status=resp.status, url=url)
                    return Err(ApiError(message=f"Unexpected status {resp.status}", status_code=resp.status, endpoint=url))

            except TimeoutError:
                self.stats["timeouts"] += 1
                if retries < max_retries:
                    backoff = min(2**retries, 30)
                    jitter = random.uniform(0, 0.5)
                    wait_time = backoff + jitter
                    logger.warning("http_timeout_async", url=url, attempt=retries + 1, wait=round(wait_time, 2))
                    await asyncio.sleep(wait_time)
                    return await self._request_result(session, url, params, retries + 1, max_retries)
                logger.error("max_timeout_retries_exceeded", url=url)
                return Err(ApiError(message=f"Timeout max retries: {url}", endpoint=url))
            except aiohttp.ClientError as e:
                self.stats["other_errors"] += 1
                logger.error("http_client_error_async", error=str(e))
                return Err(ApiError(message=str(e), endpoint=url))
            except Exception as e:
                self.stats["other_errors"] += 1
                logger.error("http_unexpected_exception_async", error=str(e))
                return Err(ApiError(message=str(e), endpoint=url))

    async def _request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict | None = None,
        retries: int = 0,
        max_retries: int = 5,
    ) -> Result[dict, ApiError]:
        return await self._request_result(session, url, params, retries, max_retries)

    async def get_vacancy_details_async(self, session: aiohttp.ClientSession, vacancy_id: str) -> Result[dict, ApiError]:
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        return await self._request(session, url)

    async def get_vacancies_details_batch_validated(self, vacancy_ids: list[str]) -> Result[list[VacancyDetailResponse], ApiError]:
        if not vacancy_ids:
            return Ok([])
        match await self._ensure_token():
            case Err(e):
                logger.warning("token_not_available", error=str(e))
            case _:
                pass
        logger.info("async_validated_batch_started", total=len(vacancy_ids))
        all_results = []
        total = len(vacancy_ids)
        batch_size = self.batch_size
        for i in range(0, total, batch_size):
            batch = vacancy_ids[i : i + batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [self.get_vacancy_details_validated(session, vid) for vid in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    match r:
                        case Ok(v) if isinstance(v, VacancyDetailResponse):
                            all_results.append(v)
                        case Err(e):
                            logger.debug("vacancy_validated_skipped", error=str(e))
                        case e if isinstance(e, Exception):
                            logger.debug("vacancy_exception_skipped", error=str(e))
            logger.info(
                "async_validated_batch_progress",
                done=len(all_results),
                total=total,
                percent=round(len(all_results) / total * 100, 1),
            )
            if i + batch_size < total:
                await asyncio.sleep(1)
        logger.info("async_validated_batch_completed", loaded=len(all_results))
        return Ok(all_results)

    def get_vacancies_details_sync_validated(self, vacancy_ids: list[str]) -> Result[list[VacancyDetailResponse], ApiError]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_vacancies_details_batch_validated(vacancy_ids))
        except Exception as e:
            return Err(ApiError(message="Sync validated batch failed", endpoint="vacancies", detail=str(e)))

    async def get_vacancies_details_batch(self, vacancy_ids: list[str]) -> Result[list[dict], ApiError]:
        if not vacancy_ids:
            return Ok([])

        await self._ensure_token()

        self._write_progress(0, len(vacancy_ids))
        logger.info("async_batch_loading_started", total=len(vacancy_ids), max_concurrent=self.max_concurrent)

        all_results = []
        self.stats = {k: 0 for k in self.stats}

        total_batches = (len(vacancy_ids) + self.batch_size - 1) // self.batch_size
        self._write_progress(0, total_batches)

        for i in range(0, len(vacancy_ids), self.batch_size):
            batch = vacancy_ids[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info("processing_batch", batch=f"{batch_num}/{total_batches}", size=len(batch))

            async with aiohttp.ClientSession() as session:
                tasks = [self.get_vacancy_details_async(session, vid) for vid in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    match r:
                        case Ok(d):
                            all_results.append(d)
                        case Err(e):
                            logger.debug("vacancy_skipped_due_to_error", error=str(e))
                        case _:
                            pass

            self._write_progress(len(all_results), len(vacancy_ids))
            logger.info("batch_progress", loaded=len(all_results), total=len(vacancy_ids))

            if i + self.batch_size < len(vacancy_ids):
                await asyncio.sleep(1)

        logger.info(
            "async_batch_loading_completed",
            loaded=len(all_results),
            total=len(vacancy_ids),
            success_rate=round((len(all_results) / len(vacancy_ids)) * 100, 1) if vacancy_ids else 0,
        )
        return Ok(all_results)

    def get_vacancies_details_sync(self, vacancy_ids: list[str]) -> Result[list[dict], ApiError]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.get_vacancies_details_batch(vacancy_ids))
        except Exception as e:
            return Err(ApiError(message="Sync batch failed", endpoint="vacancies", detail=str(e)))
