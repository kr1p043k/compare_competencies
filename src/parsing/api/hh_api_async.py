# src/parsing/hh_api_async.py

import asyncio
import random
import time
from typing import cast

import aiohttp
import structlog

from src import config
from src.models.hh_responses import (
    VacancyDetailResponse,
    parse_response,
)
from src.parsing.api.hh_api import HeadHunterAPI

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

        # === АВТОРИЗАЦИЯ ===
        self._token = token
        self._token_expires_at = token_expires_at
        self._token_lock = asyncio.Lock()

    async def get_vacancy_details_validated(
        self, session: aiohttp.ClientSession, vacancy_id: str
    ) -> VacancyDetailResponse:
        """Асинхронное получение деталей с валидацией."""
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        raw = await self._request(session, url)
        if raw is None:
            raise ValueError(f"Vacancy {vacancy_id} not found or API error")
        return cast(VacancyDetailResponse, parse_response(raw, VacancyDetailResponse))

    async def _ensure_token(self):
        """Проверяет и обновляет токен при необходимости."""
        if self._token and time.time() < self._token_expires_at:
            return

        async with self._token_lock:
            if self._token and time.time() < self._token_expires_at:
                return

            # Проверяем наличие учётных данных как SecretStr
            if not config.HH_CLIENT_ID or not config.HH_CLIENT_SECRET:
                logger.warning("hh_credentials_not_set_async")
                return

            sync_api = HeadHunterAPI()
            # Безопасно проверяем наличие и валидность токена
            if (
                hasattr(sync_api, "_token")
                and sync_api._token
                and hasattr(sync_api, "_token_expires_at")
                and time.time() < sync_api._token_expires_at
            ):
                self._token = sync_api._token
                self._token_expires_at = sync_api._token_expires_at
                logger.info("token_reused_from_sync_api")
                return

            await self._get_app_token()

    async def _get_app_token(self):
        """Получает application access token."""
        url = "https://api.hh.ru/token"
        # Извлекаем значения из SecretStr
        client_id = config.HH_CLIENT_ID.get_secret_value() if config.HH_CLIENT_ID else None
        client_secret = config.HH_CLIENT_SECRET.get_secret_value() if config.HH_CLIENT_SECRET else None
        if not client_id or not client_secret:
            logger.warning("hh_credentials_not_set_async_token")
            return

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
                else:
                    text = await resp.text()
                    if resp.status == 403:
                        logger.warning("token_403_working_without_auth")
                    else:
                        logger.error("token_request_failed_async", status=resp.status, response=text[:200])
        except Exception as e:
            logger.error("token_request_exception_async", error=str(e))

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

    async def _request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict | None = None,
        retries: int = 0,
        max_retries: int = 5,
    ) -> dict | None:
        """Асинхронный GET-запрос с обработкой ошибок, exponential backoff и авторизацией."""
        async with self.semaphore:
            await self._throttle()

            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=30), headers=self._get_headers()
                ) as resp:
                    if resp.status == 200:
                        self.stats["success"] += 1
                        return await resp.json()

                    # 429 – Too Many Requests
                    if resp.status == 429:
                        self.stats["429_errors"] += 1
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning("rate_limited_async", retry_after=retry_after, attempt=retries + 1)
                        await asyncio.sleep(retry_after)
                        if retries < max_retries:
                            return await self._request(session, url, params, retries + 1, max_retries)
                        logger.error("max_retries_exceeded_async", url=url)
                        return None

                    # 401 – токен истёк, обновим и попробуем ещё раз
                    if resp.status == 401:
                        if retries < 1:
                            logger.warning("http_401_refreshing_token_async")
                            await self._ensure_token()
                            return await self._request(session, url, params, retries + 1, max_retries)
                        logger.error("http_401_after_token_refresh_async")
                        return None

                    # 403 – нет прав
                    if resp.status == 403:
                        self.stats["403_errors"] += 1
                        try:
                            vacancy_id = url.rstrip("/").split("/")[-1]
                            logger.debug("vacancy_403_forbidden_async", vacancy_id=vacancy_id)
                        except Exception:
                            logger.debug("http_403_forbidden_async", url=url)
                        return None

                    # 404 – не найдено
                    if resp.status == 404:
                        self.stats["404_errors"] += 1
                        logger.debug("http_404_async", url=url)
                        return None

                    # 5xx – временная ошибка сервера, стоит повторить
                    if resp.status >= 500:
                        self.stats["other_errors"] += 1
                        if retries < max_retries:
                            backoff = min(2**retries, 30)  # экспоненциальная задержка
                            jitter = random.uniform(0, 0.5)  # небольшой разброс
                            wait_time = backoff + jitter
                            logger.warning(
                                "http_server_error_async",
                                status=resp.status,
                                attempt=retries + 1,
                                wait=round(wait_time, 2),
                            )
                            await asyncio.sleep(wait_time)
                            return await self._request(session, url, params, retries + 1, max_retries)
                        logger.error("max_retries_exceeded_async", url=url, status=resp.status)
                        return None

                    # Неизвестный статус
                    self.stats["other_errors"] += 1
                    logger.warning("http_unexpected_status_async", status=resp.status, url=url)
                    return None

            except TimeoutError:
                self.stats["timeouts"] += 1
                if retries < max_retries:
                    backoff = min(2**retries, 30)
                    jitter = random.uniform(0, 0.5)
                    wait_time = backoff + jitter
                    logger.warning("http_timeout_async", url=url, attempt=retries + 1, wait=round(wait_time, 2))
                    await asyncio.sleep(wait_time)
                    return await self._request(session, url, params, retries + 1, max_retries)
                logger.error("max_timeout_retries_exceeded", url=url)
                return None
            except aiohttp.ClientError as e:
                self.stats["other_errors"] += 1
                logger.error("http_client_error_async", error=str(e))
                return None
            except Exception as e:
                self.stats["other_errors"] += 1
                logger.error("http_unexpected_exception_async", error=str(e))
                return None

    async def get_vacancy_details_async(self, session: aiohttp.ClientSession, vacancy_id: str) -> dict | None:
        """Получение деталей одной вакансии асинхронно."""
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        return await self._request(session, url)

    async def get_vacancies_details_batch_validated(self, vacancy_ids: list[str]) -> list[VacancyDetailResponse]:
        if not vacancy_ids:
            return []
        await self._ensure_token()
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
                    if isinstance(r, VacancyDetailResponse):
                        all_results.append(r)
                    elif isinstance(r, Exception):
                        logger.debug("vacancy_validated_skipped", error=str(r))
            # Прогресс
            logger.info(
                "async_validated_batch_progress",
                done=len(all_results),
                total=total,
                percent=round(len(all_results) / total * 100, 1),
            )
            if i + batch_size < total:
                await asyncio.sleep(1)
        logger.info("async_validated_batch_completed", loaded=len(all_results))
        return all_results

    def get_vacancies_details_sync_validated(self, vacancy_ids: list[str]) -> list[VacancyDetailResponse]:
        """Синхронная обёртка для пакетной валидированной загрузки."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.get_vacancies_details_batch_validated(vacancy_ids))

    async def get_vacancies_details_batch(self, vacancy_ids: list[str]) -> list[dict]:
        """Загружает детали массива вакансий асинхронно с пакетной обработкой."""
        if not vacancy_ids:
            logger.warning("empty_vacancy_ids_batch")
            return []

        await self._ensure_token()

        logger.info("async_batch_loading_started", total=len(vacancy_ids), max_concurrent=self.max_concurrent)

        all_results = []
        self.stats = {k: 0 for k in self.stats}

        total_batches = (len(vacancy_ids) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(vacancy_ids), self.batch_size):
            batch = vacancy_ids[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info("processing_batch", batch=f"{batch_num}/{total_batches}", size=len(batch))

            async with aiohttp.ClientSession() as session:
                tasks = [self.get_vacancy_details_async(session, vid) for vid in batch]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if r is not None and not isinstance(r, Exception):
                        all_results.append(r)
                    elif isinstance(r, Exception):
                        logger.debug("vacancy_skipped_due_to_error", error=str(r))

            logger.info("batch_progress", loaded=len(all_results), total=len(vacancy_ids))
            logger.debug(
                "batch_stats",
                success=self.stats["success"],
                forbidden=self.stats["403_errors"],
                not_found=self.stats["404_errors"],
                rate_limited=self.stats["429_errors"],
                timeouts=self.stats["timeouts"],
            )

            if i + self.batch_size < len(vacancy_ids):
                await asyncio.sleep(1)

        success_rate = (len(all_results) / len(vacancy_ids)) * 100 if vacancy_ids else 0
        logger.info(
            "async_batch_loading_completed",
            loaded=len(all_results),
            total=len(vacancy_ids),
            success_rate=round(success_rate, 1),
        )
        logger.info(
            "async_batch_final_stats",
            success=self.stats["success"],
            forbidden=self.stats["403_errors"],
            not_found=self.stats["404_errors"],
            rate_limited=self.stats["429_errors"],
            timeouts=self.stats["timeouts"],
        )

        return all_results

    def get_vacancies_details_sync(self, vacancy_ids: list[str]) -> list[dict]:
        """Синхронная обёртка для асинхронной загрузки."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.get_vacancies_details_batch(vacancy_ids))
