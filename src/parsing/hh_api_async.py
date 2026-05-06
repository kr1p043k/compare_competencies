# src/parsing/hh_api_async.py

import aiohttp
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from src import config

logger = logging.getLogger(__name__)


class HeadHunterAPIAsync:
    BASE_URL = "https://api.hh.ru/"

    def __init__(
        self, 
        max_concurrent: int = 2,
        request_delay: float = None,
        batch_size: int = 50,
        token: str = None,           # ✅ можно передать готовый токен
        token_expires_at: float = 0  # ✅ и время его истечения
    ):
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay or 0.5
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0
        self.stats = {
            'success': 0,
            '403_errors': 0,
            '404_errors': 0,
            '429_errors': 0,
            'timeouts': 0,
            'other_errors': 0
        }

        # === АВТОРИЗАЦИЯ ===
        self._token = token                    # ✅ берём переданный токен
        self._token_expires_at = token_expires_at
        self._token_lock = asyncio.Lock()

    async def _ensure_token(self):
        """Проверяет и обновляет токен при необходимости."""
        # Быстрая проверка
        if self._token and time.time() < self._token_expires_at:
            return

        async with self._token_lock:
            # Повторная проверка
            if self._token and time.time() < self._token_expires_at:
                return

            if not config.HH_CLIENT_ID or not config.HH_CLIENT_SECRET:
                logger.warning("HH_CLIENT_ID или HH_CLIENT_SECRET не заданы.")
                return

            # ✅ Пробуем использовать токен из синхронного API
            from src.parsing.hh_api import HeadHunterAPI
            sync_api = HeadHunterAPI()
            if sync_api._token and time.time() < sync_api._token_expires_at:
                self._token = sync_api._token
                self._token_expires_at = sync_api._token_expires_at
                logger.info("✅ Использован токен из синхронного API")
                return

            # Если нет — получаем новый
            await self._get_app_token()

    async def _get_app_token(self):
        """Получает application access token."""
        url = "https://api.hh.ru/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": config.HH_CLIENT_ID,
            "client_secret": config.HH_CLIENT_SECRET
        }
        headers = {
            'User-Agent': config.HH_USER_AGENT,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=payload, headers=headers) as resp:
                    if resp.status == 200:
                        token_data = await resp.json()
                        self._token = token_data.get("access_token")
                        expires_in = token_data.get("expires_in", 86400)
                        self._token_expires_at = time.time() + expires_in - 60
                        logger.info("✅ Токен приложения успешно получен (async)")
                    else:
                        text = await resp.text()
                        # ✅ Если 403 — не фатально, работаем без токена
                        if resp.status == 403:
                            logger.warning("⚠️ 403 при получении токена — работаем без авторизации")
                        else:
                            logger.error(f"❌ Ошибка получения токена (async): {resp.status} - {text[:200]}")
        except Exception as e:
            logger.error(f"❌ Исключение при получении токена (async): {e}")
            
    def _get_headers(self) -> Dict[str, str]:
        """Формирует заголовки с авторизацией если токен есть."""
        headers = {
            'User-Agent': config.HH_USER_AGENT,
            'Accept': 'application/json; charset=utf-8'
        }
        if self._token:
            headers['Authorization'] = f'Bearer {self._token}'
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
        params: Optional[Dict] = None,
        retries: int = 0,
        max_retries: int = 5
    ) -> Optional[Dict]:
        """Асинхронный GET-запрос с обработкой ошибок и авторизацией."""
        async with self.semaphore:
            await self._throttle()

            try:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers=self._get_headers()
                ) as resp:
                    if resp.status == 200:
                        self.stats['success'] += 1
                        return await resp.json()

                    elif resp.status == 429:
                        self.stats['429_errors'] += 1
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited. Ждём {retry_after}с...")
                        await asyncio.sleep(retry_after)
                        if retries < max_retries:
                            return await self._request(
                                session, url, params, retries + 1, max_retries
                            )
                        else:
                            logger.error(f"Превышены попытки для {url}")
                            return None

                    elif resp.status == 401:
                        # Токен истёк — обновляем и пробуем снова
                        if retries < 1:
                            logger.warning("401, обновляю токен и пробую снова (async)")
                            await self._ensure_token()  # ✅ ИСПРАВЛЕНО: _ensure_token вместо _get_app_token (с локом)
                            return await self._request(
                                session, url, params, retries + 1, max_retries
                            )
                        else:
                            logger.error("Повторный 401 после обновления токена (async)")
                            return None

                    elif resp.status == 403:
                        self.stats['403_errors'] += 1
                        # ✅ ИСПРАВЛЕНО: логируем vacancy_id только если он есть в url
                        try:
                            vacancy_id = url.rstrip('/').split('/')[-1]
                            logger.debug(f"Вакансия {vacancy_id} недоступна (403)")
                        except Exception:
                            logger.debug(f"403 для {url}")
                        return None

                    elif resp.status == 404:
                        self.stats['404_errors'] += 1
                        logger.debug(f"404: {url}")
                        return None

                    else:
                        self.stats['other_errors'] += 1
                        logger.warning(f"HTTP {resp.status} для {url}")
                        return None

            except asyncio.TimeoutError:
                self.stats['timeouts'] += 1
                logger.error(f"Timeout при запросе {url}")
                if retries < max_retries:
                    wait_time = 2 ** retries
                    logger.debug(f"Повторная попытка через {wait_time}с...")
                    await asyncio.sleep(wait_time)
                    return await self._request(session, url, params, retries + 1, max_retries)
                return None
            except aiohttp.ClientError as e:
                self.stats['other_errors'] += 1
                logger.error(f"HTTP ошибка: {e}")
                return None
            except Exception as e:
                self.stats['other_errors'] += 1
                logger.error(f"Неожиданная ошибка: {e}")
                return None

    async def get_vacancy_details_async(
        self, 
        session: aiohttp.ClientSession, 
        vacancy_id: str
    ) -> Optional[Dict]:
        """Получение деталей одной вакансии асинхронно."""
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        return await self._request(session, url)

    async def get_vacancies_details_batch(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """Загружает детали массива вакансий асинхронно с пакетной обработкой."""
        if not vacancy_ids:
            logger.warning("Пустой список ID вакансий")
            return []

        # ✅ ИСПРАВЛЕНО: получаем токен перед началом загрузки (с локом)
        await self._ensure_token()

        logger.info(f"Асинхронная загрузка {len(vacancy_ids)} деталей (max_concurrent={self.max_concurrent})...")

        all_results = []
        self.stats = {k: 0 for k in self.stats}

        total_batches = (len(vacancy_ids) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(vacancy_ids), self.batch_size):
            batch = vacancy_ids[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(f"Пакет {batch_num}/{total_batches} ({len(batch)} вакансий)")

            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.get_vacancy_details_async(session, vid)
                    for vid in batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in results:
                    if r is not None and not isinstance(r, Exception):
                        all_results.append(r)
                    elif isinstance(r, Exception):
                        logger.debug(f"Пропущена вакансия из-за ошибки: {r}")

            logger.info(f"Прогресс: {len(all_results)}/{len(vacancy_ids)} вакансий")
            logger.debug(f"Статистика пакета: успешно={self.stats['success']}, "
                        f"403={self.stats['403_errors']}, 404={self.stats['404_errors']}, "
                        f"429={self.stats['429_errors']}, таймауты={self.stats['timeouts']}")

            if i + self.batch_size < len(vacancy_ids):
                await asyncio.sleep(1)

        success_rate = (len(all_results) / len(vacancy_ids)) * 100 if vacancy_ids else 0
        logger.info(f"Загрузка завершена: {len(all_results)}/{len(vacancy_ids)} деталей ({success_rate:.1f}%)")
        logger.info(f"Итоговая статистика: успешно={self.stats['success']}, "
                   f"403={self.stats['403_errors']}, 404={self.stats['404_errors']}, "
                   f"429={self.stats['429_errors']}, таймауты={self.stats['timeouts']}")

        return all_results

    def get_vacancies_details_sync(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """Синхронная обёртка для асинхронной загрузки."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.get_vacancies_details_batch(vacancy_ids))