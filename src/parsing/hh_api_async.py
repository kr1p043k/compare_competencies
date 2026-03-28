# src/parsing/hh_api_async.py

import aiohttp
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from src import config

logger = logging.getLogger(__name__)


class HeadHunterAPIAsync:
    """
    Асинхронный API клиент для hh.ru.
    Загружает детали вакансий параллельно, БЕЗ кэширования.
    
    Все данные сохраняются через main.py -> VacancyParser.
    """
    BASE_URL = "https://api.hh.ru/"

    def __init__(
        self, 
        max_concurrent: int = 3, 
        request_delay: float = None
    ):
        """
        Args:
            max_concurrent: Максимальное количество параллельных запросов (3-5 безопасно)
            request_delay: Задержка между запросами в секундах (из config по умолчанию)
        """
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay or config.REQUEST_DELAY
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0

    async def _throttle(self):
        """Соблюдает rate limit между запросами"""
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
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Асинхронный GET-запрос с обработкой ошибок
        
        Args:
            session: aiohttp ClientSession
            url: URL запроса
            params: Query параметры
            retries: Текущее количество попыток
            max_retries: Максимальное количество попыток
        
        Returns:
            JSON ответ или None при ошибке
        """
        async with self.semaphore:
            await self._throttle()
            
            try:
                async with session.get(
                    url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={
                        'User-Agent': config.HH_USER_AGENT,
                        'Accept': 'application/json; charset=utf-8'
                    }
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2))
                        logger.debug(f"Rate limited. Ждём {retry_after}с...")
                        await asyncio.sleep(retry_after)
                        if retries < max_retries:
                            return await self._request(
                                session, url, params, retries + 1, max_retries
                            )
                        else:
                            logger.error(f"Превышены попытки для {url}")
                            return None
                    
                    elif resp.status == 404:
                        logger.debug(f"Ресурс не найден (404): {url}")
                        return None
                    
                    elif resp.status == 403:
                        logger.warning(f"Доступ запрещён (403): {url}")
                        return None
                    
                    else:
                        logger.warning(f"HTTP {resp.status} для {url}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout при запросе {url}")
                return None
            except aiohttp.ClientError as e:
                logger.error(f"HTTP ошибка: {e}")
                return None
            except Exception as e:
                logger.error(f"Неожиданная ошибка: {e}")
                return None

    async def get_vacancy_details_async(
        self, 
        session: aiohttp.ClientSession, 
        vacancy_id: str
    ) -> Optional[Dict]:
        """
        Получение деталей одной вакансии асинхронно
        
        Args:
            session: aiohttp ClientSession
            vacancy_id: ID вакансии
        
        Returns:
            JSON с деталями вакансии или None
        """
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        return await self._request(session, url)

    async def get_vacancies_details_batch(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """
        Загружает детали массива вакансий асинхронно.
        
        Использование:
            api_async = HeadHunterAPIAsync(max_concurrent=3)
            detailed = await api_async.get_vacancies_details_batch(vacancy_ids)
        
        Args:
            vacancy_ids: Список ID вакансий для загрузки
        
        Returns:
            Список JSON объектов с деталями (None исключены)
        """
        if not vacancy_ids:
            logger.warning("Пустой список ID вакансий")
            return []

        logger.info(f"Асинхронная загрузка {len(vacancy_ids)} деталей (max_concurrent={self.max_concurrent})...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.get_vacancy_details_async(session, vid)
                for vid in vacancy_ids
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=False)
            valid_results = [r for r in results if r is not None]
            
            logger.info(f"Загружено {len(valid_results)}/{len(vacancy_ids)} деталей")
            
            return valid_results

    def get_vacancies_details_sync(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """
        Синхронная обёртка для асинхронной загрузки.
        Используется в main.py для совместимости.
        
        Args:
            vacancy_ids: Список ID вакансий
        
        Returns:
            Список JSON объектов с деталями
        """
        return asyncio.run(self.get_vacancies_details_batch(vacancy_ids))