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
        batch_size: int = 50
    ):
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay or 0.5
        self.batch_size = batch_size
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
        max_retries: int = 5
    ) -> Optional[Dict]:
        """Асинхронный GET-запрос с обработкой ошибок"""
        async with self.semaphore:
            await self._throttle()
            
            try:
                async with session.get(
                    url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=30),
                    headers={
                        'User-Agent': config.HH_USER_AGENT,
                        'Accept': 'application/json; charset=utf-8'
                    }
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    
                    elif resp.status == 429:
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
                if retries < max_retries:
                    await asyncio.sleep(2 ** retries)  # Экспоненциальная задержка
                    return await self._request(session, url, params, retries + 1, max_retries)
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
        """Получение деталей одной вакансии асинхронно"""
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        return await self._request(session, url)

    async def get_vacancies_details_batch(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """Загружает детали массива вакансий асинхронно с пакетной обработкой"""
        if not vacancy_ids:
            logger.warning("Пустой список ID вакансий")
            return []

        logger.info(f"Асинхронная загрузка {len(vacancy_ids)} деталей (max_concurrent={self.max_concurrent})...")
        
        all_results = []
        
        # Разбиваем на пакеты
        for i in range(0, len(vacancy_ids), self.batch_size):
            batch = vacancy_ids[i:i + self.batch_size]
            logger.info(f"Пакет {i//self.batch_size + 1}/{(len(vacancy_ids)-1)//self.batch_size + 1} ({len(batch)} вакансий)")
            
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.get_vacancy_details_async(session, vid)
                    for vid in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Обрабатываем результаты
                for r in results:
                    if r is not None and not isinstance(r, Exception):
                        all_results.append(r)
                    elif isinstance(r, Exception):
                        logger.error(f"Ошибка при загрузке: {r}")
            
            logger.info(f"Прогресс: {len(all_results)}/{len(vacancy_ids)}")
            
            # Пауза между пакетами
            if i + self.batch_size < len(vacancy_ids):
                await asyncio.sleep(2)
        
        logger.info(f"Загрузка завершена: {len(all_results)}/{len(vacancy_ids)} деталей")
        return all_results

    def get_vacancies_details_sync(
        self, 
        vacancy_ids: List[str]
    ) -> List[Dict]:
        """Синхронная обёртка для асинхронной загрузки"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_vacancies_details_batch(vacancy_ids))