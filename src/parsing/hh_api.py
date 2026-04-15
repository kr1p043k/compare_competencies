"""
HeadHunterAPI - синхронный клиент для работы с API hh.ru
"""
from datetime import datetime, timedelta
import requests
import time
from typing import List, Dict, Any, Optional
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src import config
from src.models.vacancy import Vacancy

logger = logging.getLogger(__name__)


class HeadHunterAPI:
    """
    Синхронный API клиент для hh.ru.
    Используется requests + HTTPAdapter для управления retry-логикой.
    """
    
    BASE_URL = "https://api.hh.ru/vacancies"
    BASE_URL_FULL = "https://api.hh.ru/"

    def __init__(self):
        """Инициализирует сессию с настройками retry"""
        self.session = requests.Session()
        
        # Настройка повторных попыток при сбоях
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Заголовки
        self.session.headers.update({
            'User-Agent': config.HH_USER_AGENT,
            'Accept': 'application/json; charset=utf-8'
        })
        
        # Сохраняем последний ответ для получения метаданных (например, found)
        self.last_response: Optional[Dict[str, Any]] = None
        
        logger.info(f"HeadHunterAPI инициализирован (MAX_RETRIES={config.MAX_RETRIES})")

    # =========================================================================
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # =========================================================================

    def search_vacancies(
        self,
        text: str,
        area: int,
        period_days: int = 30,
        max_pages: int = 20,
        per_page: int = 100,
        industry: Optional[int] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск вакансий по тексту
        
        Args:
            text: Поисковый запрос
            area: ID региона
            period_days: Период поиска (дней) – используется, если date_from/date_to не заданы
            max_pages: Максимальное количество страниц
            per_page: Результатов на странице (макс 100)
            industry: ID отрасли (опционально)
            date_from: Unix timestamp начала периода (опционально)
            date_to: Unix timestamp конца периода (опционально)
        
        Returns:
            Список вакансий (dict)
        """
        params = {
            'text': text,
            'area': area,
            'per_page': per_page,
            'page': 0,
            'order_by': 'publication_time',
            'clusters': False,
            'describe_arguments': False,
        }
        
        if industry:
            params['industry'] = industry
        if date_from is not None and date_to is not None:
            params['date_from'] = date_from
            params['date_to'] = date_to
        else:
            params['period'] = period_days
        
        logger.info(f"Поиск вакансий: '{text}' (регион {area}, макс {max_pages} страниц)")
        
        all_vacancies = []
        page = 0
        
        while page < max_pages:
            params['page'] = page
            
            data = self._get(self.BASE_URL, params=params)
            self.last_response = data  # сохраняем для внешнего использования
            
            if not data or 'items' not in data:
                logger.error("Не удалось получить данные")
                break
            
            items = data['items']
            if not items:
                logger.info("Нет вакансий на странице")
                break
            
            all_vacancies.extend(items)
            logger.info(f"Страница {page + 1}: получено {len(items)} вакансий (всего найдено: {data.get('found', 0)})")
            
            # Проверяем, есть ли ещё страницы
            if page >= data.get('pages', 0) - 1:
                logger.info("Достигнута последняя страница")
                break
            
            page += 1
            time.sleep(config.REQUEST_DELAY)
        
        logger.info(f"Всего загружено {len(all_vacancies)} вакансий")
        return all_vacancies

    def get_vacancy_details(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает детали конкретной вакансии
        
        Args:
            vacancy_id: ID вакансии
        
        Returns:
            Словарь с деталями вакансии или None
        """
        url = f"{self.BASE_URL_FULL}vacancies/{vacancy_id}"
        
        data = self._get(url)
        
        if data:
            logger.debug(f"Успешно загружены детали вакансии {vacancy_id}")
        else:
            logger.warning(f"Не удалось загрузить детали вакансии {vacancy_id}")
        
        return data

    def get_vacancy_details_as_object(self, vacancy_id: str) -> Optional[Vacancy]:
        """
        Получает детали вакансии и возвращает типизированный объект
        
        Args:
            vacancy_id: ID вакансии
        
        Returns:
            Объект Vacancy или None при ошибке
        """
        raw_data = self.get_vacancy_details(vacancy_id)
        if not raw_data:
            return None
        
        try:
            return Vacancy.from_api(raw_data)
        except ValueError as e:
            logger.warning(f"Невалидная вакансия {vacancy_id}: {e}")
            return None

    # =========================================================================
    # ПРИВАТНЫЕ МЕТОДЫ
    # =========================================================================

    def _get(self, url: str, params: Optional[Dict] = None, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """GET-запрос с умным retry при 429"""
        try:
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=10)
            elapsed = time.time() - start_time
            
            logger.debug(f"GET {url} - статус {response.status_code} ({elapsed:.2f}с)")
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 304:
                logger.debug("Данные не изменились (304)")
                return None
            
            elif response.status_code == 404:
                logger.warning(f"Ресурс не найден (404): {url}")
                return None
            
            elif response.status_code == 403:
                logger.error("Доступ запрещён (403). IP может быть заблокирован.")
                return None
            
            elif response.status_code == 429:
                # === УМНЫЙ RETRY ===
                retry_after = int(response.headers.get('Retry-After', 60))
                if retry_count < 3:
                    logger.warning(f"Rate limited (429). Попытка {retry_count + 1}/3. Ждём {retry_after}с...")
                    time.sleep(retry_after)
                    return self._get(url, params, retry_count + 1)
                else:
                    logger.error("Превышено максимальное количество retry при 429")
                    return None
            
            else:
                logger.warning(f"Неожиданный статус {response.status_code}")
                return None
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout при запросе к {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Ошибка соединения с {url}")
            return None
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return None

    def close(self):
        """Закрывает сессию"""
        self.session.close()
        logger.info("HeadHunterAPI сессия закрыта")

    def __enter__(self):
        """Context manager поддержка"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager поддержка"""
        self.close()