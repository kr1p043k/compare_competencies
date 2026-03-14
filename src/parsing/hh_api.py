import requests
import time
from typing import List, Dict, Any, Optional
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src import config

logger = logging.getLogger(__name__)

class HeadHunterAPI:
    """
    Класс для взаимодействия с API hh.ru.
    Основан на официальной документации: https://api.hh.ru/openapi/redoc
    """
    BASE_URL = "https://api.hh.ru/"

    def __init__(self, use_proxy: bool = False):
        self.session = requests.Session()

        # Настройка повторных попыток при сбоях
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Заголовки, обязательные для API hh.ru
        self.session.headers.update({
            'User-Agent': config.HH_USER_AGENT,
            'Accept': 'application/json; charset=utf-8'
        })

        # Настройка прокси (опционально)
        if use_proxy and config.USE_PROXY:
            self.session.proxies.update(config.PROXY)

        logger.info(f"Инициализирован HeadHunterAPI с User-Agent: {config.HH_USER_AGENT}")

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Базовый GET-запрос с обработкой ошибок и повторными попытками.
        """
        try:
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=10)
            elapsed = time.time() - start_time

            logger.debug(f"Запрос: GET {response.url}")
            logger.debug(f"Статус: {response.status_code}, время: {elapsed:.2f} сек")

            if response.status_code == 304:
                logger.debug("Данные не изменились (304)")
                return None
            elif response.status_code == 404:
                logger.error(f"Ресурс не найден: {url}")
                return None
            elif response.status_code == 403:
                logger.error("Доступ запрещён. Возможно, проблема с User-Agent или IP заблокирован.")
                return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Таймаут запроса к {url}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Ошибка соединения с {url}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка: {e}")
            if response.status_code == 400:
                logger.error(f"Тело ответа: {response.text}")
        except Exception as e:
            logger.error(f"Неизвестная ошибка: {e}")

        return None

    def get_areas(self, country_id: Optional[int] = None) -> List[Dict]:
        """
        Получение списка регионов.
        Если указан country_id, возвращает регионы только этой страны.
        """
        url = f"{self.BASE_URL}areas"
        if country_id:
            url += f"/{country_id}"

        data = self._get(url)
        if not data:
            return []

        # Если запрашивали конкретную страну, оборачиваем в список для единообразия
        if country_id:
            return [data] if data else []

        return data

    def search_vacancies(self,
                        text: str,
                        area: Optional[int] = None,
                        period_days: int = 30,
                        max_pages: int = 20,
                        per_page: int = 100,
                        search_fields: Optional[List[str]] = None,
                        employment: Optional[str] = None,
                        schedule: Optional[str] = None,
                        experience: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Поиск вакансий с полным набором параметров согласно документации.
        """
        all_vacancies = []
        page = 0

        # Базовые параметры
        params = {
            'text': text,
            'period': period_days,
            'per_page': min(per_page, 100),  # API ограничивает 100
            'page': page
        }

        if area:
            params['area'] = area
        if search_fields:
            # API допускает множественные значения search_field
            params['search_field'] = search_fields
        if employment:
            params['employment'] = employment
        if schedule:
            params['schedule'] = schedule
        if experience:
            params['experience'] = experience

        # Дополнительные полезные параметры
        params['order_by'] = 'publication_time'  # Сортировка по дате
        params['clusters'] = False               # Не нужны кластеры
        params['describe_arguments'] = False     # Не нужно описание аргументов

        logger.info(f"Начинаем поиск вакансий по запросу '{text}' с параметрами: {params}")

        while page < max_pages:
            data = self._get(f"{self.BASE_URL}vacancies", params=params)

            if not data or 'items' not in data:
                logger.error("Не удалось получить данные или неверный формат ответа")
                break

            items = data['items']
            if not items:
                logger.info("Вакансии на странице отсутствуют, завершаем.")
                break

            all_vacancies.extend(items)
            logger.info(f"Страница {page + 1}: получено {len(items)} вакансий (всего найдено: {data.get('found', 0)})")

            # Проверяем, есть ли ещё страницы
            if page >= data.get('pages', 0) - 1:
                logger.info("Достигнута последняя страница.")
                break

            page += 1
            params['page'] = page
            time.sleep(config.REQUEST_DELAY)

        logger.info(f"Всего загружено {len(all_vacancies)} вакансий по запросу '{text}'.")
        return all_vacancies

    def get_vacancy_details(self, vacancy_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение детальной информации по конкретной вакансии.
        Этот метод возвращает ПОЛНЫЕ данные, включая key_skills.
        """
        url = f"{self.BASE_URL}vacancies/{vacancy_id}"
        logger.debug(f"Запрос деталей вакансии {vacancy_id}")

        data = self._get(url)

        if data:
            logger.debug(f"Успешно загружены детали вакансии {vacancy_id}")
            # Проверяем наличие ключевых навыков
            if 'key_skills' in data:
                logger.debug(f"Найдено ключевых навыков: {len(data['key_skills'])}")
        else:
            logger.warning(f"Не удалось загрузить детали вакансии {vacancy_id}")

        return data

    def get_employer_vacancies(self, employer_id: int, area_id: Optional[int] = None,
                                page: int = 0, per_page: int = 100) -> Optional[Dict[str, Any]]:
        """
        Получение вакансий конкретного работодателя.
        """
        params = {
            'employer_id': employer_id,
            'page': page,
            'per_page': per_page
        }
        if area_id:
            params['area'] = area_id

        return self._get(f"{self.BASE_URL}vacancies", params=params)