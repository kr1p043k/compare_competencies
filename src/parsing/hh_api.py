import requests
import time
from typing import List, Dict, Any, Optional
import logging
from src import config

logger = logging.getLogger(__name__)

class HeadHunterAPI:
    BASE_URL = "https://api.hh.ru/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': config.HH_USER_AGENT})
        logger.info("Инициализирован HeadHunterAPI")

    def _get(self, url: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        try:
            start_time = time.time()
            response = self.session.get(url, params=params)
            elapsed = time.time() - start_time
            logger.debug(f"Запрос: GET {response.url}, статус: {response.status_code}, время: {elapsed:.2f} сек")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Тело ответа: {e.response.text}")
            return None

    def get_areas(self) -> List[tuple]:
        """Получает список всех регионов."""
        data = self._get(f"{self.BASE_URL}areas")
        if not data:
            return []
        areas_list = []
        for country in data:
            country_id = country['id']
            country_name = country['name']
            for region in country.get('areas', []):
                if region.get('areas'):
                    for city in region['areas']:
                        areas_list.append((country_id, country_name, city['id'], city['name']))
                else:
                    areas_list.append((country_id, country_name, region['id'], region['name']))
        return areas_list

    def search_vacancies(self, text: str, area: Optional[int] = None,
                         period_days: int = 30, max_pages: int = 20,
                         search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Поиск вакансий с возможностью указания полей для поиска.
        search_fields: список полей ('name', 'company_name', 'description').
        """
        all_vacancies = []
        page = 0
        params = {
            'text': text,
            'period': period_days,
            'per_page': 100,
            'page': page
        }
        if area:
            params['area'] = area
        if search_fields:
            # API позволяет передавать search_field несколько раз
            params['search_field'] = search_fields

        logger.info(f"Начинаем поиск вакансий по запросу '{text}' с параметрами: {params}")

        while page < max_pages:
            data = self._get(f"{self.BASE_URL}vacancies", params=params)
            if not data or 'items' not in data:
                break

            items = data['items']
            if not items:
                logger.info("Вакансии на странице отсутствуют, завершаем.")
                break

            all_vacancies.extend(items)
            logger.info(f"Страница {page + 1}: получено {len(items)} вакансий (всего найдено: {data.get('found', 0)})")

            if page >= data.get('pages', 0) - 1:
                break

            page += 1
            params['page'] = page
            time.sleep(config.REQUEST_DELAY)

        logger.info(f"Всего загружено {len(all_vacancies)} вакансий по запросу '{text}'.")
        return all_vacancies