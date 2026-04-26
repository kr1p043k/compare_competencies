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
    Использует автоматическое получение токена через client_credentials.
    """

    BASE_URL = "https://api.hh.ru/vacancies"
    BASE_URL_FULL = "https://api.hh.ru/"

    def __init__(self):
        self.session = requests.Session()

        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update({
            'User-Agent': config.HH_USER_AGENT,
            'Accept': 'application/json; charset=utf-8'
        })

        self.last_response: Optional[Dict[str, Any]] = None

        # === АВТОМАТИЧЕСКОЕ ПОЛУЧЕНИЕ ТОКЕНА ===
        self._token = None
        self._token_expires_at = 0
        if config.HH_CLIENT_ID and config.HH_CLIENT_SECRET:
            self._get_app_token()
        else:
            logger.warning("HH_CLIENT_ID или HH_CLIENT_SECRET не заданы. Запросы будут неавторизованными.")

        logger.info(f"HeadHunterAPI инициализирован (MAX_RETRIES={config.MAX_RETRIES})")

    # -----------------------------------------------------------------------
    def _get_app_token(self):
        """Получает application access token через client_credentials flow"""
        url = "https://api.hh.ru/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": config.HH_CLIENT_ID,
            "client_secret": config.HH_CLIENT_SECRET
        }
        try:
            resp = self.session.post(url, data=payload, timeout=10)
            if resp.status_code == 200:
                token_data = resp.json()
                self._token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)  # обычно около 86400 для приложений, но страховка
                self._token_expires_at = time.time() + expires_in - 30
                self.session.headers.update({"Authorization": f"Bearer {self._token}"})
                logger.info("✅ Токен приложения успешно получен")
            else:
                logger.error(f"❌ Ошибка получения токена: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"❌ Исключение при получении токена: {e}")

    def _ensure_token(self):
        if not self._token or time.time() > self._token_expires_at:
            logger.info("Токен отсутствует или истёк, получаю новый...")
            self._get_app_token()

    # ======================================================================
    def search_vacancies(self, text, area, period_days=30, max_pages=20, per_page=100,
                         industry=None, date_from=None, date_to=None):
        params = {
            'text': text, 'area': area, 'per_page': per_page, 'page': 0,
            'order_by': 'publication_time', 'clusters': False, 'describe_arguments': False,
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
            self.last_response = data
            if not data or 'items' not in data:
                logger.error("Не удалось получить данные")
                break
            items = data['items']
            if not items:
                break
            all_vacancies.extend(items)
            logger.info(f"Страница {page + 1}: получено {len(items)} вакансий (всего найдено: {data.get('found', 0)})")
            if page >= data.get('pages', 0) - 1:
                break
            page += 1
            time.sleep(config.REQUEST_DELAY)
        logger.info(f"Всего загружено {len(all_vacancies)} вакансий")
        return all_vacancies

    def get_vacancy_details(self, vacancy_id):
        url = f"{self.BASE_URL_FULL}vacancies/{vacancy_id}"
        data = self._get(url)
        if data:
            logger.debug(f"Успешно загружены детали вакансии {vacancy_id}")
        else:
            logger.warning(f"Не удалось загрузить детали вакансии {vacancy_id}")
        return data

    def get_vacancy_details_as_object(self, vacancy_id):
        raw = self.get_vacancy_details(vacancy_id)
        if not raw:
            return None
        try:
            return Vacancy.from_api(raw)
        except ValueError as e:
            logger.warning(f"Невалидная вакансия {vacancy_id}: {e}")
            return None

    # -----------------------------------------------------------------------
    def _get(self, url, params=None, retry_count=0, _is_retry_after_401=False):
        if not _is_retry_after_401:
            self._ensure_token()
        try:
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=10)
            elapsed = time.time() - start_time
            logger.debug(f"GET {url} - статус {response.status_code} ({elapsed:.2f}с)")

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 304:
                logger.debug("304 Not Modified")
                return None
            elif response.status_code == 404:
                logger.warning(f"404: {url}")
                return None
            elif response.status_code == 401:
                if not _is_retry_after_401:
                    logger.warning("401, обновляю токен и пробую снова")
                    self._get_app_token()
                    return self._get(url, params, retry_count, _is_retry_after_401=True)
                else:
                    logger.error("Повторный 401 после обновления токена")
                    return None
            elif response.status_code == 403:
                logger.error("403 Forbidden. Проверьте права приложения или IP.")
                return None
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                if retry_count < 3:
                    logger.warning(f"429. Попытка {retry_count+1}/3, жду {retry_after}с")
                    time.sleep(retry_after)
                    return self._get(url, params, retry_count + 1)
                else:
                    logger.error("Превышено число попыток при 429")
                    return None
            else:
                logger.warning(f"Неожиданный статус: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error: {url}")
            return None
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            return None

    def close(self):
        self.session.close()
        logger.info("Сессия закрыта")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()