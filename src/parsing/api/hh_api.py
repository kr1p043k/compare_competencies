"""
HeadHunterAPI - синхронный клиент для работы с API hh.ru
"""

import json
import time
from pathlib import Path
from typing import Any

import requests
import structlog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config
from src.errors import ApiError, RateLimitError, VacancyNotFoundError
from src.models.hh_responses import (
    VacancyDetailResponse,
    parse_response,
)
from src.models.vacancy import Vacancy
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)

class HeadHunterAPI:
    """
    Синхронный API клиент для hh.ru.
    Использует автоматическое получение токена через client_credentials.
    """

    BASE_URL = "https://api.hh.ru/vacancies"
    BASE_URL_FULL = "https://api.hh.ru/"
    TOKEN_CACHE_FILE: Path | None = None

    def __init__(self):
        self.session = requests.Session()

        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1,
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.session.headers.update({"User-Agent": config.HH_USER_AGENT, "Accept": "application/json; charset=utf-8"})

        self.last_response: dict[str, Any] | None = None

        # === АВТОМАТИЧЕСКОЕ ПОЛУЧЕНИЕ ТОКЕНА ===
        self._token = None
        self._token_expires_at = 0
        if self.TOKEN_CACHE_FILE is None:
            self.TOKEN_CACHE_FILE = config.DATA_CACHE_DIR / ".hh_token_cache.json"
        if config.HH_CLIENT_ID and config.HH_CLIENT_SECRET:
            match self._load_cached_token():
                case Ok(False):
                    match self._get_app_token():
                        case Ok(_):
                            pass
                        case Err(e):
                            logger.warning("auto_token_refresh_failed", error=str(e))
                case Err(_):
                    match self._get_app_token():
                        case Ok(_):
                            pass
                        case Err(e):
                            logger.warning("auto_token_refresh_failed", error=str(e))
                case _:
                    pass
        else:
            logger.warning("hh_credentials_not_set")

        logger.info("hh_api_initialized", max_retries=config.MAX_RETRIES)

    def search_vacancies_validated(
        self, text, area, period_days=30, max_pages=20, per_page=100, industry=None, date_from=None, date_to=None
    ) -> Result[list[dict], ApiError]:
        return self.search_vacancies(text, area, period_days, max_pages, per_page, industry, date_from, date_to)

    def get_vacancy_details_validated(self, vacancy_id) -> Result[VacancyDetailResponse, ApiError]:
        """Получает детали вакансии и возвращает валидированную модель."""
        raw = self._get(f"{self.BASE_URL_FULL}vacancies/{vacancy_id}")
        if raw is None:
            return Err(ApiError(message=f"Vacancy {vacancy_id} not found or API error", endpoint=f"vacancies/{vacancy_id}"))
        return Ok(parse_response(raw, VacancyDetailResponse))

    # -----------------------------------------------------------------------
    def _load_cached_token(self) -> Result[bool, ApiError]:
        try:
            cache_path = self.TOKEN_CACHE_FILE
            if not cache_path or not cache_path.exists():
                return Ok(False)
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            token = cache.get("access_token")
            expires_at = cache.get("expires_at", 0)
            if token and time.time() < expires_at:
                self._token = token
                self._token_expires_at = expires_at
                self.session.headers.update({"Authorization": f"Bearer {self._token}"})
                logger.info("cached_token_loaded", expires_in=int(expires_at - time.time()))
                return Ok(True)
            return Ok(False)
        except Exception as e:
            logger.warning("cached_token_load_failed", error=str(e))
            return Err(ApiError(message="Cached token load failed", endpoint="token_cache", detail=str(e)))

    def _save_token_cache(self) -> Result[None, ApiError]:
        try:
            cache = {
                "access_token": self._token,
                "expires_at": self._token_expires_at,
            }
            cache_path = self.TOKEN_CACHE_FILE
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            return Ok(None)
        except Exception as e:
            logger.warning("token_cache_save_failed", error=str(e))
            return Err(ApiError(message="Token cache save failed", endpoint="token_cache", detail=str(e)))

    def _clear_token_cache(self) -> Result[bool, ApiError]:
        try:
            cache_path = self.TOKEN_CACHE_FILE
            if cache_path and cache_path.exists():
                cache_path.unlink()
                logger.info("token_cache_cleared")
                return Ok(True)
            return Ok(False)
        except Exception as e:
            logger.warning("token_cache_clear_failed", error=str(e))
            return Err(ApiError(message="Token cache clear failed", endpoint="token_cache", detail=str(e)))

    def _get_app_token(self, retry_count=0) -> Result[str, ApiError]:
        if retry_count >= 2:
            return Err(ApiError(message="Token retry exhausted", endpoint="token"))
        url = "https://api.hh.ru/token"
        client_id = config.HH_CLIENT_ID.get_secret_value() if config.HH_CLIENT_ID else None
        client_secret = config.HH_CLIENT_SECRET.get_secret_value() if config.HH_CLIENT_SECRET else None
        if not client_id or not client_secret:
            return Err(ApiError(message="HH credentials not set", endpoint="token"))
        payload = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        try:
            resp = self.session.post(url, data=payload, timeout=10)
            if resp.status_code == 200:
                token_data = resp.json()
                self._token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 3600)
                self._token_expires_at = time.time() + expires_in - 60
                self.session.headers.update({"Authorization": f"Bearer {self._token}"})
                match self._save_token_cache():
                    case Err(e):
                        logger.warning("token_cache_save_failed", error=str(e))
                    case _:
                        pass
                logger.info("app_token_obtained", expires_in=expires_in)
                return Ok(self._token)
            elif resp.status_code == 403:
                return Err(ApiError(message="Token refresh too early", status_code=403, endpoint="token"))
            else:
                return Err(ApiError(message=f"Token request failed: {resp.status_code}", status_code=resp.status_code, endpoint="token"))
        except Exception as e:
            return Err(ApiError(message=f"Token request exception: {e}", endpoint="token"))

    def _ensure_token(self):
        if not self._token or time.time() > self._token_expires_at:
            logger.info("token_missing_or_expired")
            match self._load_cached_token():
                case Ok(True):
                    pass
                case _:
                    pass
            if not self._token or time.time() > self._token_expires_at:
                match self._get_app_token():
                    case Ok(_):
                        pass
                    case Err(e):
                        logger.warning("token_refresh_failed", error=str(e))

    # ======================================================================
    def search_vacancies(
        self, text, area, period_days=30, max_pages=20, per_page=100, industry=None, date_from=None, date_to=None,
        since_id: int | None = None,
    ) -> Result[list, ApiError]:
        params = {
            "text": text,
            "area": area,
            "per_page": per_page,
            "page": 0,
            "order_by": "publication_time",
            "clusters": "false",
            "describe_arguments": "false",
        }
        if industry:
            params["industry"] = industry
        if since_id is not None:
            params["vacancy_id_gt"] = since_id
            logger.info("search_vacancies_since_id", since_id=since_id)
        if date_from is not None and date_to is not None:
            params["date_from"] = date_from
            params["date_to"] = date_to
        else:
            params["period"] = period_days

        logger.info("search_vacancies_started", query=text, area=area, max_pages=max_pages)
        all_vacancies = []
        page = 0
        while page < max_pages:
            params["page"] = page
            data = self._get_result(self.BASE_URL, params=params)
            self.last_response = data.ok()
            match data:
                case Ok(d):
                    pass
                case Err(e):
                    return Err(ApiError(message="Failed to get search data", endpoint="vacancies", detail=str(e)))
            if "items" not in d:
                return Err(ApiError(message="Invalid search response: no items", endpoint="vacancies"))
            items = d["items"]
            found = d.get("found", 0)
            if found == 0:
                if items:
                    logger.warning("hh_no_exact_results_similar_queries_returned", query=text, area=area, similar_items=len(items))
                break
            if not items:
                break
            all_vacancies.extend(items)
            logger.info("search_page_loaded", page=page + 1, items=len(items), total_found=d.get("found", 0))
            if page >= d.get("pages", 0) - 1:
                break
            page += 1
            time.sleep(config.REQUEST_DELAY)
        if all_vacancies:
            query_words = [w.lower() for w in text.split() if len(w) > 1]
            if query_words:
                matching = sum(
                    1 for v in all_vacancies if any(w in (v.get("name", "") or "").lower() for w in query_words)
                )
                match_pct = matching / len(all_vacancies) * 100
                if match_pct < 20:
                    logger.warning(
                        "hh_search_returned_similar_queries",
                        query=text,
                        area=area,
                        total=len(all_vacancies),
                        matching_pct=round(match_pct, 1),
                    )

        logger.info("search_vacancies_completed", total=len(all_vacancies))
        return Ok(all_vacancies)

    def get_vacancy_details(self, vacancy_id) -> Result[dict, ApiError]:
        url = f"{self.BASE_URL_FULL}vacancies/{vacancy_id}"
        return self._get_result(url)

    def get_vacancy_details_as_object(self, vacancy_id) -> Result[Vacancy, ApiError]:
        raw_r = self.get_vacancy_details(vacancy_id)
        match raw_r:
            case Ok(raw):
                pass
            case Err(e):
                return Err(e)
        if not raw:
            return Err(ApiError(message=f"Vacancy {vacancy_id} not found", endpoint=f"vacancies/{vacancy_id}"))
        try:
            return Ok(Vacancy.from_api(raw))
        except ValueError as e:
            logger.warning("invalid_vacancy", vacancy_id=vacancy_id, error=str(e))
            return Err(ApiError(message=str(e), endpoint=f"vacancies/{vacancy_id}"))

    # -----------------------------------------------------------------------
    def _get_result(self, url, params=None, retry_count=0, _is_retry_after_401=False) -> Result[dict, ApiError]:
        """GET-запрос с явным Result[T, E] — все ошибки типизированы."""
        if not _is_retry_after_401:
            self._ensure_token()
        try:
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=10)
            elapsed = time.time() - start_time
            logger.debug("http_request", url=url, status=response.status_code, elapsed=round(elapsed, 2))

            if response.status_code == 200:
                return Ok(response.json())
            elif response.status_code == 304:
                logger.debug("http_304_not_modified")
                return Err(ApiError(message="Not modified", status_code=304, endpoint=url))
            elif response.status_code == 404:
                logger.warning("http_404", url=url)
                return Err(VacancyNotFoundError(message=f"Resource not found: {url}", detail=url))
            elif response.status_code == 401:
                if not _is_retry_after_401:
                    logger.warning("http_401_refreshing_token")
                    match self._get_app_token():
                        case Ok(_):
                            pass
                        case Err(e):
                            logger.error("token_refresh_failed", error=str(e))
                    return self._get_result(url, params, retry_count, _is_retry_after_401=True)
                logger.error("http_401_after_token_refresh")
                return Err(ApiError(message="Unauthorized after token refresh", status_code=401, endpoint=url))
            elif response.status_code == 403:
                err_text = response.text[:300]
                if "token-revoked" in err_text or "token_expired" in err_text or "bad_authorization" in err_text:
                    logger.warning("token_invalid_attempting_refresh")
                    self._token = None
                    self._token_expires_at = 0
                    match self._clear_token_cache():
                        case Err(e):
                            logger.warning("token_cache_clear_failed", error=str(e))
                        case _:
                            pass
                    if not _is_retry_after_401:
                        match self._get_app_token():
                            case Ok(_):
                                return self._get_result(url, params, retry_count, _is_retry_after_401=True)
                            case Err(e):
                                logger.error("token_refresh_failed", error=str(e))
                    logger.error("token_refresh_failed")
                    return Err(ApiError(message="Token refresh failed", status_code=403, endpoint=url))
                logger.error("http_403_forbidden", response=err_text)
                return Err(ApiError(message=f"Forbidden: {err_text}", status_code=403, endpoint=url))
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                if retry_count < 3:
                    logger.warning("http_429_rate_limited", attempt=retry_count + 1, retry_after=retry_after)
                    time.sleep(retry_after)
                    return self._get_result(url, params, retry_count + 1)
                logger.error("http_429_max_retries_exceeded")
                return Err(RateLimitError(message="Max retries exceeded for 429", status_code=429, endpoint=url, retry_after=retry_after))
            else:
                logger.warning("http_unexpected_status", status=response.status_code, response=response.text[:500])
                return Err(ApiError(message=f"Unexpected status {response.status_code}", status_code=response.status_code, endpoint=url))
        except requests.exceptions.Timeout:
            logger.error("http_timeout", url=url)
            return Err(ApiError(message=f"Timeout: {url}", endpoint=url))
        except requests.exceptions.ConnectionError:
            logger.error("http_connection_error", url=url)
            return Err(ApiError(message=f"Connection error: {url}", endpoint=url))
        except Exception as e:
            logger.error("http_request_exception", error=str(e))
            return Err(ApiError(message=str(e), endpoint=url))

    def _get(self, url, params=None, retry_count=0, _is_retry_after_401=False):
        """Legacy: обёртка над _get_result для обратной совместимости. Устарела, используйте _get_result."""
        return self._get_result(url, params, retry_count, _is_retry_after_401).ok()

    def close(self):
        self.session.close()
        logger.info("session_closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
