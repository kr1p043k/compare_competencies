"""Клиент API rabota.ru (Работа.ру) с приведением ответов к формату hh.ru.

Зачем адаптер формата: весь пайплайн (Vacancy.from_api, VacancyParser,
db_writer, аналитика) ждёт hh-структуру вакансии (id/name/area/employer/
key_skills/snippet/salary/experience/published_at). Поэтому клиент Работы.ру
не просто качает JSON, а сразу отдаёт hh-совместимый dict. Благодаря этому
второй DataSource не нужен: см. src/parsing/api/aggregator.py.

Особенности протокола Работы.ру (проверено вживую на v5, 2026-11):
- POST на /outer_clients/vacancies/search.json, тело
  {application_id, time, request, signature}; signature =
  sha256(канонический JSON параметров + secret).
- Пагинация offset/limit, limit жёстко режется сервером до 100.
- Глубина выдачи ограничена (offset больше ~4000 отдаёт пусто).
- Отдельного эндпоинта "вакансия по id" нет: поиск сразу возвращает полный
  description и skills, поэтому детали берём из кэша, заполненного поиском.
- WAF (BI.ZONE) отбивает короткие User-Agent: нужен полноценный браузерный UA.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import requests
import structlog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config
from src.errors import ApiError, VacancyNotFoundError
from src.models.hh_responses import VacancyDetailResponse, parse_response
from src.models.vacancy import Vacancy
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)

SEARCH_PATH = "/outer_clients/vacancies/search.json"
GEOCODER_PATH = "/geocoder/alternative.json"

# Поля, которые просим у поиска: сразу полный набор, чтобы детали не догружать.
SEARCH_FIELDS = [
    "vacancies.id",
    "vacancies.title",
    "vacancies.salary",
    "vacancies.description",
    "vacancies.short_description",
    "vacancies.places",
    "vacancies.company",
    "vacancies.skills",
    "vacancies.experience",
    "vacancies.publish_start_at",
    "vacancies.modified_date",
    "vacancies.employment_id",
    "vacancies.work_schedule_id",
    "vacancies.education",
    "vacancies.professional_areas",
    "vacancies.top_position",
    "vacancies.tags",
    "vacancies.places_summary",
]

# hh-коды опыта, которые понимает src/utils.extract_experience и ml/clusters.
_RABOTA_EXPERIENCE_TO_HH: dict[int, tuple[str, str]] = {
    1: ("noExperience", "Нет опыта"),
    2: ("lessThan1Year", "Менее года"),
    3: ("between1And3", "От 1 года"),
    4: ("between1And3", "От 2 лет"),
    5: ("between3And6", "От 3 лет"),
    6: ("between3And6", "От 4 лет"),
    7: ("between3And6", "От 5 лет"),
    8: ("moreThan6", "Более 10 лет"),
}

# hh area_id -> region_id Работы.ру. Пары сверены вживую: имена из
# https://api.hh.ru/areas/<id> и регионы Работы.ру из их геокодера/поиска.
# Незнакомый id НЕЛЬЗЯ молча считать регионом Работы.ру: нумерации разные
# (hh 3 = Екатеринбург, rabota 3 = Москва), поэтому map_area предупреждает.
HH_AREA_TO_RABOTA_REGION: dict[int, int] = {
    1: 3,      # Москва
    2: 4,      # Санкт-Петербург
    3: 328,    # Екатеринбург
    54: 25,    # Красноярск
    66: 231,   # Нижний Новгород
    76: 284,   # Ростов-на-Дону
    78: 299,   # Самара
    88: 481,   # Казань
    99: 529,   # Уфа
}

# Пары для get_areas(): hh_id -> (rabota_region_id, имя)
_HH_AREA_NAMES: dict[int, tuple[int, str]] = {
    1: (3, "Москва"),
    2: (4, "Санкт-Петербург"),
    3: (328, "Екатеринбург"),
    54: (25, "Красноярск"),
    66: (231, "Нижний Новгород"),
    76: (284, "Ростов-на-Дону"),
    78: (299, "Самара"),
    88: (481, "Казань"),
    99: (529, "Уфа"),
}

_CURRENCY_TO_CODE = {
    "руб./мес.": "RUB",
    "руб.": "RUB",
    "₽": "RUB",
    "р.": "RUB",
    "usd": "USD",
    "$": "USD",
    "eur": "EUR",
    "€": "EUR",
    "kzt": "KZT",
    "₸": "KZT",
    "byn": "BYN",
    "uah": "UAH",
    "грн": "UAH",
}

# --- IT-фильтр --------------------------------------------------------------
# Работа.ру игнорирует professional_role/industry, поэтому «python» отдаёт и
# педагогов. Основной признак - дерево professional_areas. Внимание: корень
# 01.01 - это весь «IT / Интернет / Телеком», но внутри него лежат и НЕ-IT
# группы (01.01.06 продажи, 01.01.07 SMM/SEO, 01.01.08 контент,
# 01.01.09 call-центр, 01.01.11 другое), поэтому берём только IT-подветки.
IT_AREA_TREE_PREFIXES = (
    "01.01.01.",  # Программирование, разработка (программист, тестировщик)
    "01.01.02.",  # Виды ПО (1С, ERP, банковское, игровое)
    "01.01.03.",  # Аналитика (бизнес/системный аналитик, техписатель)
    "01.01.04.",  # Сети, защита информации (сисадмин/DBA, ИБ)
)

# Слова в названии вакансии/профессии. Регексы с \b (в Python 3 \b работает
# для кириллицы), чтобы «qa» не ловилось внутри «Qatar», а «1с» - внутри «1сентября».
_IT_TITLE_PATTERNS = [
    r"\bpython\b", r"\bпитон\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\b(?:node|vue|react|angular)\.?js\b", r"\bphp\b", r"\bc\+\+\b", r"\bc#\b",
    r"\bgolang\b", r"\brust\b", r"\bscala\b", r"\bkotlin\b", r"\bswift\b",
    r"\b1с\b", r"\b1c\b", r"\bdevops\b", r"\bdev-ops\b", r"\bsre\b",
    r"\bqa\b", r"\bтестировщик\b", r"\bтестирование\b", r"\bавтотест\w*",
    r"\bпрограммист\w*", r"\bразработчик\w*", r"\bdeveloper\b", r"\bengineer\b",
    r"\bbackend\b", r"\bfrontend\b", r"\bfull[-\s]?stack\b", r"\bверстальщик\w*",
    r"\bсистемный аналитик\w*", r"\bбизнес-аналитик\w*", r"\bаналитик данных\b",
    r"\bdata\s?scientist\b", r"\bdata\s?engineer\b", r"\bdata\s?analyst\b",
    r"\bml\s?(?:engineer|инженер|разработчик)\w*", r"\bмашинн\w+ обучени\w+",
    r"\bсистемн\w+ администратор\w*", r"\bсетев\w+ инженер\w*", r"\bсисадмин\w*",
    r"\bинженер\w*\s+по\s+сет\w*", r"\bинформационн\w+ безопасност\w*",
    r"\bпентест\w*", r"\bбитрикс\w*", r"\bархитектор\s+по\b",
    r"\bteamlead\b", r"\bteam\s?lead\b", r"\bтимлид\w*",
]
_IT_TITLE_RE = re.compile("|".join(_IT_TITLE_PATTERNS), re.IGNORECASE)

# Образовательные роли в IT-ветке (Работа.ру метит их как 01.01.01): для
# анализа рынка навыков это шум, hh их тоже не отдаёт в IT-ролях.
_NON_IT_TITLE_RE = re.compile(
    r"\b(?:преподаватель\w*|учитель\w*|педагог\w*|репетитор\w*|тьютор\w*|методист\w*)\b",
    re.IGNORECASE,
)


def _is_it_vacancy(raw: dict[str, Any]) -> bool:
    """True, если вакансия относится к IT (по дереву областей или по названию)."""
    title = str(raw.get("title") or "")
    top = raw.get("top_position")
    if isinstance(top, dict):
        title = f"{title} {top.get('name') or ''}"
    if _NON_IT_TITLE_RE.search(title):
        return False
    for area in raw.get("professional_areas") or []:
        tree = str(area.get("tree") or "") if isinstance(area, dict) else ""
        if tree.startswith(IT_AREA_TREE_PREFIXES):
            return True
    return bool(_IT_TITLE_RE.search(title))


def _canonical_json(value: Any) -> str:
    """Канонизация как в доке Работы.ру (PHP: ksort + json_encode).

    Ключи сортируются рекурсивно, JSON без пробелов. ensure_ascii=True
    повторяет поведение PHP json_encode по умолчанию (не-ASCII в \\uXXXX).
    """
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(str(k), ensure_ascii=True)
            + ":"
            + _canonical_json(value[k])
            for k in sorted(value, key=lambda x: str(x))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_json(v) for v in value) + "]"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def compute_signature(params: dict[str, Any], secret: str) -> str:
    """sha256(канонический JSON параметров + secret) по схеме из dev.rabota.ru."""
    payload = _canonical_json(params) + secret
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _currency_code(raw: Any) -> str:
    if not raw:
        return "RUB"
    text = str(raw).strip()
    return _CURRENCY_TO_CODE.get(text.lower(), text[:10] if text.isascii() else "RUB")


def _plain_text(html: str | None) -> str | None:
    """Грубая очистка HTML для snippet (в hh это уже plain text)."""
    if not html:
        return None
    import re

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def rabota_to_hh(raw: dict[str, Any], *, id_offset: int = 0) -> dict[str, Any]:
    """Преобразует вакансию Работы.ру в hh-совместимый dict.

    id сдвигается на id_offset, чтобы не пересечься с hh_id в таблице
    vacancies (там UNIQUE(hh_id)) и в дедупликации по id.
    """
    if not isinstance(raw, dict):
        raise ValueError("rabota vacancy must be a dict")

    rabota_id = raw.get("id")
    if rabota_id is None:
        raise ValueError("rabota vacancy has no id")
    title = (raw.get("title") or "").strip()
    if not title:
        raise ValueError("rabota vacancy has no title")

    vacancy_id = int(rabota_id) + id_offset

    places = raw.get("places") or []
    first_place = places[0] if places and isinstance(places[0], dict) else {}
    region = first_place.get("region") or {}
    area = {
        "id": region.get("id") or 0,
        "name": region.get("name") or first_place.get("name") or "Не указан",
    }

    company = raw.get("company") or {}
    logo = company.get("logo") or {}
    employer = {
        "id": str(company.get("id")) if company.get("id") is not None else "",
        "name": company.get("name") or "Не указан",
        "url": None,
        "logo_urls": {"original": logo.get("url_original")} if logo.get("url_original") else None,
        "trusted": bool(company.get("is_verified")),
    }

    key_skills = []
    for skill in raw.get("skills") or []:
        name = (skill.get("name") or "").strip() if isinstance(skill, dict) else str(skill).strip()
        if name:
            key_skills.append({"name": name, "id": str(skill.get("id")) if isinstance(skill, dict) else None})

    salary_raw = raw.get("salary") or {}
    salary = None
    if salary_raw:
        from_amount = salary_raw.get("from") or None
        to_amount = salary_raw.get("to") or None
        if from_amount or to_amount:
            salary = {
                "from": from_amount,
                "to": to_amount,
                "currency": _currency_code(salary_raw.get("currency")),
                "gross": str(salary_raw.get("pay_type") or "").lower() == "gross",
            }

    experience = None
    exp_raw = raw.get("experience")
    if isinstance(exp_raw, dict):
        exp_id = exp_raw.get("id")
        hh_code, hh_name = _RABOTA_EXPERIENCE_TO_HH.get(
            int(exp_id) if isinstance(exp_id, (int, str)) and str(exp_id).isdigit() else -1,
            ("", ""),
        )
        name = exp_raw.get("name") or hh_name
        if name:
            experience = {"id": hh_code or name, "name": name}

    description = raw.get("description") or None
    snippet = None
    short = raw.get("short_description")
    if short or description:
        snippet = {"requirement": _plain_text(short), "responsibility": None}

    professional_roles = []
    for pa in raw.get("professional_areas") or []:
        if isinstance(pa, dict) and pa.get("name"):
            professional_roles.append({
                "id": str(pa.get("id")),
                "name": pa.get("name"),
                "tree": pa.get("tree"),
            })

    published = raw.get("publish_start_at") or raw.get("modified_date")

    return {
        "id": str(vacancy_id),
        "name": title,
        "area": area,
        "employer": employer,
        "salary": salary,
        "snippet": snippet,
        "experience": experience,
        "description": description,
        "key_skills": key_skills,
        "published_at": published,
        "alternate_url": f"https://www.rabota.ru/vacancy/{rabota_id}/",
        "professional_roles": professional_roles,
        "archived": False,
        "source": "rabota",
        "rabota_id": int(rabota_id),
    }


class RabotaAPI:
    """Синхронный клиент публичного поиска вакансий rabota.ru.

    Интерфейс совместим с HeadHunterAPI в той части, которую использует
    пайплайн (search_vacancies / get_vacancy_details* / last_response).
    """

    def __init__(
        self,
        app_id: int | None = None,
        secret: str | None = None,
        base_url: str | None = None,
        user_agent: str | None = None,
        id_offset: int | None = None,
        request_delay: float | None = None,
        timeout: float = 45.0,
    ):
        app_id = app_id if app_id is not None else config.RABOTA_APP_ID
        if secret:
            self._secret = secret
        elif config.RABOTA_SECRET is not None:
            self._secret = config.RABOTA_SECRET.get_secret_value()
        else:
            self._secret = None
        self._app_id = app_id
        self._base_url = (base_url or config.RABOTA_BASE_URL).rstrip("/")
        self._id_offset = id_offset if id_offset is not None else getattr(config, "RABOTA_ID_OFFSET", 0)
        self._request_delay = request_delay if request_delay is not None else max(config.REQUEST_DELAY, 0.5)
        self._timeout = timeout

        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1.2,
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": user_agent or config.RABOTA_USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        })

        # Совместимость с helpers.load_vacancies_details (асинхронная ветка
        # берёт токен hh-клиента). У Работы.ру токена нет.
        self._token: str | None = None
        self._token_expires_at: float = 0
        self.last_response: dict[str, Any] | None = None
        # id -> hh-совместимый dict (поиск отдаёт полные данные, детали не нужны)
        self._details_cache: dict[str, dict[str, Any]] = {}

        if not self._app_id or not self._secret:
            logger.warning("rabota_credentials_missing", app_id=self._app_id)

        logger.info(
            "rabota_api_initialized",
            base_url=self._base_url,
            app_id=self._app_id,
            authenticated=bool(self._app_id and self._secret),
        )

    # ------------------------------------------------------------------ auth
    @property
    def enabled(self) -> bool:
        return bool(self._app_id and self._secret)

    def _sign(self, params: dict[str, Any]) -> str:
        return compute_signature(params, self._secret or "")

    def _post(self, path: str, request: dict[str, Any], *, endpoint: str) -> Result[dict[str, Any], ApiError]:
        if not self.enabled:
            return Err(ApiError(message="Rabota.ru credentials not configured", endpoint=endpoint))

        envelope: dict[str, Any] = {
            "application_id": int(self._app_id),
            "time": int(time.time()),
            "request": request,
        }
        envelope["signature"] = self._sign(envelope)
        try:
            resp = self.session.post(
                f"{self._base_url}{path}",
                data=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            logger.error("rabota_http_timeout", endpoint=endpoint)
            return Err(ApiError(message=f"Timeout: {endpoint}", endpoint=endpoint))
        except requests.exceptions.ConnectionError as exc:
            logger.error("rabota_http_connection_error", endpoint=endpoint)
            return Err(ApiError(message=f"Connection error: {endpoint}", detail=str(exc), endpoint=endpoint))
        except Exception as exc:
            logger.error("rabota_http_exception", endpoint=endpoint, error=str(exc))
            return Err(ApiError(message=str(exc), endpoint=endpoint))

        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "json" in content_type:
            return Ok(resp.json())
        if resp.status_code == 200:
            # 200 + HTML = страница WAF, а не данные.
            return Err(ApiError(
                message="Rabota.ru returned HTML instead of JSON (WAF/User-Agent?)",
                status_code=resp.status_code,
                endpoint=endpoint,
            ))
        body = resp.text[:300]
        if resp.status_code >= 500:
            logger.warning("rabota_http_server_error", status=resp.status_code, endpoint=endpoint)
            return Err(ApiError(
                message=f"Rabota.ru server error {resp.status_code}",
                status_code=resp.status_code,
                endpoint=endpoint,
                detail=body,
            ))
        logger.warning("rabota_http_error", status=resp.status_code, endpoint=endpoint, body=body)
        return Err(ApiError(
            message=f"Rabota.ru HTTP {resp.status_code}: {body}",
            status_code=resp.status_code,
            endpoint=endpoint,
            detail=body,
        ))

    # --------------------------------------------------------------- helpers
    @staticmethod
    def map_area(area: int | None) -> int:
        """hh area_id -> region_id Работы.ру.

        Нумерации разные (hh 3 = Екатеринбург, rabota 3 = Москва), поэтому
        неизвестный id нельзя передавать как есть: это молча дало бы чужой
        город. Переводим только сверенные пары, остальное - регион по
        умолчанию с предупреждением (передавайте id Работы.ру напрямую).
        """
        if area is None:
            return config.RABOTA_DEFAULT_REGION_ID
        if area in HH_AREA_TO_RABOTA_REGION:
            return HH_AREA_TO_RABOTA_REGION[area]
        if area in config.RABOTA_REGION_IDS:
            return int(area)
        logger.warning(
            "rabota_area_not_mapped",
            area=area,
            fallback=config.RABOTA_DEFAULT_REGION_ID,
            hint="используйте id региона Работы.ру (RABOTA_REGION_IDS)",
        )
        return config.RABOTA_DEFAULT_REGION_ID

    # ---------------------------------------------------------------- search
    def search_vacancies(
        self,
        text: str,
        area: int | None = None,
        period_days: int = 30,
        max_pages: int = 20,
        per_page: int = 100,
        industry: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        since_id: int | None = None,
        professional_role: str | list[str] | None = None,
        it_only: bool | None = None,
        **kwargs: Any,
    ) -> Result[list[dict[str, Any]], ApiError]:
        per_page = max(1, min(int(per_page or 100), int(config.RABOTA_PER_PAGE), 100))
        region_id = self.map_area(area)
        max_offset = int(config.RABOTA_MAX_OFFSET)
        max_pages = max(1, int(max_pages or 1))
        it_only = config.RABOTA_IT_ONLY if it_only is None else it_only

        base_request: dict[str, Any] = {
            "fields": SEARCH_FIELDS,
            "query": text or "",
            "limit": per_page,
            "offset": 0,
            "location": {"type": "region", "region_id": region_id},
            "sort": {"field": "date", "direction": "desc"},
        }
        filters: dict[str, Any] = {}
        ts = self._period_start_ts(period_days, date_from)
        if ts is not None:
            filters["from_timestamp"] = ts
        if filters:
            base_request["filters"] = filters
        if industry is not None or professional_role is not None or since_id is not None:
            logger.debug(
                "rabota_search_unsupported_params_ignored",
                industry=industry,
                professional_role=professional_role,
                since_id=since_id,
            )

        logger.info("rabota_search_started", query=text, region_id=region_id, max_pages=max_pages)
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        offset = 0
        page = 0
        total_available: int | None = None
        filtered_out = 0

        while page < max_pages and offset < max_offset:
            request = dict(base_request)
            request["offset"] = offset
            match self._post(SEARCH_PATH, request, endpoint=SEARCH_PATH):
                case Ok(data):
                    self.last_response = data
                case Err(e):
                    if collected:
                        logger.warning("rabota_search_stopped_on_error", error=e.message, collected=len(collected))
                        break
                    return Err(e)
            response = data.get("response") or {}
            vacancies = response.get("vacancies") or []
            if total_available is None:
                total_available = response.get("total")
            if not vacancies:
                break
            for raw in vacancies:
                # IT-фильтр: API не умеет ограничивать по профессии, режем сами.
                if it_only and not _is_it_vacancy(raw):
                    filtered_out += 1
                    continue
                try:
                    mapped = rabota_to_hh(raw, id_offset=self._id_offset)
                except ValueError as exc:
                    logger.debug("rabota_vacancy_skipped", error=str(exc))
                    continue
                vid = mapped["id"]
                if vid in seen:
                    continue
                seen.add(vid)
                self._details_cache[vid] = mapped
                collected.append(mapped)
            logger.info(
                "rabota_search_page_loaded",
                page=page + 1,
                offset=offset,
                items=len(vacancies),
                it_only=it_only,
                filtered_out=filtered_out,
                total=total_available,
            )
            if len(vacancies) < per_page:
                break
            offset += per_page
            page += 1
            self._sleep()

        if total_available is not None:
            self.last_response = {
                **(self.last_response or {}),
                "found": total_available,
                "pages": max(1, (int(total_available) + per_page - 1) // per_page),
                "page": page,
                "per_page": per_page,
                "filtered_out": filtered_out,
            }
        logger.info(
            "rabota_search_completed",
            total=len(collected),
            filtered_out=filtered_out,
            it_only=it_only,
            query=text,
        )
        return Ok(collected)

    def search_vacancies_validated(self, *args, **kwargs) -> Result[list[dict[str, Any]], ApiError]:
        return self.search_vacancies(*args, **kwargs)

    # --------------------------------------------------------------- details
    def get_vacancy_details(self, vacancy_id: str | int) -> Result[dict[str, Any], ApiError]:
        key = str(vacancy_id)
        cached = self._details_cache.get(key)
        if cached is not None:
            return Ok(cached)
        # У Работы.ру нет GET по id: если вакансии нет в кэше поиска, её
        # нельзя получить, пока не выполнен поиск, который её вернёт.
        logger.warning("rabota_detail_not_cached", vacancy_id=key)
        return Err(VacancyNotFoundError(
            message=f"Rabota.ru vacancy {key} not in search cache",
            detail=f"vacancy/{key}",
            vacancy_id=key,
        ))

    def get_vacancy_details_validated(self, vacancy_id: str | int) -> Result[VacancyDetailResponse, ApiError]:
        match self.get_vacancy_details(vacancy_id):
            case Ok(raw):
                return Ok(parse_response(raw, VacancyDetailResponse))
            case Err(e):
                return Err(ApiError(
                    message=f"Vacancy {vacancy_id} not found or API error",
                    endpoint=f"vacancies/{vacancy_id}",
                    detail=str(e),
                ))

    def get_vacancy_details_as_object(self, vacancy_id: str | int) -> Result[Vacancy, ApiError]:
        match self.get_vacancy_details(vacancy_id):
            case Ok(raw):
                try:
                    return Ok(Vacancy.from_api(raw))
                except ValueError as exc:
                    return Err(ApiError(message=str(exc), endpoint=f"vacancies/{vacancy_id}"))
            case Err(e):
                return Err(e)

    def get_areas(self) -> Result[list[dict[str, Any]], ApiError]:
        """Справки регионов у v5 нет; отдаём известные сопоставления hh->rabota."""
        return Ok([
            {"id": hh_id, "name": name, "rabota_region_id": region_id}
            for hh_id, (region_id, name) in _HH_AREA_NAMES.items()
        ])

    def resolve_region(self, address: str) -> Result[int, ApiError]:
        """Город/адрес -> region_id через их геокодер (для CLI/маппинга)."""
        match self._post(GEOCODER_PATH, {"address": address}, endpoint=GEOCODER_PATH):
            case Ok(data):
                region_id = (data.get("response") or {}).get("vacancy_address_region_id")
                if region_id is None:
                    return Err(ApiError(message=f"region not resolved: {address}", endpoint=GEOCODER_PATH))
                return Ok(int(region_id))
            case Err(e):
                return Err(e)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _period_start_ts(period_days: int, date_from: str | None) -> int | None:
        from datetime import UTC, datetime

        if date_from:
            try:
                parsed = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=UTC)
                return int(parsed.timestamp())
            except ValueError:
                logger.warning("rabota_bad_date_from", date_from=date_from)
        if period_days and period_days > 0:
            return int(time.time()) - int(period_days) * 86400
        return None

    def _sleep(self) -> None:
        if self._request_delay > 0:
            time.sleep(self._request_delay)

    def close(self) -> None:
        self.session.close()
        logger.info("rabota_session_closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
