"""Единый вход к источникам вакансий (hh.ru + rabota.ru).

Пайплайну не нужно знать, откуда вакансия: клиенты отдают hh-совместимый
формат, а `get_vacancy_feed()` возвращает синглтон, который маршрутизирует
запросы к включённым источникам и склеивает результаты.

Источник выбирается `config.VACANCY_SOURCE`:
- "hh"     - только hh.ru (поведение по умолчанию, без изменений);
- "rabota" - только Работа.ру;
- "both"   - оба, с дедупликацией по (название, работодатель, регион);
- "auto"   - всё, для чего есть креды (rabota без ключей пропускается).

Так второй DataSource не нужен: `HhDataSource` берёт клиент из
`get_vacancy_feed()` и работает с ним как с обычным hh-клиентом.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog

from src import config
from src.errors import ApiError
from src.result import Err, Ok, Result

logger = structlog.get_logger(__name__)

VALID_SOURCES = ("hh", "rabota", "both", "auto")


class VacancyFeed:
    """hh-совместимый фасад над одним или несколькими источниками вакансий."""

    def __init__(self, source: str | None = None):
        self.source = (source or getattr(config, "VACANCY_SOURCE", "hh") or "hh").lower()
        if self.source not in VALID_SOURCES:
            logger.warning("vacancy_feed_unknown_source", source=self.source, fallback="hh")
            self.source = "hh"

        self._hh = None
        self._rabota = None
        self.last_response: dict[str, Any] | None = None
        # Передаётся make_vacancy_client: hh-фабрика вызывающего модуля,
        # чтобы подмены HeadHunterAPI в тестах продолжали действовать.
        self._hh_factory = None

    # ------------------------------------------------------------- сборка
    def _hh_client(self):
        if self._hh is None:
            if self._hh_factory is not None:
                self._hh = self._hh_factory()
            else:
                # Ленивый импорт: тесты подменяют src.parsing.api.hh_api.HeadHunterAPI,
                # и фабрика должна видеть подмену в момент первого обращения.
                from src.parsing.api.hh_api import HeadHunterAPI

                self._hh = HeadHunterAPI()
        return self._hh

    def _rabota_client(self):
        if self._rabota is None:
            from src.parsing.api.rabota_api import RabotaAPI

            self._rabota = RabotaAPI()
        return self._rabota

    def _rabota_ready(self) -> bool:
        try:
            return self._rabota_client().enabled
        except Exception as exc:  # noqa: BLE001 - отсутствие кредов не должно ронять пайплайн
            logger.warning("rabota_init_failed", error=str(exc))
            return False

    def enabled_sources(self) -> list[str]:
        if self.source == "hh":
            return ["hh"]
        if self.source == "rabota":
            return ["rabota"] if self._rabota_ready() else []
        if self.source == "auto":
            out = ["hh"]
            if self._rabota_ready():
                out.append("rabota")
            return out
        # both
        out = ["hh"]
        if self._rabota_ready():
            out.append("rabota")
        else:
            logger.warning("rabota_disabled_no_credentials")
        return out

    # ---------------------------------------------------- совместимость API
    @property
    def supports_async_details(self) -> bool:
        """Асинхронная догрузка деталей умеет только hh (у Работы.ру её нет)."""
        return "hh" in self.enabled_sources()

    @property
    def _token(self) -> str | None:
        if "hh" in self.enabled_sources():
            return getattr(self._hh_client(), "_token", None)
        return None

    @property
    def _token_expires_at(self) -> float:
        if "hh" in self.enabled_sources():
            return getattr(self._hh_client(), "_token_expires_at", 0)
        return 0

    # ---------------------------------------------------------------- поиск
    def search_vacancies(self, *args: Any, **kwargs: Any) -> Result[list[dict[str, Any]], ApiError]:
        sources = self.enabled_sources()
        if not sources:
            return Err(ApiError(message="Нет доступных источников вакансий (проверьте креды)"))

        collected: list[dict[str, Any]] = []
        errors: list[str] = []
        found_total = 0

        for name in sources:
            client = self._hh_client() if name == "hh" else self._rabota_client()
            result = client.search_vacancies(*args, **kwargs)
            match result:
                case Ok(vacancies):
                    collected.extend(vacancies)
                    resp = getattr(client, "last_response", None) or {}
                    found_total += int(resp.get("found") or 0)
                case Err(e):
                    errors.append(f"{name}: {e.message}")
                    logger.warning("vacancy_feed_source_failed", source=name, error=e.message)

        if not collected and errors:
            return Err(ApiError(message="Все источники недоступны: " + "; ".join(errors)))

        merged = self._dedupe(collected) if len(sources) > 1 else collected
        if len(sources) > 1:
            logger.info(
                "vacancy_feed_merged",
                sources=sources,
                raw=len(collected),
                unique=len(merged),
            )
        self.last_response = {"found": found_total, "pages": None, "source": "+".join(sources)}
        return Ok(merged)

    def search_vacancies_validated(self, *args: Any, **kwargs: Any) -> Result[list[dict[str, Any]], ApiError]:
        return self.search_vacancies(*args, **kwargs)

    @staticmethod
    def _dedupe(vacancies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Убирает межсайтовые дубли: один и тот же оффер публикуется на обоих."""
        seen_ids: set[str] = set()
        seen_fingerprints: set[tuple[str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for vac in vacancies:
            vid = str(vac.get("id"))
            if vid in seen_ids:
                continue
            employer = (vac.get("employer") or {}).get("name") or ""
            area = (vac.get("area") or {}).get("name") or ""
            fingerprint = (
                " ".join((vac.get("name") or "").lower().split()),
                " ".join(str(employer).lower().split()),
                " ".join(str(area).lower().split()),
            )
            if all(fingerprint) and fingerprint in seen_fingerprints:
                continue
            seen_ids.add(vid)
            if all(fingerprint):
                seen_fingerprints.add(fingerprint)
            out.append(vac)
        return out

    # -------------------------------------------------------------- детали
    def _client_for_id(self, vacancy_id: str | int):
        try:
            numeric = int(vacancy_id)
        except (TypeError, ValueError):
            numeric = None
        offset = int(getattr(config, "RABOTA_ID_OFFSET", 0))
        if numeric is not None and offset and numeric >= offset and self._rabota_ready():
            return self._rabota_client()
        return self._hh_client()

    def get_vacancy_details(self, vacancy_id: str | int) -> Result[dict[str, Any], ApiError]:
        return self._client_for_id(vacancy_id).get_vacancy_details(vacancy_id)

    def get_vacancy_details_validated(self, vacancy_id: str | int):
        return self._client_for_id(vacancy_id).get_vacancy_details_validated(vacancy_id)

    def get_vacancy_details_as_object(self, vacancy_id: str | int):
        return self._client_for_id(vacancy_id).get_vacancy_details_as_object(vacancy_id)

    def get_areas(self) -> Result[list[dict[str, Any]], ApiError]:
        if "hh" in self.enabled_sources():
            return self._hh_client().get_areas()
        return self._rabota_client().get_areas()

    def close(self) -> None:
        for client in (self._hh, self._rabota):
            if client is not None and hasattr(client, "close"):
                client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


_feed_lock = threading.Lock()
_feed_instance: VacancyFeed | None = None


def get_vacancy_feed(source: str | None = None) -> VacancyFeed:
    """Синглтон-вход к источникам вакансий.

    source=None -> config.VACANCY_SOURCE. Явный source создаёт/пересоздаёт
    экземпляр (нужно CLI-флагу --source и тестам).
    """
    global _feed_instance
    with _feed_lock:
        if source is not None:
            if _feed_instance is None or _feed_instance.source != source.lower():
                _feed_instance = VacancyFeed(source)
            return _feed_instance
        if _feed_instance is None:
            _feed_instance = VacancyFeed()
        return _feed_instance


def make_vacancy_client(hh_factory, source: str | None = None):
    """Клиент для пайплайна: hh напрямую или синглтон-агрегатор.

    hh_factory передаёт вызывающий модуль, чтобы сохранить monkeypatch-подмены
    `HeadHunterAPI` в этом модуле (тесты патчат модульную ссылку, а не
    src.parsing.api.hh_api).
    """
    resolved = (source or getattr(config, "VACANCY_SOURCE", "hh") or "hh").lower()
    if resolved in ("", "hh"):
        return hh_factory()
    feed = get_vacancy_feed(resolved)
    feed._hh_factory = hh_factory
    return feed


def reset_vacancy_feed() -> None:
    """Сбрасывает синглтон (тесты, смена конфигурации)."""
    global _feed_instance
    with _feed_lock:
        if _feed_instance is not None:
            _feed_instance.close()
        _feed_instance = None
