"""Клиент Course Trends API ЮФУ (academic-api-dev).

Новый сервис заменил старый Academic Gap-Analysis API. Отличие: единица
работы - программа (`pid`), а рекомендации предрассчитаны заранее (без LLM)
с учётом контекста ОПОП и РПД. Статьи/тренды приходят по курсам программы.

Авторизация: тот же hub-JWT, что выдаёт hub.sfedu.ru после входа
(`Authorization: Bearer <jwt>`). Без токена сервис отвечает 401.

WAF отдаёт 403 на User-Agent python-httpx, поэтому UA маскируется под Chrome.

ВАЖНО: ответы сервиса описаны в OpenAPI без моделей (`"schema": {}`), поэтому
клиент возвращает сырой JSON и НЕ навязывает структуру. Разбор полей - дело
вызывающего кода, чтобы не терять данные при изменении сервиса на стороне ЮФУ.
"""

from __future__ import annotations

import httpx
import structlog

from src import config

logger = structlog.get_logger(__name__)

# WAF режет python-httpx (403); UA должен выглядеть браузерным.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

DEFAULT_TIMEOUT = 60.0


class CourseTrendsError(RuntimeError):
    """Сервис ЮФУ недоступен или вернул ошибку."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


class CourseTrendsClient:
    """Тонкая обёртка над Course Trends API.

    Все методы возвращают распарсенный JSON как есть (dict/list): сервис не
    публикует схемы ответов, поэтому любые предположения о полях были бы
    догадкой. Ошибки транспорта и статусы != 200 поднимаются как
    CourseTrendsError, чтобы вызывающий код мог отличить "нет данных" от
    "сервис лежит".
    """

    def __init__(self, token: str, base_url: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        if not token:
            raise CourseTrendsError("Не задан hub-JWT для сервиса ЮФУ", 401)
        self.token = token
        self.base_url = (
            base_url or config.ACADEMIC_TRENDS_API_BASE
        ).rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> object:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout, headers=_HEADERS) as client:
                resp = client.get(url, headers={"Authorization": f"Bearer {self.token}"})
        except httpx.HTTPError as exc:
            logger.warning("course_trends_unreachable", path=path, error=str(exc))
            raise CourseTrendsError(f"Сервис ЮФУ недоступен: {exc}") from None

        if resp.status_code == 401:
            raise CourseTrendsError(
                "Доступ к сервису ЮФУ истёк (токен действует 1 час)", 401
            )
        if resp.status_code != 200:
            logger.warning("course_trends_error", path=path, status=resp.status_code,
                           body=resp.text[:300])
            raise CourseTrendsError(
                f"Ошибка сервиса ЮФУ ({resp.status_code}): {resp.text[:300]}",
                resp.status_code,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise CourseTrendsError(f"Сервис ЮФУ вернул не-JSON: {exc}") from None

    # ── методы API ────────────────────────────────────────────────────────

    def list_programs(self) -> object:
        """`/programs` — список программ с метками (n_courses, avg_coverage)."""
        return self._get("/programs")

    def program_bundle(self, pid: str) -> object:
        """`/programs/{pid}` — всё по программе одним ответом."""
        return self._get(f"/programs/{pid}")

    def program_summary(self, pid: str) -> object:
        """`/programs/{pid}/summary` — summary + appendix (виджет)."""
        return self._get(f"/programs/{pid}/summary")

    def course_trends(self, pid: str) -> object:
        """`/programs/{pid}/course-trends` — тренды/статьи по курсам."""
        return self._get(f"/programs/{pid}/course-trends")

    def course_coverage(self, pid: str) -> object:
        """`/programs/{pid}/course-coverage` — покрытие по курсам."""
        return self._get(f"/programs/{pid}/course-coverage")

    def recommendations(self, pid: str) -> object:
        """`/programs/{pid}/recommendations` — предрассчитанные рекомендации."""
        return self._get(f"/programs/{pid}/recommendations")

    def glossary(self) -> object:
        """`/meta/glossary` — глоссарий терминов сервиса."""
        return self._get("/meta/glossary")
