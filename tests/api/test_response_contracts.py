"""Контрактные тесты: ответы API валидируются против объявленных response_model.

BACKLOG 7.5.2. Для набора GET-маршрутов проверяем:
1) в OpenAPI у ответа 200 есть $ref на модель с ожидаемым именем (это и есть
   объявленный response_model, попавший в схему);
2) реальный JSON-ответ валидируется этой моделью через model_validate.
Плюс негативный контроль: заведомо испорченное тело НЕ проходит валидацию
(доказывает, что проверка способна падать).

Пропущены (документировано): маршруты с живым hub/LLM и маршруты с
response_model=dict (без типа) - там контракта схемы нет.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.api_pkg.deps as deps
from src import Ok
from src.api_pkg import create_app
from src.api_pkg.routers.health import LiveBody
from src.models.api_responses import (
    HealthResponse,
    ProfessionsResponse,
    TopSkillsResponse,
    TrendsResponse,
)
from tests.conftest import open_all_gates


class _FakeTrendAnalyzer:
    def get_trending_skills(self, top_n: int = 15, min_change_percent: float = 3.0):
        return Ok({"rising": [], "falling": [], "emerging": []})


@pytest.fixture
def app():
    application = create_app()
    open_all_gates(application)
    limiter = getattr(application.state, "limiter", None)
    if limiter is not None:
        limiter.reset()
    return application


def _client(application) -> TestClient:
    # Без `with`: не запускаем lifespan (никакой БД/моделей).
    return TestClient(application, raise_server_exceptions=False)


def _openapi_ref_name(application, path: str) -> str | None:
    spec = application.openapi()
    for candidate in (path, "/api" + path):
        item = spec.get("paths", {}).get(candidate)
        if not item:
            continue
        responses = item.get("get", {}).get("responses", {})
        ok = responses.get("200") or responses.get(200)
        if not ok:
            continue
        schema = ok.get("content", {}).get("application/json", {}).get("schema", {})
        ref = schema.get("$ref", "")
        if ref.startswith("#/components/schemas/"):
            return ref.rsplit("/", 1)[-1]
    return None


CASES = [
    # (path, expected_model, needs_stubs)
    ("/health", HealthResponse, False),
    ("/live", LiveBody, False),
    ("/api/taxonomy/professions", ProfessionsResponse, False),
    ("/api/market/top-skills", TopSkillsResponse, True),
    ("/api/trends", TrendsResponse, True),
]


def _apply_stubs(application, path: str) -> None:
    if path == "/api/market/top-skills":
        application.dependency_overrides[deps.get_skill_weights] = lambda: {"python": 1.0, "sql": 0.5}
        application.dependency_overrides[deps.get_skill_freq] = lambda: {"python": 10, "sql": 5}
    elif path == "/api/trends":
        application.dependency_overrides[deps.get_trend_analyzer] = lambda: _FakeTrendAnalyzer()


@pytest.mark.parametrize(("path", "model", "needs_stubs"), CASES)
def test_response_matches_declared_model(app, path, model, needs_stubs):
    # 1) объявленная модель видна в OpenAPI как $ref с тем же именем
    assert _openapi_ref_name(app, path) == model.__name__, path

    if needs_stubs:
        _apply_stubs(app, path)

    resp = _client(app).get(path)
    assert resp.status_code == 200, (path, resp.status_code, resp.text[:200])
    # 2) реальный ответ валидируется объявленной моделью
    model.model_validate(resp.json())


def test_negative_control_tampered_body_fails(app):
    # Берём реальный ответ и портим тип поля - модель обязана это отвергнуть,
    # иначе проверка контракта была бы пустой.
    body = _client(app).get("/health").json()
    body["status"] = 12345  # status: str -> число
    with pytest.raises(ValidationError):
        HealthResponse.model_validate(body)
