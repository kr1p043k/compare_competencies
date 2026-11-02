"""Тесты прокси-роутера Course Trends (academic_trends) и чистых хелперов academic.

Прокси тестируется через TestClient с реальным app: авторизация сессии
(`_get_sso_token`) и клиент Course Trends подменяются, поэтому сеть не
используется. Проверяется поведение: маппинг ошибок сервиса (401->419,
404->404, прочее->502 без утечки внутренностей), пути проксирования,
нормализация статей (extract_articles + query/limit/offset).

Хелперы academic.py (`_jwt_username`, `_jwt_exp`, `_request_token`,
`_local_competencies`) - чистые функции без БД, тестируются напрямую.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api_pkg import create_app
from src.api_pkg.routers.academic import (
    _jwt_exp,
    _jwt_username,
    _local_competencies,
    _request_token,
)
from src.api_pkg.routers import academic_trends as trends_mod
from src.infrastructure.course_trends import CourseTrendsError
from tests.conftest import open_all_gates

# =====================================================================
# Фикстуры: реальный app, подменённые sso-сессия и CourseTrendsClient
# =====================================================================


class _FakeTrendsClient:
    """Подмена CourseTrendsClient: без сети, поведение задаётся через outcomes."""

    def __init__(self, token: str, base_url: str | None = None,
                 timeout: float = 60.0) -> None:
        self.token = token
        self.timeout = timeout
        self.calls: list[str] = []
        state["instances"].append(self)

    def _get(self, path: str) -> object:
        self.calls.append(path)
        outcome = state["outcomes"].pop(0) if state["outcomes"] else {"ok": True}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


state: dict = {"outcomes": [], "instances": []}


@pytest.fixture()
def trends(monkeypatch) -> SimpleNamespace:
    """TestClient с реальным create_app; sso-токен и клиент сервиса подменены."""
    state["outcomes"] = []
    state["instances"] = []

    monkeypatch.setattr(trends_mod, "CourseTrendsClient", _FakeTrendsClient)
    sso = AsyncMock(return_value=("hub-jwt-token", "tok-hash"))
    monkeypatch.setattr(trends_mod, "_get_sso_token", sso)

    app = create_app()
    open_all_gates(app)
    return SimpleNamespace(
        client=TestClient(app, raise_server_exceptions=False),
        outcomes=state["outcomes"],
        instances=state["instances"],
        sso=sso,
    )


def _jwt(payload: dict) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


# Реалистичный сырой ответ Course Trends: разные имена полей (сервис не
# публикует схему), вложенный не-статьёй объект, OpenAlex-подобные поля.
RAW_COURSE_TRENDS = {
    "program": {"id": "09.03.02", "name": "Программная инженерия", "courses_count": 8},
    "trends": [
        {
            "course": "Машинное обучение",
            "papers": [
                {
                    "title": "Deep Learning for Requirements Classification",
                    "abstract": "We apply transformers to classify requirements artifacts.",
                    "year": 2023,
                    "authors": ["Ivanov I.I.", "Petrov P.P."],
                    "doi": "10.1000/jml.2023.001",
                    "citations": 12,
                    "venue": "Journal of ML",
                },
                {
                    "name": "SQL Query Optimization in Practice",
                    "summary": "Index scans and adaptive plans.",
                    "published": "2021-05-14",
                    "author": ["Sidorov S.S."],
                    "cited_by_count": 5,
                },
                {"meta": {"version": 2}},
            ],
        },
    ],
}

ARTICLE_KEYS = {"title", "abstract", "year", "authors", "url", "doi",
                "citations", "source", "course"}


# =====================================================================
# Прокси: успешные ответы и пути
# =====================================================================


def test_list_programs_returns_payload_and_propagates_hub_token(trends):
    payload = {"programs": [{"pid": "09.03.02", "n_courses": 8, "avg_coverage": 0.61}]}
    trends.outcomes.append(payload)

    resp = trends.client.get("/api/academic/programs")

    assert resp.status_code == 200
    assert resp.json() == payload
    inst = trends.instances[0]
    assert inst.calls == ["/programs"]
    assert inst.token == "hub-jwt-token"


@pytest.mark.parametrize(
    ("url", "expected_path"),
    [
        ("/api/academic/glossary", "/meta/glossary"),
        ("/api/academic/programs/09.03.02/summary", "/programs/09.03.02/summary"),
        ("/api/academic/programs/09.03.02/course-trends", "/programs/09.03.02/course-trends"),
        ("/api/academic/programs/09.03.02/course-coverage", "/programs/09.03.02/course-coverage"),
        ("/api/academic/programs/09.03.02/recommendations", "/programs/09.03.02/recommendations"),
        ("/api/academic/programs/09.03.02", "/programs/09.03.02"),
    ],
)
def test_proxy_endpoints_hit_expected_service_paths(trends, url, expected_path):
    resp = trends.client.get(url)
    assert resp.status_code == 200
    assert trends.instances[0].calls == [expected_path]


# =====================================================================
# Прокси: маппинг ошибок сервиса
# =====================================================================


def test_service_401_maps_to_419_with_hub_hint(trends):
    trends.outcomes.append(CourseTrendsError("токен истёк", 401))

    resp = trends.client.get("/api/academic/programs")

    assert resp.status_code == 419
    detail = resp.json()["detail"]
    assert "истёк" in detail
    assert "hub.sfedu.ru" in detail


def test_service_404_maps_to_404_with_clear_message(trends):
    trends.outcomes.append(CourseTrendsError("not found", 404))

    resp = trends.client.get("/api/academic/programs/09.03.02/summary")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Программа не найдена в сервисе ЮФУ"


def test_service_500_maps_to_502_without_leaking_details(trends):
    trends.outcomes.append(
        CourseTrendsError('Ошибка сервиса ЮФУ (500): SECRET-INTERNAL-PAYLOAD {"stack": "x"}', 500)
    )

    resp = trends.client.get("/api/academic/programs")

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Сервис ЮФУ вернул ошибку. Попробуйте позже."
    assert "SECRET-INTERNAL-PAYLOAD" not in resp.text


def test_transport_failure_maps_to_502_without_leaking_details(trends):
    trends.outcomes.append(
        CourseTrendsError("Сервис ЮФУ недоступен: ConnectError host=10.0.0.7 SECRET", 0)
    )

    resp = trends.client.get("/api/academic/glossary")

    assert resp.status_code == 502
    assert "SECRET" not in resp.text
    assert "10.0.0.7" not in resp.text


def test_missing_hub_session_propagates_401(trends):
    trends.sso.side_effect = HTTPException(status_code=401, detail="Unauthorized")

    resp = trends.client.get("/api/academic/programs")

    assert resp.status_code == 401
    assert trends.instances == []


def test_session_without_hub_binding_propagates_403(trends):
    trends.sso.side_effect = HTTPException(status_code=403, detail="Сессия не привязана к хабу")

    resp = trends.client.get("/api/academic/programs")

    assert resp.status_code == 403


# =====================================================================
# /articles: нормализация, фильтр query, пагинация, сортировка
# =====================================================================


def _setup_articles(trends):
    trends.outcomes.append(RAW_COURSE_TRENDS)


def test_articles_return_canonical_shape_and_metadata(trends):
    _setup_articles(trends)

    resp = trends.client.get("/api/academic/programs/09.03.02/articles")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"pid", "total", "limit", "offset", "items", "source"}
    assert body["pid"] == "09.03.02"
    assert body["total"] == 2
    assert body["limit"] == 500
    assert body["offset"] == 0
    for item in body["items"]:
        assert set(item) == ARTICLE_KEYS
    first = body["items"][0]
    assert first["title"] == "Deep Learning for Requirements Classification"
    assert first["doi"] == "10.1000/jml.2023.001"
    assert first["authors"] == ["Ivanov I.I.", "Petrov P.P."]
    assert first["course"] == "Машинное обучение"
    # вторая статья нормализована из альтернативных имён полей
    second = body["items"][1]
    assert second["title"] == "SQL Query Optimization in Practice"
    assert second["year"] == 2021
    assert second["citations"] == 5


def test_articles_default_response_sorts_by_year_desc(trends):
    _setup_articles(trends)

    items = trends.client.get("/api/academic/programs/09.03.02/articles").json()["items"]

    years = [item["year"] for item in items]
    assert years == sorted(years, reverse=True)
    assert years == [2023, 2021]


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("sql", "SQL Query Optimization in Practice"),          # по заголовку
        ("IVANOV", "Deep Learning for Requirements Classification"),  # по автору, без учёта регистра
        ("transformers", "Deep Learning for Requirements Classification"),  # по аннотации
    ],
)
def test_articles_query_filters_title_author_abstract(trends, query, expected_title):
    _setup_articles(trends)

    body = trends.client.get(
        "/api/academic/programs/09.03.02/articles", params={"query": query}
    ).json()

    assert body["total"] == 1
    assert [item["title"] for item in body["items"]] == [expected_title]


def test_articles_query_without_matches_yields_empty_items(trends):
    _setup_articles(trends)

    body = trends.client.get(
        "/api/academic/programs/09.03.02/articles", params={"query": "quantum-blockchain"}
    ).json()

    assert body["total"] == 0
    assert body["items"] == []


def test_articles_pagination_limit_and_offset(trends):
    _setup_articles(trends)

    body = trends.client.get(
        "/api/academic/programs/09.03.02/articles", params={"limit": 1, "offset": 1}
    ).json()

    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["total"] == 2
    assert [item["title"] for item in body["items"]] == ["SQL Query Optimization in Practice"]


def test_articles_limit_clamped_to_at_least_one(trends):
    _setup_articles(trends)

    body = trends.client.get(
        "/api/academic/programs/09.03.02/articles", params={"limit": 0}
    ).json()

    assert body["limit"] == 1
    assert len(body["items"]) == 1


# =====================================================================
# academic.py: чистые хелперы JWT
# =====================================================================


class TestJwtUsername:
    def test_extracts_username(self):
        assert _jwt_username(_jwt({"username": "ivanov"})) == "ivanov"

    def test_falls_back_to_sub(self):
        assert _jwt_username(_jwt({"sub": "petrov"})) == "petrov"

    def test_falls_back_to_user_name(self):
        assert _jwt_username(_jwt({"user_name": "sidorov"})) == "sidorov"

    def test_no_username_keys_returns_none(self):
        assert _jwt_username(_jwt({"role": "admin"})) is None

    def test_malformed_token_returns_none(self):
        assert _jwt_username("not-a-jwt") is None
        assert _jwt_username("") is None

    def test_broken_base64_returns_none(self):
        assert _jwt_username("h.@@@!.s") is None

    def test_empty_username_value_returns_none(self):
        assert _jwt_username(_jwt({"username": ""})) is None


class TestJwtExp:
    def test_int_exp_returned(self):
        assert _jwt_exp(_jwt({"exp": 1735689600})) == 1735689600

    def test_float_exp_cast_to_int(self):
        assert _jwt_exp(_jwt({"exp": 1735689600.75})) == 1735689600

    def test_missing_exp_returns_none(self):
        assert _jwt_exp(_jwt({"username": "x"})) is None

    def test_malformed_token_returns_none(self):
        assert _jwt_exp("garbage") is None


class TestRequestToken:
    @staticmethod
    def _request(headers: list[tuple[bytes, bytes]] | None = None,
                 query_string: bytes = b""):
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers or [],
            "query_string": query_string,
        }
        return Request(scope)

    def test_bearer_header_wins(self):
        req = self._request([(b"authorization", b"Bearer abc123")],
                            query_string=b"token=q1")
        assert _request_token(req) == "abc123"

    def test_cookie_token(self):
        req = self._request([(b"cookie", b"token=c1")])
        assert _request_token(req) == "c1"

    def test_query_param_token(self):
        req = self._request(query_string=b"token=q1")
        assert _request_token(req) == "q1"

    def test_no_token_anywhere_is_empty(self):
        assert _request_token(self._request()) == ""


# =====================================================================
# academic.py: _local_competencies (AcademicGapAnalyzer подменён)
# =====================================================================


class TestLocalCompetencies:
    def test_defaults_to_it_direction_when_no_codes(self):
        with patch("src.analyzers.academic_gap.AcademicGapAnalyzer") as cls:
            cls.return_value.recommend.return_value = {"recommended_competencies": ["UK-1"]}
            result = _local_competencies("ml", 5, [])
        assert cls.call_args.kwargs == {"dir_code": "09.03.02"}
        cls.return_value.recommend.assert_called_once_with("ml", final_top_k=5)
        assert result == {"recommended_competencies": ["UK-1"]}

    def test_stops_at_first_direction_with_results(self):
        with patch("src.analyzers.academic_gap.AcademicGapAnalyzer") as cls:
            cls.return_value.recommend.side_effect = [
                {"recommended_competencies": [], "found_trends": ["python"]},
                {"recommended_competencies": ["UK-9"]},
            ]
            result = _local_competencies("ml", None, ["01.03.01", "02.03.02"])
        # found_trends уже в первом направлении: второй анализатор не создаётся
        assert [c.kwargs for c in cls.call_args_list] == [{"dir_code": "01.03.01"}]
        assert cls.return_value.recommend.call_count == 1
        assert result == {"recommended_competencies": [], "found_trends": ["python"]}

    def test_walks_directions_until_results_found(self):
        with patch("src.analyzers.academic_gap.AcademicGapAnalyzer") as cls:
            cls.return_value.recommend.side_effect = [
                {"recommended_competencies": [], "found_trends": []},
                {"recommended_competencies": [], "found_trends": []},
                {"recommended_competencies": ["UK-3"]},
            ]
            result = _local_competencies("ml", 3, ["01.03.01", "02.03.02", "09.03.02"])
        assert cls.return_value.recommend.call_count == 3
        assert result == {"recommended_competencies": ["UK-3"]}

    def test_returns_last_result_when_nothing_found(self):
        with patch("src.analyzers.academic_gap.AcademicGapAnalyzer") as cls:
            cls.return_value.recommend.return_value = {"recommended_competencies": []}
            result = _local_competencies("ml", None, ["01.03.01"])
        assert cls.return_value.recommend.call_count == 1
        assert result == {"recommended_competencies": []}
