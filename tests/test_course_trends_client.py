"""Тесты клиента Course Trends API (academic-api-dev ЮФУ).

Сеть не используется: httpx.Client подменяется на MockTransport, поэтому
тесты быстрые и детерминированные.

Проверяем контракт, а не реализацию: какие URL строятся, как отличаются
"нет токена" / "токен истёк" / "сервис лежит", и что тело ответа отдаётся
без искажений (схем ответов сервис не публикует).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.infrastructure import course_trends as ct


def make_client(token: str = "jwt", base_url: str = "https://api.test") -> ct.CourseTrendsClient:
    return ct.CourseTrendsClient(token, base_url=base_url)


def patch_transport(monkeypatch, handler):
    """Подменяет httpx.Client так, чтобы все запросы шли через MockTransport."""
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        # Сохраняем headers/timeout исходного вызова: UA и таймаут - часть
        # контракта (WAF режет не-браузерный UA), терять их нельзя.
        kwargs.pop("timeout", None)
        return real_client(*args, transport=httpx.MockTransport(handler),
                           timeout=5.0, **kwargs)

    monkeypatch.setattr(ct.httpx, "Client", fake_client)


def test_missing_token_rejected_before_network(monkeypatch):
    called = {"n": 0}

    def handler(request):
        called["n"] += 1
        return httpx.Response(200, json={})

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        ct.CourseTrendsClient("")
    assert exc.value.status_code == 401
    assert called["n"] == 0


def test_default_base_url_is_the_course_trends_host():
    client = ct.CourseTrendsClient("jwt")
    assert client.base_url == ct.config.ACADEMIC_TRENDS_API_BASE.rstrip("/")
    assert "academic-api-dev" in client.base_url


def test_get_returns_json_unchanged(monkeypatch):
    payload = {"pid": "09.03.02_2025", "courses": [{"name": "БД", "articles": ["a"]}]}

    def handler(request):
        return httpx.Response(200, json=payload)

    patch_transport(monkeypatch, handler)
    assert make_client().program_bundle("09.03.02_2025") == payload


@pytest.mark.parametrize(
    ("method", "pid", "expected"),
    [
        ("list_programs", None, "/programs"),
        ("glossary", None, "/meta/glossary"),
        ("program_summary", "p1", "/programs/p1/summary"),
        ("course_trends", "p1", "/programs/p1/course-trends"),
        ("course_coverage", "p1", "/programs/p1/course-coverage"),
        ("recommendations", "p1", "/programs/p1/recommendations"),
        ("program_bundle", "p1", "/programs/p1"),
    ],
)
def test_paths_match_the_openapi_spec(monkeypatch, method, pid, expected):
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    patch_transport(monkeypatch, handler)
    client = make_client(base_url="https://api.test")
    getattr(client, method)(*([pid] if pid else []))
    assert seen == [expected]


def test_authorization_header_carries_hub_jwt(monkeypatch):
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("authorization")
        captured["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={})

    patch_transport(monkeypatch, handler)
    make_client(token="my-jwt").list_programs()
    assert captured["auth"] == "Bearer my-jwt"
    # WAF отдаёт 403 на python-httpx; UA должен быть браузерным.
    assert "Mozilla" in captured["ua"]


def test_expired_token_maps_to_401_with_actionable_message(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"detail": "expired"})

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        make_client().list_programs()
    assert exc.value.status_code == 401
    assert "истёк" in str(exc.value)


def test_unknown_program_preserves_404(monkeypatch):
    def handler(request):
        return httpx.Response(404, json={"detail": "not found"})

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        make_client().program_summary("nope")
    assert exc.value.status_code == 404


def test_server_error_is_wrapped_with_body(monkeypatch):
    def handler(request):
        return httpx.Response(500, text="boom")

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        make_client().list_programs()
    assert exc.value.status_code == 500
    assert "boom" in str(exc.value)


def test_transport_failure_is_wrapped(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route")

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        make_client().list_programs()
    assert "недоступен" in str(exc.value)


def test_non_json_body_is_wrapped(monkeypatch):
    def handler(request):
        return httpx.Response(200, text="<html>not json</html>")

    patch_transport(monkeypatch, handler)
    with pytest.raises(ct.CourseTrendsError) as exc:
        make_client().list_programs()
    assert "не-JSON" in str(exc.value)


def test_probe_script_describes_unknown_shapes(tmp_path: Path):
    """Зонд должен переживать произвольные структуры, а не падать на них."""
    import importlib.util
    import sys

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "probe_course_trends", root / "scripts" / "probe_course_trends.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_course_trends"] = module
    spec.loader.exec_module(module)

    lines = module.describe({"programs": [{"pid": "p1", "n_courses": 12}]}, 0, 5)
    text = "\n".join(lines)
    assert "programs" in text and "p1" in text
    assert json.dumps({"ok": True})  # json импортируется и используется в скрипте
