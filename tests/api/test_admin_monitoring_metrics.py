"""Метрики /admin/monitoring/summary: корректный p95 (без 1e18) и честные топы."""
import math

from src.api_pkg.routers.admin import _p95_from_buckets

INF = float("inf")


def test_p95_empty_buckets_returns_none() -> None:
    assert _p95_from_buckets([]) is None


def test_p95_zero_total_returns_none() -> None:
    assert _p95_from_buckets([(0.1, 0.0), (INF, 0.0)]) is None


def test_p95_in_finite_bucket_returns_that_bucket() -> None:
    buckets = [(0.1, 5.0), (0.5, 9.5), (1.0, 10.0), (INF, 10.0)]
    assert _p95_from_buckets(buckets) == 0.5


def test_p95_reached_only_in_inf_returns_largest_finite_bucket() -> None:
    buckets = [(0.1, 9.0), (1.0, 9.0), (INF, 10.0)]
    result = _p95_from_buckets(buckets)
    assert result is not None
    assert result == 1.0
    assert result != 1e18
    assert math.isfinite(result)


def test_p95_single_finite_bucket_returns_that_bucket() -> None:
    assert _p95_from_buckets([(0.5, 7.0)]) == 0.5


async def test_admin_monitoring_summary_produces_finite_p95(monkeypatch) -> None:
    from starlette.requests import Request

    import src.api_pkg.routers.admin as adminmod
    import src.monitoring.metrics as metricsmod

    payload = (
        "# HELP api_request_duration_seconds API request duration in seconds\n"
        "# TYPE api_request_duration_seconds histogram\n"
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="0.01"} 1.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="0.05"} 3.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="0.1"} 6.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="0.25"} 8.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="0.5"} 9.5\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="1.0"} 10.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="2.5"} 10.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="5.0"} 10.0\n'
        'api_request_duration_seconds_bucket{method="GET",path="/api/x",status="200",le="+Inf"} 10.0\n'
        'api_request_duration_seconds_count{method="GET",path="/api/x",status="200"} 10.0\n'
        'api_request_duration_seconds_sum{method="GET",path="/api/x",status="200"} 3.7\n'
    )

    monkeypatch.setattr(metricsmod, "get_metrics", lambda: (payload.encode(), "text/plain"))
    # DB-блок пропускаем: get_pool() возвращает None.
    monkeypatch.setattr(adminmod, "get_pool", lambda: None)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/admin/monitoring/summary",
        "query_string": b"",
        "headers": [],
    }
    fn = getattr(adminmod.admin_monitoring_summary, "__wrapped__", adminmod.admin_monitoring_summary)
    out = await fn(Request(scope))

    p95 = out["latency_p95_by_endpoint"].get("GET /api/x")
    assert p95 is not None
    assert p95 != 1e18
    assert math.isfinite(p95)
    assert p95 <= 5.0
    assert {"endpoint": "GET /api/x", "requests": 10} in out["busiest_by_requests"]
