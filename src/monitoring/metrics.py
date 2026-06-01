from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics

# ── Auto-instrumentation (latency, RPS, errors per endpoint) ──────────────
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health", "/ready"],
)

# ── Custom pipeline metrics ───────────────────────────────────────────────

pipeline_duration = Histogram(
    "cc_pipeline_duration_seconds",
    "Pipeline full-run duration",
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600),
)

pipeline_stage_duration = Histogram(
    "cc_pipeline_stage_duration_seconds",
    "Duration per pipeline stage",
    labelnames=["stage"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

pipeline_runs_total = Counter(
    "cc_pipeline_runs_total",
    "Pipeline runs by status (success / failure)",
    labelnames=["status"],
)

skills_cached_total = Counter(
    "cc_skills_cache_operations_total",
    "Cache hits / misses for parsed skills",
    labelnames=["result"],
)

n8n_webhook_calls_total = Counter(
    "cc_n8n_webhook_calls_total",
    "Incoming n8n webhook calls by event type",
    labelnames=["event"],
)

active_pipelines = Gauge(
    "cc_active_pipelines",
    "Number of pipeline runs currently in progress",
)


@contextmanager
def track_pipeline_duration() -> Generator[None, Any, None]:
    active_pipelines.inc()
    start = time.monotonic()
    try:
        yield
        pipeline_runs_total.labels(status="success").inc()
    except BaseException:
        pipeline_runs_total.labels(status="failure").inc()
        raise
    finally:
        elapsed = time.monotonic() - start
        pipeline_duration.observe(elapsed)
        active_pipelines.dec()


@contextmanager
def track_stage_duration(stage: str) -> Generator[None, Any, None]:
    start = time.monotonic()
    try:
        yield
    finally:
        pipeline_stage_duration.labels(stage=stage).observe(time.monotonic() - start)


def setup_metrics(app: Any) -> None:
    instrumentator.add(metrics.latency(buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)))
    instrumentator.add(metrics.requests())
    instrumentator.add(metrics.request_size())
    instrumentator.add(metrics.response_size())
    instrumentator.add(metrics.requests_in_progress())
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
