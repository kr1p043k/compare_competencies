"""Prometheus metrics for the pipeline and API."""

import time
from functools import wraps

import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src import config

logger = structlog.get_logger(__name__)

pipeline_duration = Histogram(
    "pipeline_stage_duration_seconds",
    "Pipeline stage duration in seconds",
    ["stage"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
)

pipeline_errors = Counter(
    "pipeline_errors_total",
    "Total pipeline errors",
    ["stage"],
)

api_latency = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "path", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status"],
)

recommendations_generated = Counter(
    "recommendations_generated_total",
    "Total recommendations generated",
    ["profile"],
)

ltr_training_duration = Histogram(
    "ltr_training_duration_seconds",
    "LTR model training duration in seconds",
    buckets=(10, 30, 60, 120, 300, 600),
)

ltr_model_metrics = Gauge(
    "ltr_model_metric",
    "LTR model evaluation metrics",
    ["metric"],
)

vacancies_loaded = Gauge(
    "vacancies_loaded_total",
    "Total vacancies loaded",
)

active_profiles = Gauge(
    "active_profiles_total",
    "Number of active student profiles",
)

skill_count = Gauge(
    "skill_count_total",
    "Number of unique skills in the system",
)

shap_computation_duration = Histogram(
    "shap_computation_duration_seconds",
    "SHAP values computation duration",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)


def track_pipeline_stage(stage_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                pipeline_errors.labels(stage=stage_name).inc()
                raise
            finally:
                duration = time.time() - start
                pipeline_duration.labels(stage=stage_name).observe(duration)
        return wrapper
    return decorator


def get_metrics():
    return generate_latest(), CONTENT_TYPE_LATEST
