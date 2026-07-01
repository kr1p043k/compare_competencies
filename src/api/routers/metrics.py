"""API эндпоинты для метрик конверсии и gap-анализа."""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.monitoring.pipeline_metrics import pipeline_metrics, step_success_rate, pipeline_success_rate
from src.monitoring.gap_metrics import gap_metrics, gap_success_rate, gap_cache_hit_rate

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/prometheus")
async def prometheus_metrics():
    """Эндпоинт для Prometheus."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/conversion/funnel")
async def get_conversion_funnel():
    """Воронка конверсии по шагам пайплайна."""
    return {
        "funnel": pipeline_metrics.get_conversion_funnel(),
        "pipeline_success_rate": pipeline_success_rate._value.get() or 100.0
    }


@router.get("/gap/stats")
async def get_gap_stats():
    """Статистика по gap-анализу."""
    return {
        "success_rates": {
            profile: gap_success_rate.labels(profile_type=profile)._value.get() or 100.0
            for profile in ["base", "dc", "top_dc", "all"]
        },
        "cache_hit_rates": {
            profile: gap_cache_hit_rate.labels(profile_type=profile)._value.get() or 0.0
            for profile in ["base", "dc", "top_dc"]
        },
        "detailed": gap_metrics.get_stats()
    }


@router.get("/pipeline/summary")
async def get_pipeline_summary():
    """Сводка по пайплайну."""
    step_rates = {}
    for step in ["vacancy_fetch", "spam_filter", "skill_parse", 
                 "weight_normalize", "level_assign", "cluster_train", 
                 "ltr_train", "gap_compute"]:
        value = step_success_rate.labels(step=step)._value.get()
        step_rates[step] = value if value is not None else 100.0
    
    return {
        "total_success_rate": pipeline_success_rate._value.get() or 100.0,
        "step_success_rates": step_rates,
        "step_durations": pipeline_metrics.get_step_duration_stats(),
        "latest_pipeline": pipeline_metrics.get_latest()
    }


@router.get("/conversion/history")
async def get_conversion_history():
    """История конверсий между шагами."""
    return {
        "conversions": pipeline_metrics.conversion_counts
    }


@router.post("/pipeline/reset")
async def reset_pipeline_metrics():
    """Сбросить метрики текущего пайплайна."""
    pipeline_metrics.reset()
    return {"status": "reset", "message": "Pipeline metrics reset"}


@router.get("/health/metrics")
async def metrics_health():
    """Health check для метрик."""
    return {
        "status": "healthy",
        "metrics_available": True,
        "pipeline_metrics": pipeline_metrics.start_time is not None
    }