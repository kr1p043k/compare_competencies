"""Clusters summary and detail."""

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.clustering import vacancy_clustering as _vc
from src.analyzers.clustering.vacancy_clustering import (
    VacancyClusterer,
    load_cached_clusterer,
)
from src.api_pkg.routers.auth import require_authenticated
from src.models.api_responses import ClustersByLevelResponse, ClusterSummaryResponse
from src.models.enums import ExperienceLevel

logger = structlog.get_logger("api")

router = APIRouter(tags=["clusters"], dependencies=[Depends(require_authenticated())])
limiter = Limiter(key_func=get_remote_address)


def _load_clusterer(level) -> VacancyClusterer | None:
    """Кластерер уровня: из кэша, либо напрямую если класс подменён в тестах."""
    if VacancyClusterer is not _vc.VacancyClusterer:
        local = VacancyClusterer()
        local.load_model(level)
        return local
    return load_cached_clusterer(level)


@router.get(
    "/clusters/summary",
    response_model=ClusterSummaryResponse,
    response_model_exclude_none=True,
)
@limiter.limit("20/minute")
async def clusters_summary(
    request: Request,
):
    """Сводка кластеров вакансий."""
    # joblib.load (диск + unpickle) 4 моделей - синхронно в async это блокирует
    # event loop; считаем в потоке. Сама десериализация кэшируется по mtime/size
    # в vacancy_clustering: повторные запросы joblib.load не делают.
    def _compute() -> dict:
        result = {}
        for lvl in ExperienceLevel:
            # Локальный объект: глобальный кластерер не мутирует (нет гонок),
            # каждый уровень читается своей моделью (нет last-wins).
            local = _load_clusterer(lvl)
            if local is None or not local.is_fitted:
                result[lvl] = {"error": "not_fitted"}
                continue
            result[lvl] = {
                "clusters": local.n_clusters_,
                "type": local.clusterer_type,
                "top_clusters": [
                    {
                        "id": cid,
                        "name": local._generate_cluster_name(cid),
                        "top_skills": local.get_top_skills_in_cluster(
                            cid, top_n=5
                        ),
                    }
                    for cid in range(local.n_clusters_)
                ],
            }
        return result

    return await asyncio.to_thread(_compute)


@router.get("/clusters/{level}", response_model=ClustersByLevelResponse)
@limiter.limit("60/minute")
async def get_clusters(
    request: Request,
    level: ExperienceLevel = ExperienceLevel.MIDDLE,
):
    """Кластеры вакансий заданного уровня."""
    # Грузим запрошенный уровень явно: глобал после boot = senior (last-wins),
    # читать из него под видом `level` - stale-read. Локальный объект дешевле гонки.
    # load_model - синхронный диск/unpickle, но десериализация кэшируется по
    # mtime/size: считаем в потоке (генерация имён кластеров по-прежнему CPU).
    def _compute():
        local = _load_clusterer(level)
        if local is None or not local.is_fitted:
            return None
        return [
            {
                "id": cid,
                "name": local._generate_cluster_name(cid),
                "top_skills": local.get_top_skills_in_cluster(cid, top_n=5),
            }
            for cid in range(local.n_clusters_)
        ]

    clusters = await asyncio.to_thread(_compute)
    if clusters is None:
        raise HTTPException(status_code=503, detail="Модели кластеров не загружены")
    return {"level": level, "clusters": clusters}
