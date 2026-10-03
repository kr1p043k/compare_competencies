"""Clusters summary and detail."""

import structlog
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.models.api_responses import ClustersByLevelResponse, ClusterSummaryResponse
from src.models.enums import ExperienceLevel

logger = structlog.get_logger("api")

router = APIRouter(tags=["clusters"])
limiter = Limiter(key_func=get_remote_address)


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
    result = {}
    for lvl in ExperienceLevel:
        # Локальный объект: глобальный кластерер не мутирует (нет гонок),
        # каждый уровень читается своей моделью (нет last-wins).
        local = VacancyClusterer()
        local.load_model(lvl)
        if local.is_fitted:
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
        else:
            result[lvl] = {"error": "not_fitted"}
    return result


@router.get("/clusters/{level}", response_model=ClustersByLevelResponse)
@limiter.limit("60/minute")
async def get_clusters(
    request: Request,
    level: ExperienceLevel = ExperienceLevel.MIDDLE,
):
    """Кластеры вакансий заданного уровня."""
    # Грузим запрошенный уровень явно: глобал после boot = senior (last-wins),
    # читать из него под видом `level` — stale-read. Локальный объект дешевле гонки.
    local = VacancyClusterer()
    local.load_model(level)
    if not local.is_fitted:
        raise HTTPException(status_code=503, detail="Модели кластеров не загружены")
    clusters = []
    for cid in range(local.n_clusters_):
        clusters.append(
            {
                "id": cid,
                "name": local._generate_cluster_name(cid),
                "top_skills": local.get_top_skills_in_cluster(
                    cid, top_n=5
                ),
            }
        )
    return {"level": level, "clusters": clusters}
