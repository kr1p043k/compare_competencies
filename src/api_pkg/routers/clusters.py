"""Clusters summary and detail."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.analyzers.clustering.vacancy_clustering import VacancyClusterer
from src.models.api_responses import ClusterSummaryResponse, ClustersByLevelResponse
from src.models.enums import ExperienceLevel

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["clusters"])
limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/api/clusters/summary",
    response_model=ClusterSummaryResponse,
    response_model_exclude_none=True,
)
@limiter.limit("20/minute")
async def clusters_summary(
    request: Request,
    clusterer_instance: VacancyClusterer = Depends(deps.get_clusterer),
):
    result = {}
    for lvl in ExperienceLevel:
        clusterer_instance.load_model(lvl)
        if clusterer_instance.is_fitted:
            result[lvl] = {
                "clusters": clusterer_instance.n_clusters_,
                "type": clusterer_instance.clusterer_type,
                "top_clusters": [
                    {
                        "id": cid,
                        "name": clusterer_instance._generate_cluster_name(cid),
                        "top_skills": clusterer_instance.get_top_skills_in_cluster(
                            cid, top_n=5
                        ),
                    }
                    for cid in range(clusterer_instance.n_clusters_)
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
    clusterer_instance: VacancyClusterer = Depends(deps.get_clusterer),
):
    if not clusterer_instance.is_fitted:
        clusterer_instance.load_model(level)
    if not clusterer_instance.is_fitted:
        raise HTTPException(status_code=503, detail="Модели кластеров не загружены")
    clusters = []
    for cid in range(clusterer_instance.n_clusters_):
        clusters.append(
            {
                "id": cid,
                "name": clusterer_instance._generate_cluster_name(cid),
                "top_skills": clusterer_instance.get_top_skills_in_cluster(
                    cid, top_n=5
                ),
            }
        )
    return {"level": level, "clusters": clusters}
