import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.analyzers.vacancy_clustering import VacancyClusterer

logger = structlog.get_logger(__name__)

for level in ["junior", "middle", "senior"]:
    c = VacancyClusterer()
    if c.load_model(level):
        logger.info("cluster_model_found", level=level, clusters=c.n_clusters_)
        for cluster_id in range(min(c.n_clusters_, 5)):
            skills = c.get_top_skills_in_cluster(cluster_id, top_n=5)
            name = c._generate_cluster_name(cluster_id)
            logger.info("cluster_info", level=level, cluster_id=cluster_id, name=name, top_skills=skills)
            print(f"  Кластер {cluster_id}: {name}")
            print(f"    → {', '.join(skills)}")
    else:
        logger.warning("cluster_model_not_found", level=level)
        print(f"{level}: модель не найдена")
