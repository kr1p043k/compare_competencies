import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.vacancy_clustering import VacancyClusterer

for level in ["junior", "middle", "senior"]:
    c = VacancyClusterer()
    if c.load_model(level):
        print(f"{level}: {c.n_clusters_} кластеров")
        for cluster_id in range(min(c.n_clusters_, 5)):
            skills = c.get_top_skills_in_cluster(cluster_id, top_n=5)
            name = c._generate_cluster_name(cluster_id)
            print(f"  Кластер {cluster_id}: {name}")
            print(f"    → {', '.join(skills)}")
    else:
        print(f"{level}: модель не найдена")
