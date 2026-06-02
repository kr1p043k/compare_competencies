"""Full rebuild: clean cache → pipeline → clusters → model → gap analysis."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import setup_structlog
from src.pipeline import runner as pr
from scripts.train_clusters import train_clusters

setup_structlog()

PIPELINE_ARGS = dict(
    skip_collection=True, skip_gap_analysis=True,
    query="", area_id=2, max_pages=10, period=30,
    show_vacancies=False, skip_details=False, excel=False,
    no_filter=False, queries_file=None, regions="113",
    industry=None, interactive=False, max_vacancies_per_query=2000,
    it_sector=False, use_async=True, async_workers=3,
    async_threshold=10000, run_gap_analysis=False,
    run_notebooks=False, force=False, use_llm=False,
)

print("=" * 60)
print("   ПОЛНАЯ ПЕРЕСБОРКА")
print("=" * 60)

pr.rebuild()

print("\n>>> Пайплайн (сбор не требуется)")
pr.run_full_pipeline(argparse.Namespace(**PIPELINE_ARGS))

print("\n>>> Кластеризация")
ok = train_clusters(level="all", save_report=True, interpret=True)
if not ok:
    sys.exit(1)

print("\n>>> LTR-модель")
pr.run_train_model()

print("\n>>> GAP-анализ")
gap_args = {**PIPELINE_ARGS, "skip_gap_analysis": False, "run_gap_analysis": True}
pr.run_full_pipeline(argparse.Namespace(**gap_args))

print("\n" + "=" * 60)
print("   ПЕРЕСБОРКА ЗАВЕРШЕНА")
print("=" * 60)
