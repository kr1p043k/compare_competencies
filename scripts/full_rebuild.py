# scripts/full_rebuild.py
import shutil
import subprocess
import sys
from pathlib import Path

import structlog

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
logger = structlog.get_logger(__name__)

# Список файлов и папок, подлежащих удалению
to_remove = [
    DATA / "cache" / "parsed_skills.joblib",
    DATA / "processed" / "skill_weights.json",
    DATA / "cache" / "clusters" / "vacancy_clusters_junior.pkl",
    DATA / "cache" / "clusters" / "vacancy_clusters_middle.pkl",
    DATA / "cache" / "clusters" / "vacancy_clusters_senior.pkl",
    DATA / "models" / "ltr_ranker_xgb_regressor.joblib",
    DATA / "cache" / "embeddings" / "market_embeddings_junior.pkl",
    DATA / "cache" / "embeddings" / "market_embeddings_middle.pkl",
    DATA / "cache" / "embeddings" / "market_embeddings_senior.pkl",
]

logger.info("full_rebuild_started")

# Удаляем папку cache/embeddings целиком (кэш эмбеддингов)
cache_dir = DATA / "cache" / "embeddings"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    logger.info("cache_directory_removed", path=str(cache_dir))

removed_count = 0
for f in to_remove:
    if f.exists():
        f.unlink()
        logger.info("file_removed", path=str(f))
        removed_count += 1

# Также удаляем папку cache/clusters, если там остались другие файлы (манифесты)
clusters_dir = DATA / "cache" / "clusters"
if clusters_dir.exists():
    shutil.rmtree(clusters_dir)
    logger.info("clusters_directory_removed", path=str(clusters_dir))

logger.info("cleanup_completed", files_removed=removed_count)
print(f"Удалено {removed_count} файлов кэша и моделей.")

# Формируем список команд для последовательного выполнения
commands = [
    ["python", "main.py", "--skip-collection"],
    ["python", "scripts/train_clusters.py", "--level", "all"],
    ["python", "main.py", "--train-model"],
    ["python", "main.py", "--skip-collection", "--run-gap-analysis"],
]

# Выполняем все команды по очереди
CMD_TIMEOUT = 1800  # 30 минут на команду
for cmd in commands:
    logger.info("running_command", command=" ".join(cmd))
    print(f"\n>>> Запуск: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=BASE, timeout=CMD_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.error("command_timeout", command=" ".join(cmd), timeout=CMD_TIMEOUT)
        print(f"Таймаут ({CMD_TIMEOUT}с) при выполнении: {cmd}")
        sys.exit(1)
    if result.returncode != 0:
        logger.error("command_failed", command=" ".join(cmd), returncode=result.returncode)
        print(f"Ошибка при выполнении: {cmd}")
        sys.exit(1)
    logger.info("command_completed", command=" ".join(cmd))

logger.info("full_rebuild_completed")
print("\nПолный цикл перестроения завершён. Рекомендации и графики обновлены.")
