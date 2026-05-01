# scripts/full_rebuild.py
import sys
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

# Список файлов и папок, подлежащих удалению
to_remove = [
    DATA / "processed" / "parsed_skills.pkl",
    DATA / "processed" / "skill_weights.json",
    DATA / "processed" / "vacancy_clusters_junior.pkl",
    DATA / "processed" / "vacancy_clusters_middle.pkl",
    DATA / "processed" / "vacancy_clusters_senior.pkl",
    DATA / "models" / "ltr_ranker_xgb_regressor.joblib",
    DATA / "embeddings" / "cache" / "skill_embeddings.json",
    DATA / "embeddings" / "skill_embeddings.json",
    DATA / "embeddings" / "vacancy_embeddings.json",
    DATA / "embeddings" / "market_embeddings_junior.pkl",
    DATA / "embeddings" / "market_embeddings_middle.pkl",
    DATA / "embeddings" / "market_embeddings_senior.pkl"
]

# Удаляем папку embeddings/cache целиком
cache_dir = DATA / "embeddings" / "cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)

for f in to_remove:
    if f.exists():
        f.unlink()
        print(f"Удалён: {f}")

print("Все старые кэши и модели удалены.")

# Формируем список команд для последовательного выполнения
commands = [
    ["python", "main.py", "--skip-collection"],                               # пересчёт навыков
    ["python", "scripts/train_clusters.py", "--level", "all"],                # обучение кластеров
    ["python", "main.py", "--train-model"],                                   # обучение LTR
    ["python", "main.py", "--skip-collection", "--run-gap-analysis"]          # gap-анализ с рекомендациями
]

# Выполняем все команды по очереди
for cmd in commands:
    print(f"\n>>> Запуск: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"Ошибка при выполнении: {cmd}")
        sys.exit(1)

print("\nПолный цикл перестроения завершён. Рекомендации и графики обновлены.")