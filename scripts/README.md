# scripts/

Операционные скрипты проекта. Запуск: `python -m scripts.<имя>` (каталог - пакет,
есть `__init__.py`).

## Здесь (операционные и регулярно используемые)

- `preflight.py` - предполётные проверки окружения и зависимостей;
- `deploy.sh` - деплой (см. `deploy/` для полного цикла);
- `backfill_embeddings.py`, `seed_all_directions.py` - наполнение БД;
- `smoke.py`, `check_env.py`, `check_dependency_drift.py`, `check_krm_db_consistency.py`,
  `check_metric_stability.py` - проверки;
- `sync_*.py`, `prune_*.py`, `audit_*.py`, `merge_*.py`, `normalize_*.py`,
  `find_skill_duplicates.py` - обслуживание данных KRM/навыков;
- `eval_frozen.py` - гейт регресса ML (используется как CI-ориентир, не эксперимент);
- `check_model_manifests.py` - проверка, что у каждого отгружаемого артефакта есть
  `.manifest.json`/`.metrics.json` с `feature_names` и NDCG@10 (`--write` их создаёт);
- `check_calibration.py` - отчёт калибровки отгружаемого ранкера (log_loss/Brier на
  frozen-сплите; калибратор НЕ отдаётся в прод);
- `check_feature_drift.py` - дрейф признаков против baseline (KS/PSI/сдвиг среднего),
  `--write-baseline` сохраняет эталон;
- `ml_eval_common.py` - общий сборщик frozen-фреймов для eval_frozen/калибровки/дрейфа;
- `train_ltr_ranker.py`, `_dsn.py` - общие модули для скриптов и экспериментов.

## `experiments/` - разовые исследования

Скрипты для прогонов/экспериментов (battery*, loro*, calibrate*, make_*ensemble,
hpo_ranker, score_hpo, label_weak_v1, build_labeling_pairs, adjudicate,
benchmark_matching, probe_course_trends). Не входят в CI и операционные цепочки;
меняются редко, могут зависеть от локальных данных. Их импорты используют
абсолютные пути `scripts.*` / `scripts.experiments.*`.
