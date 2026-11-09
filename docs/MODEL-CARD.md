# MODEL-CARD: LTR skill ranker

Карточка модели для Learning-to-Rank ранкера навыков. Описывает назначение,
признаки, метрики, ограничения, происхождение артефакта и шаги переобучения.
Код: `src/predictors/ltr_recommendation_engine.py`. Контракт выбора прод-модели
проверяется `tests/predictors/test_ltr_contract.py`.

## Назначение

Модель ранжирует навыки для студента: по профилю студента и рынку вакансий
выдаёт порядок навыков, наиболее релевантных целевой роли. Оценки
**query-relative** (сравнимы внутри одного запроса/роли), поэтому используются
для упорядочивания, а не как абсолютная вероятность.

- Запрос (query) = роль/профессия; документ = навык.
- Выход: сырой балл релевантности; NDCG@10 как основная метрика качества.

## Признаки

Отгружаемый артефакт использует **8 признаков** (`_RUNTIME_FEATURES`):

| признак | смысл |
|---|---|
| `level_encoded` | уровень навыка (junior/middle/senior), закодирован 1..3 |
| `category_encoded` | категория навыка (23 категории таксономии) |
| `cosine_sim` | близость навыка к профилю роли (эмбеддинги) |
| `in_student_profile` | навык уже есть у студента (0/1) |
| `skill_freq_normalized` | нормированная частота навыка на рынке |
| `co_occurrence` | совместная встречаемость с навыками студента |
| `category_avg_weight` | средний рыночный вес категории навыка |
| `student_skills_count` | размер профиля студента, насыщенный к 1 |

Пространство признаков зафиксировано: артефакт с другим набором/порядком
признаков не выбирается автоматически (`INCOMPATIBLE_MODELS`), а `feature_names`
хранятся и в самом `.joblib`, и в манифесте/`metrics.json` (см. ниже).

## Метрики (frozen-сплит)

Единый сплит для всех артефактов: `tests/fixtures/gold_frozen.csv`,
по-ролевой стратифицированный 25% holdout, seed 42. Скрипт:
`scripts/eval_frozen.py`; отчёт: `data/reports/frozen_eval.json`.

| модель | NDCG@10 | P@10 | MAP@10 |
|---|---|---|---|
| **ltr_ranker_ensemble_human.joblib** (прод по умолчанию) | **0.7455** | 0.750 | 0.4516 |
| baseline_frequency | 0.5227 | 0.575 | 0.3123 |
| baseline_cosine | 0.4978 | 0.550 | 0.2670 |

`ltr_ranker_ensemble_human` - усреднение 3 seed-моделей XGBRanker (seeds 42/43/44).
Прод-модель выбирается **детерминированно по метрике**: среди
feature-совместимых human-артефактов берётся наибольший NDCG@10, источник метрики
по приоритету - `frozen_eval.json` → `*.manifest.json` (`human_ndcg_at_10`) →
`*.joblib.metrics.json` (`ndcg@10`); ничьи - по имени файла.

## Калибровка (НЕ отдаётся в прод)

Калибратор (`data/models/ranker_calibrator*.joblib`) **в сервинге не участвует**:
в `src/` нет ни одной ссылки на него, движок ранжирует сырые query-relative баллы
и вероятность наружу не отдаёт. Отчёт мониторинга: `scripts/check_calibration.py`
→ `data/reports/calibration_report.json`.

Текущий замер на frozen-сплите (Platt на z-нормированных баллах): `log_loss=3.89`,
`brier=0.281`, тогда как константный baseline (`base_rate`, log_loss `0.69`) его
превосходит. Вывод: наивная калибровка по сырым баллам плохая; включать
калибратор в сервинг нельзя без отдельного решения и честного OOF-протокола.

## Дрейф признаков

`scripts/check_feature_drift.py` сравнивает распределение признаков на frozen
serving-фрейме с эталоном `data/reports/feature_drift_baseline.json` по KS, PSI и
сдвигу среднего; падает (`exit 1`) при выходе за пороги (KS/PSI `0.25`, |z| `1.0`).
Проверено, что реальный сдвиг (cosine_sim +0.2) даёт `KS=0.562, PSI=3.724` и
ненулевой exit. Эталон регенерируется ОСОЗНАННО (`--write-baseline`) после
намеренного изменения модели/данных.

## Манифест и версионирование

Каждый отгружаемый артефакт (`data/models/ltr_ranker_*.joblib`) обязан иметь
`<name>.manifest.json` или `<name>.joblib.metrics.json` с `feature_names` и
NDCG@10; валидатор - `scripts/check_model_manifests.py` (`--write` создаёт/обновляет
сайдкары из артефакта и frozen-отчёта). При загрузке движок сверяет
`model_version` манифеста с текущей моделью эмбеддингов (несовпадение - warning,
не ошибка).

## CI-гейт

Джоба `ml-regression` в `.github/workflows/ci.yml` запускает
`python scripts/eval_frozen.py --check` и падает при просадке NDCG@10 любой
оценённой модели против `data/reports/frozen_eval.json` (допуск `--tolerance 0.005`).

## Ограничения

- Данные разметки маленькие (~500 пар human-gold); сплит фиксирован и не может
  расти без потери сопоставимости метрик.
- NDCG@10 - среднее по 4 ролям; метрика чувствительна к составу сплита.
- Калибровка не сервится и (по текущему замеру) некачественная.
- Дрейф-эталон сравнивается с самим собой на детерминированном сплите: он ловит
  изменения кода признаков/данных, а не живой прод-трафик.
- `ltr_ranker_canon/ensemble_final` (10 признаков) и `ltr_ranker_xgb_rank` (старый
  корпус 272 навыка) - legacy, автоматически не выбираются.
- Артефакты в `data/` гитигнорятся; отгружаемые файлы добавлены force-add и должны
  обновляться вместе с кодом.

## Происхождение (provenance)

1. Разметка: `labeling/gold_v1.csv` (human gold, 0/1/2) + `tests/fixtures/gold_frozen.csv`
   (frozen-сплит для гейта).
2. Обучение: `scripts/train_ltr_ranker.py` (XGBRanker `rank:ndcg`, qid по ролям,
   вакансии из Postgres, LOO-профиль роли без утечки меток).
3. Сборка ансамбля: `scripts/experiments/make_ensemble.py` (3 seed-модели →
   `ltr_ranker_ensemble_human.joblib`).
4. Метрики: `scripts/eval_frozen.py` → `data/reports/frozen_eval.json`.
5. Артефакт + сайдкары: `data/models/ltr_ranker_ensemble_human.{joblib,manifest.json}`,
   `...joblib.metrics.json`.

## Шаги переобучения

Требуется живая PostgreSQL (вакансии) и `SECRET_KEY` в окружении.

```bash
# 1. Полный цикл: обучение -> ансамбль -> манифесты -> гейт регресса
make retrain-ranker
# (эквивалент) python scripts/retrain_ranker.py --ensemble

# только рангер без ансамбля
python scripts/retrain_ranker.py --gold labeling/gold_v1.csv --vacancies 5000

# 2. После намеренного изменения модели/данных перегенерировать дрейф-эталон
python scripts/check_feature_drift.py --write-baseline

# 3. Перегенерировать калибровочный отчёт
python scripts/check_calibration.py

# 4. Доказать отсутствие регресса
python scripts/eval_frozen.py --check
```

Порядок обязателен: гейт `eval_frozen --check` идёт последним и отвергает
переобучение, если NDCG@10 просел против `data/reports/frozen_eval.json`.
