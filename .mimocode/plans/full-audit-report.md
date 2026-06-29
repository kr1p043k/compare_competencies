# Полный аудит проекта compare_competencies
# Дата: 2026-06-28

---

## ЧАСТЬ 1: ОШИБКИ (баги)

### CRITICAL (11)

| # | Файл | Строка | Описание |
|---|------|--------|----------|
| C1 | krm_models.py | 44,64 | `datetime.now(timezone.utc)` вызывается при импорте. Все строки = один timestamp |
| C2 | prophet_forecast.py | 161 | `n_points` не определена в `predict()`. NameError при каждом прогнозе |
| C3 | pipeline_steps.py | 386 | `gap_metrics.record_analysis()` не существует. AttributeError |
| C4 | fix_rpd_data.py | 162 | SQL injection через f-string |
| C5 | seed_users.py | 10 | Захардкоженные креды БД `postgres:700009` в git |
| C6 | frontend/VacancyCard.tsx | 288 | XSS через `dangerouslySetInnerHTML` без санитайза |
| C7 | frontend/auth.tsx | 41 | JWT декодит `[0]` (header) вместо `[1]` (payload) |
| C8 | frontend/api.ts | 5-13 | Spread options перезаписывает Authorization header |
| C9 | test_ltr_recommendation_engine.py | 234,329 | Дублирующий тест — первый никогда не запускается |
| C10 | test_ltr_recommendation_engine.py | 603,627 | Дублирующий тест — первый никогда не запускается |
| C11 | runner.py | 288-296 | `asyncio.run()` + `get_pool()` = None. Падает запись в БД |

### HIGH (35)

| # | Файл | Строка | Описание |
|---|------|--------|----------|
| H1 | gap_analyzer.py | 37-43 | SkillMetrics category ставится до setattr — всегда WRONG |
| H2 | skills/trends.py | 68-71 | Ключ `_meta` в frequencies перезаписывает метаданные |
| H3 | market_metrics.py | 58-66 | `__post_init__` мёртвый — `if not self.category` всегда False |
| H4 | vacancy.py | 87-89 | Зарплата 0 treated as None (falsy check) |
| H5 | metrics.py | 60 | GET endpoint вызывает `end_pipeline()` — ломает трекер |
| H6 | recommendation_engine.py | 287 | Case-sensitive lookup теряет бонусы трендов |
| H7 | recommendation_engine.py | 206,216 | Case-sensitive missing skills — завышает LTR |
| H8 | ltr_recommendation_engine.py | 368-386 | Softmax vs raw — разная семантика score |
| H9 | webhooks.py | 44-47 | None секрет = True — все вебхуки проходят |
| H10 | populate_parsed_skills.py | 7 | Windows-only путь `"\\"` |
| H11 | gap_metrics.py | 134 | Histogram observe() в цикле 100 раз |
| H12 | frontend/App.tsx | 101 | 7 fetch без authHeaders() |
| H13 | frontend/App.tsx | 204 | Stale closure: role всегда null |
| H14 | frontend/App.tsx | 285 | .json() без проверки статуса |
| H15 | initial_schema.py | 259 | Role CHECK нет 'student', но seed его создаёт |
| H16 | sql+alembic | — | Два источника схемы расходятся |
| H17 | remove_non_it_disciplines.py | 41 | downgrade пустой — удаление необратимо |
| H18 | sql/002_audit_fixes.sql | 24 | ALTER мёртвой таблицы |
| H19 | sql/001 | 587 | DELETE сCASCADE в стартовом скрипте |
| H20 | conftest.py | 25-35 | Два autouse мока sleep — конфликт |
| H21 | test_decorators.py | 51 | Маскирует баг ручным override |
| H22 | test_n8n_webhooks.py | 74-133 | Мокает всё, assertions = "ok" is True |
| H23 | test_recommendation_engine.py | 214 | Неполные mock данные — KeyError |
| H24 | evaluation/base.py | 49 | coverage_ratio считает count, а не ratio |
| H25 | frontend/VacanciesList.tsx | 142 | cityFilter не в deps useEffect |
| H26 | frontend/AdminDashboard.tsx | 265 | Drop+Seed без подтверждения |
| H27 | vacancy_schema.py | 1-76 | Весь модуль мёртвый |
| H28 | api_responses.py | 233 | Дефолт RUR вместо RUB |
| H29 | startup.py | 214 | `create_task` без сохранения ref — GC может убить |
| H30 | student_loader.py | 44 | `target_level="middle"` для всех профилей |
| H31 | student_loader.py | 39-45 | `competencies = skills` — один список вместо разных |
| H32 | rpd_loader.py | 139 | Новый easyocr.Reader на каждый вызов — OOM |
| H33 | docker-compose.yml | 15 | Volume прячет HF кэш — модель перекачивается |
| H34 | docker-compose.yml | 152 | n8n ENCRYPTION_KEY пустой — creds без шифрования |
| H35 | pyproject.toml | 10 | `dependencies = []` — pip install ставит ничего |

### MEDIUM (60)

| # | Файл | Строка | Описание |
|---|------|--------|----------|
| M1 | profile_evaluator.py | 273 | Формула readiness — перепутаны операнды |
| M2 | config.py | 133-136 | Readiness веса ≠ 1.0 (сумма 0.70) |
| M3 | recommendation_engine.py | 449 | `c["id"]` KeyError |
| M4 | reranker.py | 182,195 | KeyError при падении всех rerankers |
| M5 | skill_forecast.py | 48 | Отрицательные частоты после мутации |
| M6 | runner.py | 373 | asyncio.run() connection churn |
| M7 | db_writer.py | 157 | ValueError на нечисловом id |
| M8 | pipeline_metrics.py | 106 | Приватный API Prometheus |
| M9 | gap_metrics.py | 87 | Приватный API Prometheus |
| M10 | gap_runner.py | 201 | TimeoutError убивает parallel evaluation |
| M11 | profile_evaluator.py | 350 | Мёртвый `_get_or_create_comparator` |
| M12 | profile_evaluator.py | 365 | Мёртвый `_get_recommendation` |
| M13 | profile_evaluator.py | 387 | Мёртвый `_get_student_hash` |
| M14 | profession_taxonomy.py | 58 | `pass` вместо логирования |
| M15 | trend_analyzer.py | 66 | OVERRIDE_PREV дублируется |
| M16 | skills/trends.py | 223 | re.compile в цикле |
| M17 | student.py+krm_models.py | — | Два `ProfileEvaluation` — shadowing |
| M18 | enums.py | 37 | ComparisonLevel дублирует ExperienceLevel |
| M19 | backup_db.py | 25 | pg_restore для .sql файла |
| M20 | api_pkg/__init__.py | 76 | ValueError на кривом Content-Length |
| M21 | teacher_analysis_runner.py | 508 | Pool shadowing |
| M22 | teacher_analysis_runner.py | 243 | Unbounded cache |
| M23 | evaluation/report.py | 48 | pass_rate=1.0 при 0 метрик |
| M24 | skill_filter.py | 105 | Lowercasit ключи — ломает lookup |
| M25 | skill_filter.py | 441 | git в devops+tools — elif мёртвый |
| M26 | ltr_recommendation_engine.py | 416 | Дублирующий import Ok |
| M27 | ltr_recommendation_engine.py | 187 | hasattr на всегда-существующем config |
| M28 | coverage_analyzer.py | 43 | Двойной matching |
| M29 | skill_matcher.py | 91 | Fuzzy confidence хардкод 0.5 |
| M30 | vacancy.py | 391 | avg_skills_per_vacancy — неправильная формула |
| M31 | sql:570 | — | Пароль teacher123 vs prepod |
| M32 | sql:296 | — | pipeline_runs.user_id в SQL, нет в Alembic |
| M33 | sql:003 | — | Нет parsed_skills колонки |
| M34 | main.py | 100 | Нет проверки конфликтующих флагов |
| M35 | main.py | 94 | Нет top-level exception handler |
| M36 | seed_users.py | 13 | Нет обработки отсутствия users.json |
| M37 | seed_users.py | 22 | KeyError на password/role |
| M38 | fix_rpd_data.py | 311 | DELETE всех competency_skills |
| M39 | frontend/VacanciesList.tsx | 229 | Двойной fetch при поиске |
| M40 | frontend/App.tsx | 257 | Stale profile в navigate-analysis |
| M41 | frontend/VacancyCard.tsx | 171 | "5 дня" вместо "5 дней" |
| M42 | frontend/PredictionsTab.tsx | 180 | Division by zero при 1 элементе |
| M43 | frontend TeacherDashboard vs CompetencyTree | — | covColor пороги расходятся |
| M44 | frontend/VacanciesList.tsx | 210 | refreshVacancies без cityFilter |
| M45 | frontend/AdminDashboard | 188 | Создание admin без подтверждения |
| M46 | frontend/AnalysisPanel | 45 | dirCode не используется |
| M47 | pipeline/clean.py | 17 | open() без close |
| M48 | hh_provider.py | 33 | vacancy_id: str vs int |
| M49 | ml/clusters.py | 43 | `vac["experience"].get()` — NoneType crash |
| M50 | visualization/coverage.py | 154 | ZeroDivisionError при total=0 |
| M51 | visualization/clusters.py | 35 | `c["id"]` KeyError |
| M52 | visualization/orchestration.py | 129 | subprocess.run блокирует event loop |
| M53 | infrastructure/hh_provider.py | 42 | requests.get() блокирует event loop |
| M54 | visualization/_config.py | 29 | Глобальный мутация matplotlib rcParams |
| M55 | Dockerfile | 16 | numpy 1.24.3 → 2.2.6 конфликт |
| M56 | Dockerfile | 27 | data/ в образ — утечка данных |
| M57 | docker-compose.yml | 192 | PostgreSQL порт на хост |
| M58 | docker-compose.yml | 148 | Дефолтный пароль n8n |
| M59 | .env.example | — | Нет WEBUI_SECRET_KEY, N8N_DB_PASSWORD и др. |
| M60 | startup.py | 129 | read_bytes() блокирует event loop |

### LOW (72)

| # | Описание |
|---|----------|
| L1 | profile_evaluator.py:39 — level_difficulty не используется |
| L2 | recommendation_engine.py:47 — self.gap_analyzer мёртвый |
| L3 | recommendation_engine.py:503 — _empty_recommendations мёртвый |
| L4 | ltr_recommendation_engine.py:494 — _fallback_impacts мёртвый |
| L5 | base.py:28 — abstract predict_impact тип неправильный |
| L6 | vacancy_clustering.py:165 — best_score=-1 дважды |
| L7 | vacancy_clustering.py:274 — level type mismatch |
| L8 | hh_responses.py:37 — from_ alias |
| L9 | teacher_analysis.py:87 — generated_at: str |
| L10 | krm_models.py:16 — _uuid() str vs UUID |
| L11 | config.py:199 — type annotation |
| L12 | api/routers/metrics.py — нет __init__.py |
| L13 | data_contracts.py vs student.py — top_recommendations типы |
| L14 | pipeline/clean.py:8 — reconfigure при импорте |
| L15 | pipeline/clean.py:78 — sys import shadowed |
| L16 | cli/seed_db.py:108 — datetime.utcnow() deprecated |
| L17 | cli/fix_rpd_data.py:57 — datetime.utcnow() deprecated |
| L18 | scoring/vacancy_quality_scorer.py:146 — "охраник" опечатка |
| L19 | test_decorators.py:8 — не проверяет elapsed_sec |
| L20 | conftest.py:36 — глобальный мок sentence_transformers |
| L21 | test_cache_manager.py:138 — файл напрямую, не через API |
| L22 | test_retry.py:45 — jitter bounds хрупкие |
| L23 | test_snapshots.py — могут принять регрессию |
| L24 | test_monitoring.py:36 — counters не сбрасываются |
| L25 | test_analyzers.py:14 — неполные assertions |
| L26 | sql:006 — DROP без CASCADE |
| L27 | sql:003:29 — индекс без WHERE |
| L28 | main.py:44 — --use-async всегда True |
| L29 | main.py:63 — base64 без error handling |
| L30 | main.py:41 — --interactive не используется |
| L31 | main.py:10 — stdout rewrap |
| L32 | seed_users.py:17 — неатомарные вставки |
| L33 | seed_users.py:22 — нет проверки pgcrypto |
| L34 | seed_users.py:34 — нет CLI аргументов |
| L35 | alembic add_directions:91 — HARD DELETE без бэкапа |
| L36 | alembic add_directions:114 — downgrade не восстанавливает |
| L37 | alembic add_audit_fixes:60 — downgrade не восстанавливает CHECK |
| L38 | alembic merge_branches:18 — пустой merge |
| L39 | alembic add_llm_tables:68 — downgrade полагается на CASCADE |
| L40 | frontend RegionCombobox — мёртвый |
| L41 | frontend StatsCards — мёртвый |
| L42 | frontend ImageWithFallback — мёртвый |
| L43 | frontend App.tsx — неиспользуемые импорты |
| L44 | frontend CompetencyTrendsPanel — неверный endpoint |
| L45 | frontend PredictionsTab:123 — declining показывает growing |
| L46 | frontend MiniChart — Infinity при 1 элементе |
| L47 | startup.py:51 — docstring после statement |
| L48 | startup.py:399 —冗余ная загрузка top_dc |
| L49 | startup.py:265 — title override только для MIDDLE |
| L50 | ml/clusters.py:98 — "c" и "na" дубли, skill C фильтруется |
| L51 | visualization/correlation.py:77 — numpy vs list |
| L52 | visualization/radar.py:22 — пустой all_skills |
| L53 | infrastructure/hh_provider.py:45 — resp.json() не validated |
| L54 | loaders/rpd_loader.py:273 — dead code |
| L55 | ground_truth/hh_proxy.py:33 — repeated failed I/O |
| L56 | api_pkg/request_logger.py:43 — SECRET_KEY fragile |
| L57 | api_pkg/request_logger.py:64 — naive datetime |
| L58 | database.py:32 — async_session_factory return type |
| L59 | visualization/coverage.py:159 — zero-height figure |
| L60 | startup.py:113 — non-atomic file write |
| L61 | startup.py:45 — load_model в waiting mode |
| L62 | pyproject.toml:43 — mypy ignore全局 |
| L63 | docker-compose volume прячет HF кэш |
| L64 | startup.py:151 — дублирующий парсинг |
| L65 | api_pkg/__init__.py:45 — dispose engine при shutdown |
| L66 | visualization/coverage.py:143 — empty profiles |
| L67 | ltr_recommendation_engine.py:187 — hasattr |
| L68 | ml/tracker.py:79 — '+' в имени файла Windows |
| L69 | startup.py:214 — fire-and-forget task |
| L70 | api_pkg/__init__.py:81 — каждый запрос логируется |
| L71 | database.py:83 — pool после close |
| L72 | cache_manager.py:24 — нет checksum |

---

## ЧАСТЬ 2: УЛУЧШЕНИЯ

### P0 — Безопасность / данные

| # | Область | Текущее состояние | Предлагаемое изменение |
|---|---------|-------------------|----------------------|
| I1 | Источники в коде | Пароли в seed_users.py, .env.example, docker-compose | Вынести в vault/env, добавить .gitignore для .env |
| I2 | Тесты auth | Ноль тестов аутентификации | Покрыть login/logout/token validation/role enforcement |
| I3 | XSS защита | Нет DOMPurify | Санитайзировать весь HTML из внешних источников |

### P1 — Точность / производительность / надёжность

| # | Область | Текущее состояние | Предлагаемое изменение |
|---|---------|-------------------|----------------------|
| I4 | Readiness formula | 4 веса с путаницей имён | Переписать: market=0.45, skill=0.30, gap=-0.25 |
| I5 | SkillMetrics category | __post_init__ мёртвый | Дефолт category=None, вызов после setattr |
| I6 | GapAnalyzer в RecommendationEngine | Плоский dict → crash | Обернуть или удалить мёртвый экземпляр |
| I7 | Prophet n_points | NameError в predict() | Сохранять при fit, читать в predict |
| I8 | ForecastEngine | SkillForecastEngine = random genes | Добавить warning если используется как основной |
| I9 | Recommendations | Все LOW, expected_outcome +1 | Пересчитать пороги, реальный расчёт coverage |
| I10 | Case sensitivity | 3 места с mismatch | Привести все ключи к .lower() |
| I11 | Pipeline steps | serial: collect → parse → analyze → recommend | Параллелить независимые этапы |
| I12 | hh.ru API | Нет circuit breaker | Добавить после 5 consecutive failures |
| I13 | Event loop blocking | requests.get(), subprocess.run, read_bytes() | Заменить на async аналоги |

### P2 — Качество / сопровождаемость

| # | Область | Текущее состояние | Предлагаемое изменение |
|---|---------|-------------------|----------------------|
| I14 | Два источника схемы | SQL-файлы + Alembic расходятся | Оставить Alembic как единственный источник |
| I15 | Мёртвый код | 20+ мёртвых методов/модулей | Удалить: vacancy_schema, 3 мёртвых компонента frontend |
| I16 | Rate limits | /vacancies/info без rate limiter | Добавить @limiter.limit |
| I17 | Кеширование API | Results читаются с диска каждый раз | HTTP Cache-Control / ETag |
| I18 | Error boundaries | React без ErrorBoundary | Обернуть основные вкладки |
| I19 | Тесты pipeline | Ноль тестов runner.py, background_collector | Интеграционные тесты ключевых путей |
| I20 | Prometheus metrics | Нет disk_usage, нет alerting rules | Добавить disk gauge + alertmanager |
| I21 | config.py | Хардкоки: THRESHOLDS, TIMEOUTS | Вынести в .env с дефолтами |
| I22 | O(n²) в analyzers | coverage_analyzer двойной matching | Один проход |
| I23 | Visualization | rcParams глобально мутируется при импорте | Lazy init или isolated figure settings |

### P3 — Хорошо иметь

| # | Область | Текущее состояние | Предлагаемое изменение |
|---|---------|-------------------|----------------------|
| I24 | EasyOCR | Новый Reader на каждый вызов | Кешировать как в RPDLoader._get_ocr_reader() |
| I25 | Joblib cache | Нет integrity check | Добавить checksum при load |
| I26 | Frontend timer | Сбрасывается при смене step | useRef дляхранения elapsed |
| I27 | Russian pluralization | "5 дня" вместо "5 дней" | Хелпер для окончаний |
| I28 | covColor пороги | Разные в TeacherDashboard/CompetencyTree | Унифицировать |
| I29 | DI container | Singleton без health check | Добавить ready status |
| I30 | Pipeline tasks | In-memory dict, теряется при restart | Persist на каждое изменение |

---

## ИТОГО

| Категория | CRIT | HIGH | MED | LOW | Всего |
|-----------|------|------|-----|-----|-------|
| Баги (часть 1) | 11 | 35 | 60 | 72 | 178 |
| Улучшения (часть 2) | 3 | 10 | 10 | 7 | 30 |
| **Гранд total** | **14** | **45** | **70** | **79** | **208** |
