# Полный аудит проекта compare_competencies
# Дата: 2026-06-28

## CRITICAL (11)

### C1 — krm_models.py:44,64 — datetime.now() при импорте
`datetime.now(timezone.utc)` вызывается один раз при загрузке модуля. Все строки в БД получают одинаковый timestamp.
**Fix:** `default=lambda: datetime.now(timezone.utc)`

### C2 — prophet_forecast.py:161 — NameError n_points
`n_points` определена в `_fit_prophet_for_skill()`, но используется в `predict()`. Каждый прогноз через Prophet падает.
**Fix:** Сохранять `self._skill_points[skill] = n_points` при fit, читать в predict.

### C3 — pipeline_steps.py:386 — record_analysis() не существует
`gap_metrics.record_analysis()` вызывается, но метода нет в GapMetricsTracker. Падает gap-compute шаг.
**Fix:** Использовать `gap_metrics.start()` + `add_recommendations()` + `end()`.

### C4 — fix_rpd_data.py:162 — SQL injection
F-string интерполяция текста KSA в SQL. Одинарная кавычка ломает запрос.
**Fix:** Параметризованные запросы через `$N`.

### C5 — seed_users.py:10 — захардкоженные креды БД
`postgresql://postgres:700009@localhost:5432` прямо в коде. Репозиторий публичный.
**Fix:** Использовать env vars или config.settings.DATABASE_URL.

### C6 — frontend VacancyCard.tsx:288 — XSS
`dangerouslySetInnerHTML` на данных hh.ru. 3 точки инъекции. Нет DOMPurify.
**Fix:** Санитайз HTML или strip тегов.

### C7 — frontend auth.tsx:41 — JWT декодит header вместо payload
`token.split(".")[0]` берёт header, а не payload. Username всегда undefined.
**Fix:** Заменить `[0]` на `[1]`.

### C8 — frontend api.ts:5-13 — spread перезаписывает auth headers
`...options` после merge headers перезаписывает Authorization.
**Fix:** Не spread options.headers поверх merged.

### C9 — test_ltr_recommendation_engine.py:234,329 — дублирующий тест
Второй `test_predict_single_missing_skill` перекрывает первый. Первый никогда не запускается.
**Fix:** Переименовать или объединить.

### C10 — test_ltr_recommendation_engine.py:603,627 — дублирующий тест
Та же проблема с `test_load_model_with_manifest_present`.
**Fix:** Переименовать.

### C11 — runner.py:288-296 — asyncio.run() + get_pool()
`asyncio.run()` создаёт новый loop, `get_pool()` возвращает None. Падает запись в БД.
**Fix:** Создавать пул напрямую в `_write_db()`.

---

## HIGH (28)

### H1 — gap_analyzer.py:37-43 — SkillMetrics category всегда MISSING
`__post_init__` ставит category до setattr. Все навыки wrong category.
**Fix:** Устанавливать category после всех setattr.

### H2 — skills/trends.py:68-71 — ключ _meta перезаписывает метаданные
Если skill назван `_meta`, он затирает метаданные снапшота.
**Fix:** `if k == "_meta": continue`

### H3 — market_metrics.py:58-66 — __post_init__ мёртвый
`if not self.category` всегда False (дефолт MISSING truthy). Авто-categorization не работает.
**Fix:** Изменить дефолт на None или пустую строку.

### H4 — vacancy.py:87-89 — зарплата 0 = None
`if self.from_amount and self.to_amount` — 0 falsy. Нулевая зарплата теряется.
**Fix:** `if self.from_amount is not None and self.to_amount is not None`

### H5 — metrics.py:60 — GET endpoint ломает пайплайн
`end_pipeline()` в GET сбрасывает состояние трекера.
**Fix:** Использовать read-only getter.

### H6 — recommendation_engine.py:287 — case-sensitive trend lookup
`skill in trend_bonuses` без `.lower()`. Теряются бонусы трендов.
**Fix:** `skill.lower() in {k.lower(): v for k, v in trend_bonuses.items()}`

### H7 — recommendation_engine.py:206,216 — case-sensitive missing skills
`"Python" not in {"python"}` завышает missing в LTR.
**Fix:** Привести all_market к нижнему регистру.

### H8 — ltr_recommendation_engine.py:368-386 — разная семантика score
Softmax для 2+ навыков vs raw для 1. Разные абсолютные значения.
**Fix:** Всегда нормализовать одним способом.

### H9 — webhooks.py:44-47 — None секрет = True
Если секрет не задан, все вебхуки проходят.
**Fix:** Возвращать False, отдавать 403.

### H10 — populate_parsed_skills.py:7 — Windows-only путь
`"\\"` в разделителе. На Linux падает.
**Fix:** `Path` или `os.sep`.

### H11 — gap_metrics.py:134 — Histogram observe() в цикле
100 вызовов observe(count) искажает статистику.
**Fix:** Один вызов observe().

### H12 — frontend App.tsx:101 — fetch без authHeaders()
7 вызовов fetch без авторизации. Зависят от monkey-patch.
**Fix:** Добавить `...authHeaders()` в opts.

### H13 — frontend App.tsx:204 — stale closure role
role всегда null в pollPipeline из-за захвата при первом рендере.
**Fix:** useRef для role.

### H14 — frontend App.tsx:285 — .json() без проверки статуса
HTML ошибка → SyntaxError вместо понятного сообщения.
**Fix:** Проверить `res.ok` до `.json()`.

### H15 — initial_schema.py:259 — role CHECK нет student
Alembic не позволяет role='student', но seed его создаёт.
**Fix:** Добавить 'student' в CHECK или убрать из seed.

### H16 — sql + alembic — расходятся по схеме
Два источника правды: SQL-файлы и Alembic миграции.
**Fix:** Выбрать один источник, удалить другой.

### H17 — remove_non_it_disciplines.py:41 — downgrade пустой
Удаление 13 дисциплин необратимо. downgrade() — pass.
**Fix:** Восстановить данные в downgrade или документировать.

### H18 — sql/002_audit_fixes.sql:24 — ALTER мёртвой таблицы
market_skill_mappings удалена в 006. Запуск 002 после 006 падает.
**Fix:** Убрать ALTER или добавить IF EXISTS.

### H19 — sql/001:587 — DELETE сCASCADE в стартовом скрипте
Повторный запуск уничтожает данные.
**Fix:** Вынести в отдельную миграцию с backup.

### H20 — conftest.py:25-35 — два autouse мока sleep
Дублируют patch time.sleep. Второй перезаписывает первый.
**Fix:** Оставить одну фикстуру.

### H21 — test_decorators.py:51 — маскирует баг
Ручной `tb.log_key = "test_block"` скрывает потенциальную ошибку timed_block.
**Fix:** Убрать ручное присвоение.

### H22 — test_n8n_webhooks.py — мокает всё, проверяет nothing
Все зависимости замоканы, assertions = `result["ok"] is True`.
**Fix:** Интеграционные тесты с реальными путями.

### H23 — test_recommendation_engine.py:214 — неполные mock данные
Нет `student_skills`, `level_weights_used`, `cluster_context`. Потенциальный KeyError.
**Fix:** Добавить все обязательные ключи.

### H24 — evaluation/base.py:49 — coverage_ratio считает count
`metric_value = len(skills)` — считает количество, а не ratio.
**Fix:** Переименовать или пересчитать.

### H25 — frontend VacanciesList.tsx:142 — cityFilter не в deps
Изменение города не триггерит перезагрузку корректно.
**Fix:** Добавить cityFilter в dependency array.

### H26 — frontend AdminDashboard.tsx:265 — Drop+Seed без подтверждения
Одним кликом уничтожается БД.
**Fix:** window.confirm() или модал.

### H27 — vacancy_schema.py:1-76 — весь модуль мёртвый
Нигде не импортируется. Устаревший код.
**Fix:** Удалить.

### H28 — api_responses.py:233 — дефолт RUR вместо RUB
Устаревший ISO код. Везде в системе RUB.
**Fix:** Заменить на "RUB".

---

## MEDIUM (48)

### M1 — profile_evaluator.py:273 — формула readiness
DOMAIN_WEIGHT × weak_ratio, GAP_PENALTY × domain_coverage — перепутаны.
**Fix:** Переписать формулу.

### M2 — config.py:133-136 — readiness веса ≠ 1.0
Сумма 0.70. READINESS_DOMAIN_WEIGHT = 0.0.
**Fix:** Пересчитать веса.

### M3 — recommendation_engine.py:449 — KeyError c["id"]
Прямой доступ без .get(). Падает при отсутствии ключа.
**Fix:** `c.get("id", "unknown")`

### M4 — reranker.py:182,195 — KeyError при падении rerankers
Исключённые документы отсутствуют в blended.
**Fix:** Добавить fallback score для отсутствующих.

### M5 — skill_forecast.py:48 — отрицательные частоты
predict() может вернуть < 0 после мутации генов.
**Fix:** `max(0.0, result)`

### M6 — runner.py:373 — asyncio.run() connection churn
Каждый вызов создаёт/уничтажает пул.
**Fix:** Переиспользовать пул.

### M7 — db_writer.py:157 — ValueError на нечисловом id
`int(v.get("id", 0))` не обработан.
**Fix:** try/except или валидация.

### M8 — pipeline_metrics.py:106 — приватный API Prometheus
`_value.get()` сломается при обновлении библиотеки.
**Fix:** Использовать публичный API.

### M9 — gap_metrics.py:87 — приватный API Prometheus
Та же проблема.
**Fix:** Аналогично M8.

### M10 — gap_runner.py:201 — TimeoutError убивает parallel evaluation
Потеря результатов всех профилей.
**Fix:** Catch TimeoutError, логировать, продолжать.

### M11 — profile_evaluator.py:350 — мёртвый _get_or_create_comparator
Никогда не вызывается.
**Fix:** Удалить.

### M12 — profile_evaluator.py:365 — мёртвый _get_recommendation
Никогда не вызывается.
**Fix:** Удалить.

### M13 — profile_evaluator.py:387 — мёртвый _get_student_hash
Никогда не вызывается.
**Fix:** Удалить.

### M14 — profession_taxonomy.py:58 — pass вместо логирования
Неизвестные домены проглатываются молча.
**Fix:** Добавить warning.

### M15 — trend_analyzer.py:66-72 — OVERRIDE_PREV дублируется
Одинаковый dict в двух методах. При изменении одного забудут второй.
**Fix:** Вынести в константу.

### M16 — skills/trends.py:223 — re.compile в цикле
Регулярка компилируется на каждой итерации.
**Fix:** Module-level константа.

### M17 — student.py + krm_models.py — два ProfileEvaluation
Одно имя, два разных класса. Risk shadowing.
**Fix:** Переименовать один.

### M18 — enums.py:37-50 — ComparisonLevel дублирует ExperienceLevel
Три одинаковых enum. JobSearchLevel мёртвый.
**Fix:** Удалить дубли.

### M19 — backup_db.py:25 — pg_restore для .sql файла
Расширение должно быть .backup.
**Fix:** Заменить расширение.

### M20 — api_pkg/__init__.py:76 — ValueError на кривом Content-Length
`int(content_length)` без try/except.
**Fix:** Обернуть в try.

### M21 — teacher_analysis_runner.py:508 — pool shadowing
ThreadPoolExecutor затирает имя pool (asyncpg).
**Fix:** Переименовать.

### M22 — teacher_analysis_runner.py:243 — unbounded cache
Большие dict кешируются без TTL/размера.
**Fix:** Ограничить размер.

### M23 — evaluation/report.py:48 — pass_rate=1.0 при 0 метрик
Должно быть 0.0 или None.
**Fix:** `passed / total if total else 0.0`

### M24 — skill_filter.py:105 — lowercasит ключи
Ломает downstream lookup по оригинальным именам.
**Fix:** Сохранять оригинальный регистр.

### M25 — skill_filter.py:441,459 — git в devops и tools
elif делает tools мёртвым.
**Fix:** Убрать дубликат.

### M26 — ltr_recommendation_engine.py:416 — дублирующий import Ok
Уже импортирован на уровне модуля.
**Fix:** Убрать.

### M27 — ltr_recommendation_engine.py:187 — hasattr на всегда-существующем config
GLOBAL_RANDOM_SEED всегда есть.
**Fix:** Убрать hasattr.

### M28 — coverage_analyzer.py:43-89 — двойной matching
Каждый навык матчится дважды.
**Fix:** Один проход.

### M29 — skill_matcher.py:91 — fuzzy confidence хардкод 0.5
Не зависит от реального сходства строк.
**Fix:** Использовать реальный score.

### M30 — vacancy.py:391 — avg_skills_per_vacancy неправильная формула
Считает unique/skills_vacancies вместо total/vacancies.
**Fix:** Пересчитать.

### M31 — sql:570 — пароль teacher123 в seed
Слабый пароль в SQL. В Alembic — другой.
**Fix:** Использовать один пароль.

### M32 — sql:296 — pipeline_runs.user_id в SQL, нет в Alembic
Колонка потеряется при миграции через Alembic.
**Fix:** Добавить миграцию.

### M33 — sql:003 — нет parsed_skills колонки
Устаревший файл, расходится с 001.
**Fix:** Удалить или обновить.

### M34 — main.py:100 — нет проверки конфликтующих флагов
Противоречивые аргументы молча игнорируются.
**Fix:** Добавить mutual exclusion.

### M35 — main.py:94 — нет top-level exception handler
Сырые tracebacks при ошибках.
**Fix:** Обернуть в try/except с logging.

### M36 — seed_users.py:13 — нет обработки отсутствия users.json
FileNotFoundError без сообщения.
**Fix:** Проверить существование.

### M37 — seed_users.py:22 — KeyError на password/role
Прямой доступ к dict без проверки ключей.
**Fix:** `.get()` + валидация.

### M38 — fix_rpd_data.py:311 — DELETE всех competency_skills
Удаляет правильные данные из-за другой parse_version.
**Fix:** Добавить WHERE parse_version_id.

### M39 — frontend VacanciesList.tsx:229 — двойной fetch при поиске
setCurrentPage(1) + loadVacancies() = 2 запроса.
**Fix:** Дождаться состояния.

### M40 — frontend App.tsx:257 — stale profile в navigate-analysis
Empty deps = profile всегда "base".
**Fix:** useRef для profile.

### M41 — frontend VacancyCard.tsx:171 — "5 дня" вместо "5 дней"
Русская грамматика pluralization.
**Fix:** Корректные окончания.

### M42 — frontend PredictionsTab.tsx:180 — division by zero
Один элемент в data → data.length-1 = 0.
**Fix:** Проверка length > 1.

### M43 — frontend TeacherDashboard vs CompetencyTree — covColor пороги
0.55 = green в одном, yellow в другом.
**Fix:** Унифицировать пороги.

### M44 — frontend VacanciesList.tsx:210 — refreshVacancies без cityFilter
После pipeline city filter теряется.
**Fix:** Добавить cityFilter в refresh.

### M45 — frontend AdminDashboard:188 — создание admin без подтверждения
Teacher может создать admin.
**Fix:** Подтверждение.

### M46 — frontend AnalysisPanel:45 — dirCode не используется
В deps, но не в URL.
**Fix:** Убрать из deps или использовать.

### M47 — pipeline/clean.py:17 — open() без close
Resource leak.
**Fix:** with statement.

### M48 — hh_provider.py:33 — vacancy_id: str vs int
API ждёт int, параметр str.
**Fix:** Привести к int.

---

## LOW (51)

### L1 — profile_evaluator.py:39 — level_difficulty не используется
### L2 — recommendation_engine.py:47 — self.gap_analyzer мёртвый
### L3 — recommendation_engine.py:503 — _empty_recommendations мёртвый
### L4 — ltr_recommendation_engine.py:494 — _fallback_impacts мёртвый
### L5 — base.py:28 — abstract predict_impact возвращает T_pred, а не Result
### L6 — skill_filter.py:441 — "git" в devops, elif делает tools мёртвым
### L7 — vacancy_clustering.py:165 — best_score = -1 дважды
### L8 — vacancy_clustering.py:274 — level: ExperienceLevel vs str
### L9 — hh_responses.py:37 — from_ alias отличается от from_amount
### L10 — teacher_analysis.py:87 — generated_at: str вместо datetime
### L11 — krm_models.py:16 — _uuid() возвращает str, колонка UUID
### L12 — config.py:199 — validate_secret_key type annotation
### L13 — api/routers/metrics.py — нет __init__.py
### L14 — data_contracts.py:96 vs student.py:65 — top_recommendations типы
### L15 — ltr_recommendation_engine.py:416 — дублирующий import Ok
### L16 — ltr_recommendation_engine.py:187 — hasattr на GLOBAL_RANDOM_SEED
### L17 — pipeline/clean.py:8 — sys.stdout.reconfigure при импорте
### L18 — pipeline/clean.py:78 — sys импорт затенён
### L19 — cli/seed_db.py:108 — datetime.utcnow() deprecated
### L20 — cli/fix_rpd_data.py:57 — datetime.utcnow() deprecated
### L21 — scoring/vacancy_quality_scorer.py:146 — "охраник" опечатка
### L22 — evaluation/report.py:48 — pass_rate=1.0 при 0 метрик
### L23 — test_decorators.py:8 — не проверяет elapsed_sec
### L24 — conftest.py:36 — глобальный мок sentence_transformers
### L25 — test_cache_manager.py:138 — читает файл напрямую, не через API
### L26 — test_retry.py:45 — jitter bounds хрупкие
### L27 — test_snapshots.py — могут принять регрессию
### L28 — test_monitoring.py:36 — Prometheus counters не сбрасываются
### L29 — test_analyzers.py:14 — неполные assertions
### L30 — sql:006 — DROP без CASCADE
### L31 — sql:003:29 — индекс без WHERE
### L32 — main.py:44 — --use-async всегда True
### L33 — main.py:63 — base64 decode без error handling
### L34 — main.py:41 — --interactive не используется
### L35 — main.py:37,43 — --no-filter/--it-sector использование не проверяется
### L36 — main.py:10 — stdout/stderr rewrap потенциальные leak
### L37 — seed_users.py:17 — неатомарные вставки
### L38 — seed_users.py:22 — нет проверки pgcrypto
### L39 — seed_users.py:34 — нет CLI аргументов
### L40 — alembic add_directions:91 — HARD DELETE users без бэкапа
### L41 — alembic add_directions:114 — downgrade не восстанавливает
### L42 — alembic add_audit_fixes:60 — downgrade не восстанавливает CHECK
### L43 — alembic merge_branches:18 — пустой merge без валидации
### L44 — alembic add_llm_tables:68 — downgrade полагается на CASCADE
### L45 — frontend RegionCombobox — мёртвый компонент
### L46 — frontend StatsCards — мёртвый компонент
### L47 — frontend figma/ImageWithFallback — мёртвый компонент
### L48 — frontend App.tsx — неиспользуемые импорты
### L49 — frontend CompetencyTrendsPanel — потенциально неверный endpoint
### L50 — frontend PredictionsTab:123 — declining tab показывает growing
### L51 — frontend MiniChart — Infinity координаты при 1 элементе

---

## ИТОГО

| Уровень | Кол-во |
|---------|--------|
| CRITICAL | 11 |
| HIGH | 28 |
| MEDIUM | 48 |
| LOW | 51 |
| **Всего** | **138** |
