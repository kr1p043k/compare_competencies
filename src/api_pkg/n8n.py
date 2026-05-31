"""n8n integration: endpoint reference, auth, webhook registration."""

# ──────────────────────────────────────────────
# 1. AUTHENTICATION
# ──────────────────────────────────────────────
# Для вызова API из n8n требуется заголовок:
#   Authorization: Bearer <N8N_API_KEY>
#
# Получить ключ: настроить в .env N8N_API_KEY
# (генерируется при старте, если не задан)
#
# Rate limit: большинство эндпоинтов — 20-60 запросов/мин.
# Если нужно больше — настрой slowapi в config.
#
# WebSocket: /api/pipeline/ws — real-time pipeline progress

# ──────────────────────────────────────────────
# 2. ALL ENDPOINTS (51 HTTP + 1 WebSocket)
# ──────────────────────────────────────────────
# Legend:
#   M — Method, P — Path, RL — Rate Limit, Q — Query params, B — Body

N8N_ENDPOINTS: dict[str, list[dict]] = {
    "01_monitoring": [
        {"M": "GET", "P": "/",                    "RL": "-",   "Q": {},                        "B": {}, "note": "Service info"},
        {"M": "GET", "P": "/health",               "RL": "-",   "Q": {},                        "B": {}, "note": "Health check (evaluator + engine)"},
        {"M": "GET", "P": "/api/health",           "RL": "-",   "Q": {},                        "B": {}, "note": "Alias for /health"},
        {"M": "GET", "P": "/ready",                "RL": "-",   "Q": {},                        "B": {}, "note": "Readiness check (all components)"},
        {"M": "GET", "P": "/api/status",           "RL": "-",   "Q": {},                        "B": {}, "note": "Full status (vacancies, weights, clusters, trends)"},
        {"M": "GET", "P": "/api/regions",          "RL": "60/m","Q": {},                        "B": {}, "note": "All regions/cities from vacancies"},
        {"M":"POST","P": "/api/log",               "RL": "-",   "Q": {},                        "B": {"level": "str", "message": "str", "data?": "any"}, "note": "Write log entry"},
    ],
    "02_profiles": [
        {"M": "GET", "P": "/api/profiles/compare",              "RL": "20/m","Q": {},                                              "B": {}, "note": "Evaluate ALL student profiles"},
        {"M": "GET", "P": "/api/profiles/{profile}",            "RL": "60/m","Q": {},                                              "B": {}, "note": "Single profile details"},
        {"M": "GET", "P": "/api/profiles/{profile}/profession-evaluation", "RL": "30/m","Q": {},                                  "B": {}, "note": "Profession evaluation"},
        {"M": "GET", "P": "/api/recommendations/{profile}",     "RL": "30/m","Q": {},                                              "B": {}, "note": "Full LTR recommendations"},
        {"M": "GET", "P": "/api/skills/missing",                "RL": "30/m","Q": {"min_frequency": "int=1"},                    "B": {}, "note": "Skills not in whitelist"},
        {"M": "GET", "P": "/api/skills/dead",                   "RL": "30/m","Q": {},                                              "B": {}, "note": "Skills in whitelist not in vacancies"},
    ],
    "03_market": [
        {"M": "GET", "P": "/api/market/top-skills",           "RL": "60/m","Q": {"limit": "int=15"},                              "B": {}, "note": "Top market skills by hybrid weight"},
        {"M": "GET", "P": "/api/market/skill/{skill}",        "RL": "60/m","Q": {},                                                "B": {}, "note": "Single skill detail"},
        {"M": "GET", "P": "/api/market-competencies",         "RL": "60/m","Q": {},                                                "B": {}, "note": "All competencies (top 100)"},
    ],
    "04_vacancies": [
        {"M": "GET", "P": "/api/vacancies",                    "RL": "60/m","Q": {"limit": "int=50", "offset": "int=0", "experience?": "str", "search?": "str"}, "B": {}, "note": "Paginated vacancy list"},
        {"M": "GET", "P": "/api/vacancies/info",               "RL": "-",   "Q": {},                                              "B": {}, "note": "Vacancy file metadata"},
        {"M": "GET", "P": "/api/vacancies/{vacancy_id}",       "RL": "60/m","Q": {},                                              "B": {}, "note": "Full vacancy detail"},
        {"M": "GET", "P": "/api/vacancies/stats/summary",      "RL": "30/m","Q": {},                                              "B": {}, "note": "Aggregate stats by level"},
    ],
    "05_clusters": [
        {"M": "GET", "P": "/api/clusters/summary",             "RL": "20/m","Q": {},                                              "B": {}, "note": "All cluster levels summary"},
        {"M": "GET", "P": "/api/clusters/{level}",             "RL": "60/m","Q": {},                                              "B": {}, "note": "Clusters for level (junior/middle/senior)"},
    ],
    "06_trends": [
        {"M": "GET", "P": "/api/trends",                       "RL": "60/m","Q": {"top_n": "int=15", "min_change": "float=3.0"},  "B": {}, "note": "Trending skills analysis"},
    ],
    "07_taxonomy": [
        {"M": "GET", "P": "/api/taxonomy/coverage",            "RL": "20/m","Q": {},                                              "B": {}, "note": "Taxonomy coverage by category"},
        {"M": "GET", "P": "/api/taxonomy/professions",         "RL": "60/m","Q": {},                                              "B": {}, "note": "All professions"},
        {"M": "GET", "P": "/api/taxonomy/profession/{name}",   "RL": "60/m","Q": {},                                              "B": {}, "note": "Profession detail"},
        {"M": "GET", "P": "/api/taxonomy/profession/{name}/krm-coverage", "RL": "30/m","Q": {"skills": "str"},                    "B": {}, "note": "Profession coverage for skills"},
    ],
    "08_results": [
        {"M": "GET", "P": "/api/results/summary",              "RL": "30/m","Q": {},                                              "B": {}, "note": "All analysis results summary"},
        {"M": "GET", "P": "/api/results/recommendations/{profile}", "RL": "30/m","Q": {},                                        "B": {}, "note": "Saved recommendations"},
        {"M": "GET", "P": "/api/results/images/{profile}/{image_type}", "RL": "60/m","Q": {},                                    "B": {}, "note": "Profile image (radar/ml_importance/...)"},
        {"M": "GET", "P": "/api/results/images/coverage-comparison", "RL": "30/m","Q": {},                                       "B": {}, "note": "Coverage comparison PNG"},
        {"M": "GET", "P": "/api/results/images/skills-heatmap",     "RL": "30/m","Q": {},                                       "B": {}, "note": "Skills heatmap PNG"},
        {"M": "GET", "P": "/api/results/images/skill-correlation",  "RL": "30/m","Q": {},                                       "B": {}, "note": "Skill correlation PNG"},
    ],
    "09_pipeline": [
        {"M":"POST","P": "/api/pipeline/{action}",              "RL": "5/m", "Q": {"skip_collection": "bool", "run_gap_analysis": "bool", "regions?": "str", "query?": "str", "max_pages": "int=10", "period": "int=30"}, "B": {}, "note": "Run pipeline action (full-cycle/rebuild/...)"},
        {"M": "GET", "P": "/api/pipeline/active",              "RL": "30/m","Q": {},                                              "B": {}, "note": "Currently running task"},
        {"M": "GET", "P": "/api/pipeline/task/{task_id}",      "RL": "60/m","Q": {},                                              "B": {}, "note": "Task status"},
        {"M": "GET", "P": "/api/pipeline/tasks",               "RL": "30/m","Q": {"limit": "int=10"},                           "B": {}, "note": "Recent tasks"},
        {"M": "GET", "P": "/api/pipeline/status",              "RL": "30/m","Q": {},                                              "B": {}, "note": "Artifacts status"},
        {"M":"POST","P": "/api/pipeline/rebuild",              "RL": "2/m",  "Q": {},                                              "B": {}, "note": "Full rebuild (background)"},
        {"M":"POST","P": "/api/pipeline/refresh-cache",        "RL": "5/m",  "Q": {},                                              "B": {}, "note": "Clear cache dirs"},
        {"M":"POST","P": "/api/pipeline/reload-api",           "RL": "3/m",  "Q": {},                                              "B": {}, "note": "Reload API data"},
        {"M":"POST","P": "/api/pipeline/cancel/{task_id}",     "RL": "10/m", "Q": {},                                              "B": {}, "note": "Cancel running task"},
        {"M": "GET", "P": "/api/pipeline/gap-progress/{task_id}", "RL": "60/m","Q": {},                                          "B": {}, "note": "GAP progress by task"},
        {"M": "WS",  "P": "/api/pipeline/ws",                 "RL": "-",   "Q": {},                                              "B": {}, "note": "Real-time pipeline progress"},
    ],
    "10_admin": [
        {"M": "GET", "P": "/api/admin/whitelist",              "RL": "30/m","Q": {},                                              "B": {}, "note": "List whitelist skills"},
        {"M":"POST","P": "/api/admin/whitelist/add",           "RL": "10/m","Q": {},                                              "B": {"skills": "list[str]"}, "note": "Add to whitelist"},
        {"M":"POST","P": "/api/admin/whitelist/remove",        "RL": "10/m","Q": {},                                              "B": {"skills": "list[str]"}, "note": "Remove from whitelist"},
        {"M":"POST","P": "/api/admin/whitelist/backup",        "RL": "5/m",  "Q": {},                                              "B": {}, "note": "Backup whitelist"},
        {"M": "GET", "P": "/api/admin/students",               "RL": "30/m","Q": {},                                              "B": {}, "note": "List student profiles"},
        {"M":"POST","P": "/api/admin/pipeline/trigger",        "RL": "2/m",  "Q": {},                                              "B": {"action": "str", "regions?": "str", "skip_collection": "bool", "run_gap_analysis": "bool"}, "note": "Trigger pipeline action"},
        {"M": "GET", "P": "/api/admin/export/excel",           "RL": "3/m",  "Q": {},                                              "B": {}, "note": "Export vacancies to Excel"},
    ],
}

# ──────────────────────────────────────────────
# 3. RECOMMENDED n8n WORKFLOWS
# ──────────────────────────────────────────────
# Each workflow is a JSON stub to import into n8n.
# Files are in: src/n8n/workflows/

N8N_WORKFLOWS: dict[str, str] = {
    "nightly_pipeline":      "nightly_pipeline.json    — Еженочный пайплайн: сбор → анализ → графики",
    "profile_monitor":       "profile_monitor.json     — Мониторинг профилей: изменения ко дню",
    "trend_alert":           "trend_alert.json         — Оповещение о скачках трендов навыков",
    "student_onboarding":    "student_onboarding.json  — Приём нового студента через webhook",
    "weekly_report":         "weekly_report.json       — Еженедельный отчёт: LLM + TG + Email + Postgres",
    "gap_analysis_watch":    "gap_analysis_watch.json  — Запуск gap-анализа по расписанию",
}

# ──────────────────────────────────────────────
# 4. n8n WEBHOOK ENDPOINTS (receive from n8n)
# ──────────────────────────────────────────────
# POST /api/n8n/webhook/<name> — принимает callback/trigger от n8n
# Список зарегистрированных вебхуков:
N8N_WEBHOOKS: dict[str, dict] = {
    "student-created": {
        "method": "POST",
        "path": "/api/n8n/webhook/student-created",
        "description": "n8n уведомляет о создании нового студента",
        "body": {"profile_name": "str", "skills": "list[str]", "target_level": "str"},
    },
    "pipeline-completed": {
        "method": "POST",
        "path": "/api/n8n/webhook/pipeline-completed",
        "description": "n8n уведомляет о завершении внешнего пайплайна",
        "body": {"task_id": "str", "status": "str", "artifacts": "dict"},
    },
    "alert": {
        "method": "POST",
        "path": "/api/n8n/webhook/alert",
        "description": "Входящий алерт от n8n (trend spike, error, ...)",
        "body": {"type": "str", "severity": "str", "message": "str", "data?": "any"},
    },
}

# ──────────────────────────────────────────────
# 5. ENVIRONMENT (для n8n HTTP Request узла)
# ──────────────────────────────────────────────
# BASE_URL = http://localhost:8000  (dev)
#          = https://your-domain.com (prod)
# AUTH_HEADER = Authorization: Bearer <N8N_API_KEY>
