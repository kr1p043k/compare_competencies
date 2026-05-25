"""n8n интеграция: OpenAPI-спецификация, теги, healthcheck."""

# Справочник эндпоинтов для n8n HTTP Request узлов.
# Каждый эндпоинт сгруппирован по тегу с описанием и параметрами.
#
# Использование в n8n:
#   1. Создать HTTP Request узел
#   2. Method: GET / POST (как указано)
#   3. URL: http://<host>:8000<path>
#   4. Authentication: None (или Basic, если настроено)
#   5. Query params (если есть): передать как параметры

N8N_ENDPOINTS = {
    "monitoring": [
        {"path": "/health", "method": "GET", "description": "Health check (evaluator + engine status)"},
        {"path": "/ready", "method": "GET", "description": "Readiness check (all components)"},
        {"path": "/api/status", "method": "GET", "description": "Full status (vacancies, weights, profiles, clusters, trends)"},
    ],
    "profiles": [
        {"path": "/api/profiles/compare", "method": "GET", "description": "Compare all profiles (coverage, readiness)"},
        {"path": "/api/profiles/{profile}", "method": "GET", "description": "Get profile by name (skills, competencies)"},
        {"path": "/api/profiles/{profile}/profession-evaluation", "method": "GET", "description": "Profession evaluation with KRM coverage"},
        {"path": "/api/recommendations/{profile}", "method": "GET", "description": "Full recommendations for profile"},
    ],
    "market": [
        {"path": "/api/market/top-skills?limit=N", "method": "GET", "description": "Top N market skills by weight"},
        {"path": "/api/market/skill/{skill}", "method": "GET", "description": "Skill details (frequency, weight, category)"},
        {"path": "/api/market-competencies", "method": "GET", "description": "All market competencies (top 100)"},
    ],
    "vacancies": [
        {"path": "/api/vacancies?limit=N&offset=N&experience=&search=", "method": "GET", "description": "List vacancies with filters"},
        {"path": "/api/vacancies/{id}", "method": "GET", "description": "Vacancy details"},
        {"path": "/api/vacancies/stats/summary", "method": "GET", "description": "Vacancy statistics (by experience, salary)"},
        {"path": "/api/regions", "method": "GET", "description": "List available regions from vacancies"},
    ],
    "clusters": [
        {"path": "/api/clusters/summary", "method": "GET", "description": "All cluster levels summary"},
        {"path": "/api/clusters/{level}", "method": "GET", "description": "Clusters for level (junior/middle/senior)"},
    ],
    "trends": [
        {"path": "/api/trends?top_n=15&min_change=3.0", "method": "GET", "description": "Trending skills analysis"},
    ],
    "taxonomy": [
        {"path": "/api/taxonomy/coverage", "method": "GET", "description": "Taxonomy coverage by category"},
        {"path": "/api/taxonomy/professions", "method": "GET", "description": "All professions from taxonomy"},
        {"path": "/api/taxonomy/profession/{name}", "method": "GET", "description": "Profession detail (domains, skills, KRM)"},
        {"path": "/api/taxonomy/profession/{name}/krm-coverage?skills=", "method": "GET", "description": "KRM coverage for user skills"},
    ],
    "results": [
        {"path": "/api/results/summary", "method": "GET", "description": "Summary of all analysis results"},
        {"path": "/api/results/recommendations/{profile}", "method": "GET", "description": "Saved recommendations for profile"},
        {"path": "/api/results/images/{profile}/{image_type}", "method": "GET", "description": "Profile image (radar, ml_importance, cluster_insights, deficits)"},
    ],
    "pipeline": [
        {"path": "/api/pipeline/status", "method": "GET", "description": "Pipeline artifacts status"},
        {"path": "/api/pipeline/tasks?limit=10", "method": "GET", "description": "Recent pipeline tasks"},
        {"path": "/api/pipeline/task/{task_id}", "method": "GET", "description": "Pipeline task status"},
    ],
    "admin": [
        {"path": "/api/admin/whitelist", "method": "GET", "description": "List whitelist skills"},
        {"path": "/api/admin/whitelist/add", "method": "POST", "description": "Add skills to whitelist"},
        {"path": "/api/admin/whitelist/remove", "method": "POST", "description": "Remove skills from whitelist"},
        {"path": "/api/admin/whitelist/backup", "method": "POST", "description": "Backup whitelist"},
        {"path": "/api/admin/students", "method": "GET", "description": "List student profiles"},
        {"path": "/api/admin/pipeline/trigger", "method": "POST", "description": "Run pipeline action"},
        {"path": "/api/admin/export/excel", "method": "GET", "description": "Export vacancies to Excel"},
    ],
}
