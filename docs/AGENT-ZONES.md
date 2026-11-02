# Зоны ответственности сессий (compare_online)

Обновляется при смене владельца файла. Перед правкой — `python tools/coord.py claim`.
Правило: **файл из чужой зоны не редактировать без сообщения в шину.**

## cluster-audit (эта сессия)

```
src/api_pkg/client_ip.py
src/api_pkg/routers/auth.py            (логин, брутфорс)
src/api_pkg/routers/admin.py
src/api_pkg/routers/health.py
src/api_pkg/routers/student.py
src/api_pkg/routers/teacher.py
src/api_pkg/student_actions.py
src/api_pkg/__init__.py
src/config.py                          (SECRET_KEY)
seed_users.py
alembic/versions/login_bruteforce_guard.py
frontend/src/design/**                 (новый слой)
frontend/src/styles/**                 (токены, globals)
frontend/src/app/App.tsx               (оболочка: шапка, табы, футер)
frontend/src/app/components/LoginPage.tsx
frontend/src/app/components/StudentsTab.tsx
frontend/src/app/components/PredictionsTab.tsx
frontend/vitest.config.ts              (пул форков: без него `npm test` падает OOM)
docs/AGENT-CONTRACT.md, docs/DESIGN-SYSTEM.md, docs/AGENT-ZONES.md
```

## opencode-main

```
src/api_pkg/startup.py
src/api_pkg/routers/{clusters,forecast,profiles,trends,vacancies,vacancies_by_skill,
                     trends_by_profession,pipeline,rpd,taxonomy,auth(sso-часть)}.py
src/n8n/**, src/predictors/**, src/parsing/**
остальной frontend/src/**
WS /admin/logs/ws и WS-токен в query
```

## orchestrator

```
src/pipeline/teacher_analysis_runner.py
data/reference/krm_disciplines_*.json
scripts/merge_annotations_to_krm.py
```

## perf-worker-1

```
scripts/merge_krm_aliases.py
```

## tests-worker-1 / tests-worker-4

```
tests/**
.github/workflows/**
frontend/src/lib/__tests__/**
```

## Известные общие точки (править только по договорённости)

- `src/api_pkg/routers/auth.py` — login у cluster-audit, SSO/сессии у opencode-main.
  Разделять по функциям, не переписывать файл целиком.
- `frontend/src/app/App.tsx` — оболочка у cluster-audit; вкладки и их содержимое
  у opencode-main.
- `pyproject.toml` / `requirements.txt` — изменения только с записью в NOTES.md.
