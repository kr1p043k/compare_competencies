# Онбординг разработчика

Короткий путь от чистого клона до зелёных тестов. Всё, что описано, проверено
против текущего `dev`. Если команда не работает, сначала сверьтесь с
`README.md` и `docs/RUNBOOK.md`, потом правьте этот файл тем же PR.

## 1. Что нужно

- Python 3.11 (в CI именно 3.11; 3.12 локально тоже работает).
- Node 20 и npm (фронтенд).
- PostgreSQL 16 с расширением `pgvector` (или Docker - см. `docker-compose.yml`).
- Docker - опционально, только для поднятия БД/стенда.

## 2. Установка

```bash
git clone <repo> compare_competencies
cd compare_competencies

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cd frontend && npm ci && cd ..
```

## 3. Окружение

```bash
cp .env.example .env
```

Обязательный минимум в `.env`:

- `SECRET_KEY` - подпись токенов сессий (без него приложение не стартует).
- `DATABASE_URL` - `postgresql://<user>:<pass>@localhost:5432/compare_competencies`.
- `ALLOWED_ORIGINS` - список источников фронтенда; `*` запрещён в prod.
- `TEST_DATABASE_URL` - отдельная БД для тестов (см. `docs/TEST-ENV.md`).
- `GRAFANA_*`, `N8N_*`, `OLLAMA_*` - только если пользуетесь этими разделами.

`.env` и `.secrets/` в `.gitignore` - в git их не коммитим.

## 4. База данных

```bash
alembic upgrade head          # применить миграции
alembic heads                 # должна быть РОВНО одна голова
python -m src.cli.seed_db     # демо-данные (опционально)
```

Тесты НЕ пишут в боевую БД и не должны затирать `data/reference/*.json`:
пути из `config` в тестах подменяются на `tmp_path`.

## 5. Запуск

```bash
# бэкенд (порт 8000)
uvicorn src.api_pkg:app --host 0.0.0.0 --port 8000 --reload

# фронтенд (Vite, обычно порт 5173)
cd frontend && npm run dev
```

## 6. Проверки (то, что гоняет CI)

```bash
# бэкенд
ruff check src tests scripts --select E9,F63,F7,F82   # обязательный гейт
python scripts/check_secrets.py                       # нет секретов в git
pytest tests/api/test_x.py -q                         # точечно, быстрее всего

# дрейф OpenAPI (обязательный гейт)
python scripts/export_openapi.py --output /tmp/openapi.generated.json
diff -u openapi.json /tmp/openapi.generated.json

# фронтенд (в каталоге frontend)
npm run typecheck   # tsc --noEmit
npm test            # vitest run
npm run build
```

Правила по прогонам: одна тяжёлая проверка на машину за раз. Для pytest
используйте `tools/test-fast.ps1` (он не даёт запустить второй прогон и умеет
`-Status`). Не затирайте общую БД.

## 7. Структура

- `src/api_pkg/` - FastAPI-приложение: `__init__.py` (create_app, middleware),
  `routers/` (эндпоинты), `security_settings.py`, `request_logger.py`.
- `src/pipeline/`, `src/analyzers/`, `src/predictors/` - конвейер, парсинг, ML.
- `src/infrastructure/` - внешние источники (hh, hub, course trends).
- `frontend/src/app/` - React-приложение; `frontend/src/design/` - примитивы
  дизайн-системы.
- `tests/` - pytest; `alembic/versions/` - миграции; `docs/` - документация.

## 8. Рабочий процесс

- Работаем в отдельной ветке от `dev`. Готово и зелено -> merge в `dev`.
- `main` и PROD не трогаем.
- Перед правкой файла: `python tools/coord.py claim <session> compare_competencies <file...>`.
- Коммиты conventional, без длинного тире.
- `NOTES.md` и `MEMORY.md` держим локально, не коммитим.
- UI только на компоненты `frontend/src/design` (без инлайн-стилей и
  хардкод-цветов), с проверкой в light/dark.

## 9. Частые грабли

- Забыли `SECRET_KEY` - приложение падает на старте.
- Тесты идут в общую локальную БД по фиксированным ключам: не запускайте два
  прогона параллельно на одну БД.
- После правки эндпоинтов обновите `openapi.json`, иначе красный гейт дрейфа.
- Новый эндпоинт POST/PUT/PATCH/DELETE должен писать в `audit_logs`
  (тест `tests/test_audit_coverage.py`), иначе CI упадёт.
