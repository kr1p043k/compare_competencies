# Задачи для opencode-main (зона A)

Составлено cluster-audit на основе проверки кода и живых запросов 2026-11-02.
Формат: `файл:строка` — дефект → минимальная правка → тест, который это докажет.
Guard-помощник для проверки принадлежности направления:
`src/api_pkg/routers/krm_teacher.py:100` — `_ensure_dir_access(user, dir_code)`
(лучше перенести его в `src/api_pkg/routers/auth.py` и импортировать оттуда).

## CRITICAL

### 1. zun.py — мутации без проверки направления (IDOR)
- `zun.py:995` `zun_add_entry`, `:1053` `zun_add_competency`, `:1163` `zun_patch_entry`,
  `:1187` `zun_delete_entry`, `:1348` `zun_import` (плюс `:957` `zun_put_scope`).
- Сейчас гейт только по роли (`:50`), принадлежность направления не проверяется:
  преподаватель направления A может писать в направление B.
- Правка: получить `user = await get_current_user(request)`, добавить в существующие
  SELECT/JOIN `d.code AS dir_code`, затем `_ensure_dir_access(user, dir_code)` до записи.
  Для `zun_import` — сразу после `_validate_dir_code`.
- Тест: teacher с `user_directions=['09.03.02']` пишет в компетенцию `01.03.01` → 403;
  admin → 201.

### 2. rpd.py — нет владельца у status/cancel
- `rpd.py:462-481` `rpd_status`, `:419-446` `rpd_cancel`: принимают любой `run_id`.
- Правка: у `PipelineRun` хранить владельца/направление при создании и возвращать 403
  при несовпадении; для upload/collect (`:300-336`, `:393-416`) — `_ensure_dir_access`.
- Тест: teacher A запрашивает статус запуска направления B → 403.

## MAJOR

### 3. rpd.py — гонка в `_rpd_procs` / `_rpd_cancel`
- Словари `:66-67`, записи `:103,113,138,148,215,258,296,345,389`, чтение в
  `rpd_cancel:424-425`. `_run_cli_sync` крутится через `asyncio.to_thread` (`:133`).
- Правка: `threading.Lock()` вокруг всех обращений либо хранить состояние только по
  `run_id` и мутировать лишь из event loop.
- Тест: 200 циклов cancel во время upload → нет `KeyError`/потерянных отмен.

### 4. rpd.py — загрузка PDF целиком в память
- `:320` `await file.read()`, `:325` `lstrip()[:5]` (полная копия), `:327` `write_bytes`.
- Правка: стримить чанками с лимитом (64 МБ), проверять `%PDF` по первому чанку.
- Тест: 100 МБ → 413, память не растёт; не-PDF → 400.

### 5. pipeline.py — повторный bootstrap
- `:404` и `:579` (`_reload_api_data`, доступен через `POST /pipeline/reload-api` `:554`)
  вызывают `run_startup(None)`, который (`startup.py:220`) создаёт новую задачу
  bootstrap и перезаписывает `pipeline_tasks` (`:261`).
- Правка: `asyncio.Lock` + флаг «уже идёт», из `_reload_api_data` перезагружать только
  данные, не пересоздавая задачи.
- Тест: одновременно reload-api и full-cycle → один bootstrap, статус задачи жив.

### 6. academic.py — SSO-токен в query
- `:102` `request.query_params.get("token","")` — токен утекает в логи прокси.
- Правка: убрать fallback, только `Authorization`/cookie (как `auth.py:142-152`).
- Тест: `GET /academic/krm-competencies?token=<valid>` без заголовка → 401.

### 7. academic.py — hub-JWT открытым текстом в БД
- `:267` вставляет `body.token` в `sessions.sso_token` (колонка `Text`).
- Правка: шифровать перед INSERT, расшифровывать в `_get_sso_token` (`:116-125`).
- Тест: после SSO значение в колонке ≠ сырой JWT и расшифровывается обратно.

### 8. academic.py — `/auth/sso` без лимита
- `:218-219` нет `@limiter.limit`, модуль не импортирует slowapi.
- Правка: свой `Limiter` + `10/minute`, отбрасывать токен длиннее 4096 символов.
- Тест: 11 запросов в минуту с одного IP → 429.

### 9. academic.py — IP из подделываемого заголовка
- `:261-262` читает `X-Forwarded-For` напрямую.
- Правка: `from src.api_pkg.client_ip import client_ip`.
- Тест: XFF от недоверенного пира → в `ip_address` реальный peer.

### 10. Токен из query вне auth
- `request_logger.py:39` `_extract_token` всё ещё читает `?token=`, из-за этого
  атрибуция `user` в аудите расходится с реальной авторизацией.
- Правка: убрать fallback; ожидаемо `user=anonymous`.
- Тест: запрос с `?token=` и без заголовка → в логе `anonymous`.

## MINOR

### 11. WS `/pipeline/ws` — токен в query
- `pipeline.py:853,862` `token: str = ""` (сессия через `resolve_session` уже проверяется —
  это сделано, остался только параметр).
- Правка: удалить параметр, только заголовок/cookie.
- Тест: `?token=<valid>` → код 4001; `Authorization: Bearer` → accept.

### 12. slowapi 429 без `Retry-After`
- `__init__.py:72-78` отдаёт `JSONResponse(429)` без заголовка.
- Правка: посчитать окно из лимита и добавить `Retry-After`.
- Тест: превышение лимита на `/auth/me` → заголовок присутствует.

## Заметки

- `__init__.py:24` (`from src.api_pkg import deps`) — импорт после `_disable_cmdstan()`;
  помечено `# noqa: F401,E402`, чтобы `ruff` был зелёным.
- `deps.py` переведён на ленивый `get_clusterer()`; `health.py:141` уже обновлён на
  `deps.get_clusterer().is_fitted`, иначе `/ready` падал с `AttributeError`.
- Пароли в `users.json` / `init-scripts/01-init.sql` по решению владельца пока не
  трогаем, но менять их всё равно нужно вручную.
