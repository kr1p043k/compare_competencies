# Как контрибьютить

Коротко: ветка от `dev` -> зелёные проверки -> merge в `dev`. `main` и PROD не
трогаем. Локальную разработку и окружение см. в `docs/ONBOARDING.md`.

## Ветки

- Базовая ветка всегда `dev`. Имя ветки: `feat/<кратко>`, `fix/<кратко>`,
  `docs/<кратко>`, `refactor/<кратко>`, `chore/<кратко>`, `test/<кратко>`.
- Один законченный кусок работы = одна ветка = один осмысленный результат.
- `main` не трогаем: он обновляется только владельцем.

## Координация (общий репозиторий и БД)

Перед правкой файла заявите его на шине, чтобы не пересечься с другой сессией:

```bash
python C:\forCode\tools\coord.py claim <сессия> compare_competencies <файл...>
```

При конфликте - `msg` владельцу. По завершении - `release` и сообщение
`msg <сессия> compare_competencies DONE <задача>: <коммит>`.

Тесты и серверы общие: одна тяжёлая проверка на машину за раз. Для pytest
используйте общий раннер `C:\forCode\tools\test-fast.ps1` (он вне репозитория;
умеет `-Status`, не даёт запустить второй прогон). Не затирайте боевую
локальную БД и `data/reference/*`.

## Коммиты

- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`,
  `chore:`, `perf:`, `ci:`.
- Заголовок по-русски допустим; **без длинного тире** (используйте дефис или
  двоеточие).
- Тело - что и зачем, а не пересказ диффа.
- `NOTES.md` и `MEMORY.md` держим локально: не коммитим и не пушим.

## Что прогнать перед merge

Бэкенд:

```bash
ruff check src tests scripts --select E9,F63,F7,F82   # обязательный гейт
python scripts/check_secrets.py                       # нет секретов в git
pytest tests/api/test_x.py -q                          # точечно
```

Инварианты, которые ловит CI (не удаляйте затронутые тесты):

- `tests/test_audit_coverage.py` - каждый мутирующий роут пишет `audit_logs`;
- `tests/test_ci_gates.py` - обязательные гейты CI на месте и не ослаблены;
- `tests/test_docs_links.py` - относительные ссылки в `docs/` живые;
- дрейф `openapi.json`: `python scripts/export_openapi.py --output /tmp/o.json`
  и `diff -u openapi.json /tmp/o.json`.

Фронтенд (в каталоге `frontend`):

```bash
npm run typecheck   # tsc --noEmit
npm test            # vitest run
npm run build
```

`npm run lint` (ESLint) появится в `dev` после синхронизации из `main`
(задача 10.1); до этого ориентируйтесь на `typecheck` + тесты.

Миграции: `alembic upgrade head` и ровно одна голова (`alembic heads`).
`alembic check` пока показывает известный дрейф моделей - отдельная задача.

## UI / фронтенд

- Только примитивы `frontend/src/design` (PageHeader/EmptyState/ErrorState/
  Card/Badge/Button/Table/inputClass) и токены; никаких инлайн `style={{}}` и
  хардкод-цветов.
- Проверка в light и dark; состояния loading/empty/error.
- Изменения UI сопровождайте компонентными тестами и, для видимых экранов,
  скриншотами light/dark.

## Ревью

- CRITICAL-находки блокируют merge.
- Для UI-изменений желательно >=2 независимых ревьюера.
- Чек-лист - в шаблоне PR (`.github/pull_request_template.md`).
