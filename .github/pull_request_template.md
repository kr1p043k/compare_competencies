## Что и зачем

<!-- Одно-два предложения: какую проблему решает PR. -->

## Зона / задача

<!-- Метка и пункт бэклога, например: Зона E - 7.7.3 CONTRIBUTING. -->

## Тип

- [ ] feat (новая функциональность)
- [ ] fix (багфикс)
- [ ] refactor / chore / test / ci
- [ ] docs

## Проверки (отметьте, что прогнали)

- [ ] `ruff check src tests scripts --select E9,F63,F7,F82`
- [ ] `python scripts/check_secrets.py`
- [ ] `pytest tests` (точечно или полный прогон)
- [ ] дрейф `openapi.json` отсутствует
- [ ] `npm run typecheck` и `npm test` (для фронтенда)
- [ ] миграции: `alembic upgrade head`, одна голова

## UI (если затронут фронтенд)

- [ ] только `src/design` + токены, без инлайн-стилей и хардкод-цветов
- [ ] light/dark и состояния loading/empty/error
- [ ] компонентные тесты; скриншоты light/dark приложены

## Ревью

- [ ] CRITICAL-находок нет
- [ ] для UI: ревьюеров >= 2

## Прочее

- [ ] `NOTES.md` / `MEMORY.md` не входят в дифф
- [ ] `main` / PROD не затрагиваются; цель merge - `dev`
- [ ] коммиты в стиле conventional, без длинного тире

## Ссылки

<!-- Связанные issues/коммиты, если есть. -->
