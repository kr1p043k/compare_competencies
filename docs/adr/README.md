# Architecture Decision Records (ADR)

Короткие записи о значимых технических решениях проекта: контекст, решение,
последствия. Формат - лёгкий (по мотивам Nygard/MADR).

## Формат

Каждый ADR - файл `NNNN-короткое-имя.md` со секциями:

- **Статус** - proposed / accepted / superseded by ADR-XXXX / deprecated.
- **Контекст** - что заставило принять решение.
- **Решение** - что решили.
- **Последствия** - плюсы, минусы, что это накладывает на код/эксплуатацию.

Номер не переиспользуется; отменённое решение помечаем `superseded`, а не удаляем.

## Реестр

| ADR | Решение | Статус |
|---|---|---|
| [0001](0001-two-factor-switch-in-db.md) | 2FA master-switch хранится в БД (`app_settings`), а не в файле-флаге | accepted |
| [0002](0002-design-system-tokens.md) | UI строится на токенах и примитивах `src/design`, без хардкод-цветов | accepted |
| [0003](0003-unified-structlog.md) | Единое логирование через `setup_structlog` | accepted |
| [0004](0004-api-versioning.md) | Версионирование публичного API через `/api` + `GET /api/version` | accepted |
