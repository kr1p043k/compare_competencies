-- 006_market_skill_mappings_drop.sql
-- Удаление мёртвых таблиц: market_skill_mappings
-- Таблица нигде не используется (0 Python-ссылок во всём коде)

BEGIN;

DROP TABLE IF EXISTS market_skill_mappings;

COMMIT;
