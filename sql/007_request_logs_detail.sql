-- 007: audit trail for recommendation mutations — actor/action/target detail.
-- Admin answers "who added/changed recommendation X" via GET /admin/logs
-- (detail format: "<action> | <target> | by <email> (<role>)").
ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS detail TEXT;
