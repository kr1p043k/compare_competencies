-- =============================================================================
-- Docker Entrypoint Init — запускается 1 раз при создании контейнера БД
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- Пользователи (CHECK включает 'student')
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    full_name       TEXT NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'teacher'
                    CHECK (role IN ('admin', 'teacher', 'student')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Сессии
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      TEXT NOT NULL,
    ip_address      VARCHAR(45),
    user_agent      TEXT,
    logged_in_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    logged_out_at   TIMESTAMPTZ
);

-- Триггер updated_at для users
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_users_updated
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Seed: пользователи по умолчанию (роль student исправлена)
INSERT INTO users (email, password_hash, full_name, role) VALUES
    ('admin@compare-competencies.local', crypt('admin', gen_salt('bf')), 'Администратор', 'admin'),
    ('teacher@compare-competencies.local', crypt('prepod', gen_salt('bf')), 'Преподаватель', 'teacher'),
    ('student@compare-competencies.local', crypt('student', gen_salt('bf')), 'Студент', 'student')
ON CONFLICT (email) DO NOTHING;
