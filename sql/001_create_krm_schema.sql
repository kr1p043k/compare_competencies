-- =============================================================================
-- Main Schema — PostgreSQL
-- Проект: compare_competencies (Направление 09.03.02)
-- Содержит: направления, дисциплины, компетенции, навыки, рынок, анализ
--
-- Создано: 2026-06-04
-- Использование: скопировать весь файл в pgAdmin4 → Query Tool → Execute
-- =============================================================================

-- ─── 1. База данных (создать через pgAdmin4 → Databases → Create) ──────────
-- CREATE DATABASE compare_competencies OWNER postgres;

-- ─── 2. Расширения ─────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";       -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "vector";         -- pgvector (VECTOR(384))

-- ─── 3. Тип-перечисление для KSA ───────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE ksa_type AS ENUM ('knowledge', 'abilities', 'skills');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ─── 4. Направление ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS directions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code        VARCHAR(20) UNIQUE NOT NULL,         -- "09.03.02"
    name        TEXT NOT NULL,                        -- "Информационные системы и технологии"
    profile     TEXT,                                 -- "Перспективные информационные технологии"
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_directions_code ON directions(code);

-- ─── 5. Дисциплина ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS disciplines (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_id    UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,                     -- "Операционные системы"
    name_en         TEXT,                              -- "Operating Systems"
    description     TEXT,                              -- краткое описание
    semester        INTEGER,                           -- номер семестра
    hours_total     INTEGER,                           -- всего часов
    hours_lecture   INTEGER,                           -- лекции
    hours_practice  INTEGER,                           -- практические
    hours_lab       INTEGER,                           -- лабораторные
    hours_self      INTEGER,                           -- самостоятельная работа
    control_form    VARCHAR(20),                       -- exam / test / coursework
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disciplines_direction ON disciplines(direction_id);

-- ─── 6. PDF-источники ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pdf_sources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    ocr_used        BOOLEAN NOT NULL DEFAULT FALSE,
    parse_status    VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (parse_status IN ('pending', 'parsed', 'failed', 'ocr_done')),
    error_message   TEXT,
    parsed_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_pdf_sources_discipline ON pdf_sources(discipline_id);

-- ─── 7. Версии парсинга ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS parse_versions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_id        UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    version             VARCHAR(50) NOT NULL,          -- "20260604_v1"
    total_disciplines   INTEGER NOT NULL DEFAULT 0,
    total_competencies  INTEGER NOT NULL DEFAULT 0,
    total_skills        INTEGER NOT NULL DEFAULT 0,
    total_ksa_items     INTEGER NOT NULL DEFAULT 0,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parse_versions_direction ON parse_versions(direction_id);

-- ─── 8. Компетенция ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS competencies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    code            VARCHAR(20) NOT NULL,              -- "ОПК-2"
    category        VARCHAR(10) NOT NULL               -- "УК" / "ОПК" / "ПК" / "ППК" / "ИП"
                    CHECK (category IN ('УК', 'ОПК', 'ПК', 'ППК', 'ИП')),
    number          VARCHAR(10) NOT NULL,              -- "1", "2", "3"...
    name            TEXT,                              -- формальное название
    description     TEXT,                              -- подробное описание
    parent_id       UUID REFERENCES competencies(id) ON DELETE CASCADE,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    parse_version_id UUID REFERENCES parse_versions(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_competencies_discipline ON competencies(discipline_id);
CREATE INDEX idx_competencies_category ON competencies(category);
CREATE INDEX idx_competencies_parent ON competencies(parent_id);

-- ─── 9. KSA-записи (знания / умения / навыки из RPD) ──────────────────────
CREATE TABLE IF NOT EXISTS ksa_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competency_id   UUID NOT NULL REFERENCES competencies(id) ON DELETE CASCADE,
    ksa_type        ksa_type NOT NULL,                 -- knowledge / abilities / skills
    original_text   TEXT NOT NULL,                     -- оригинальный текст из RPD
    cleaned_text    TEXT,                              -- очищенный текст
    sort_order      INTEGER NOT NULL DEFAULT 0,        -- порядок в исходном RPD
    parse_version_id UUID REFERENCES parse_versions(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ksa_competency ON ksa_entries(competency_id);
CREATE INDEX idx_ksa_type ON ksa_entries(ksa_type);

-- ─── 10. Таксономия навыков ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS skills (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,                         -- "python", "sql"
    name_en     TEXT,                                  -- English translation
    description TEXT,                                  -- описание навыка
    source      VARCHAR(20) NOT NULL DEFAULT 'it_skills'
                CHECK (source IN ('it_skills', 'rpd_skills', 'market')),
    category    VARCHAR(100),                          -- "programming", "databases"
    embedding   VECTOR(384),                           -- pgvector
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_skills_name ON skills(LOWER(name));
CREATE INDEX idx_skills_source ON skills(source);
CREATE INDEX idx_skills_category ON skills(category);

-- ─── 11. Связь компетенция ↔ навык (с типом KSA) ──────────────────────────
CREATE TABLE IF NOT EXISTS competency_skills (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competency_id     UUID NOT NULL REFERENCES competencies(id) ON DELETE CASCADE,
    skill_id          UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    ksa_type          VARCHAR(20) NOT NULL
                      CHECK (ksa_type IN ('knowledge', 'abilities', 'skills', 'flat')),
    source_text       TEXT,                            -- исходный текст, по которому нашли навык
    match_type        VARCHAR(20) NOT NULL DEFAULT 'fuzzy'
                      CHECK (match_type IN ('exact', 'fuzzy', 'stem')),
    parse_version_id  UUID REFERENCES parse_versions(id),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cs_competency ON competency_skills(competency_id);
CREATE INDEX idx_cs_skill ON competency_skills(skill_id);
CREATE UNIQUE INDEX idx_cs_unique
    ON competency_skills(competency_id, skill_id, ksa_type, COALESCE(parse_version_id, '00000000-0000-0000-0000-000000000000'));

-- ─── 12. Рекомендации преподавателя ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    competency_id   UUID REFERENCES competencies(id) ON DELETE SET NULL,
    suggestion      TEXT NOT NULL,                     -- текст рекомендации
    suggestion_type VARCHAR(20) NOT NULL
                    CHECK (suggestion_type IN ('modify', 'add', 'remove')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_recommendations_discipline ON recommendations(discipline_id);

-- ─── 13. Рыночные маппинги навыков (KRM → hh.ru) ──────────────────────────
CREATE TABLE IF NOT EXISTS market_skill_mappings (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id          UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    market_skill_name TEXT NOT NULL,                   -- как навык называется на рынке
    frequency         INTEGER NOT NULL DEFAULT 0,      -- сколько раз встретился
    weight            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    period            DATE,                            -- за какой период
    source            VARCHAR(50),                     -- "hh_search_iti", "hh_search_dc"
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_msm_skill ON market_skill_mappings(skill_id);
CREATE INDEX idx_msm_frequency ON market_skill_mappings(frequency DESC);

-- ─── 14. Анализ покрытия ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coverage_analyses (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id         UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    competency_id         UUID REFERENCES competencies(id) ON DELETE CASCADE,
    total_skills          INTEGER NOT NULL DEFAULT 0,
    market_matched_skills INTEGER NOT NULL DEFAULT 0,
    coverage_ratio        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    analysis_date         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ca_discipline ON coverage_analyses(discipline_id);
CREATE INDEX idx_ca_coverage ON coverage_analyses(coverage_ratio DESC);

-- ─── 15. Функция автообновления updated_at ─────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггеры для таблиц с updated_at
CREATE TRIGGER trg_directions_updated
    BEFORE UPDATE ON directions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_disciplines_updated
    BEFORE UPDATE ON disciplines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_competencies_updated
    BEFORE UPDATE ON competencies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_skills_updated
    BEFORE UPDATE ON skills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─── 16. Начальные данные ─────────────────────────────────────────────────
INSERT INTO directions (code, name, profile)
VALUES (
    '09.03.02',
    'Информационные системы и технологии',
    'Перспективные информационные технологии'
) ON CONFLICT (code) DO NOTHING;

-- =============================================================================
-- Готово. Для проверки:
--   SELECT table_name FROM information_schema.tables
--   WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
--   ORDER BY table_name;
-- =============================================================================
