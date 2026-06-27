-- =============================================================================
-- Main Schema — PostgreSQL
-- Проект: compare_competencies (Направление 09.03.02)
-- Содержит: направления, дисциплины, компетенции, навыки, студентов, рынок, анализ
--
-- Создано: 2026-06-04
-- Использование: скопировать весь файл в pgAdmin4 → Query Tool → Execute
-- =============================================================================

-- ─── 1. База данных (создать через pgAdmin4 → Databases → Create) ──────────
-- CREATE DATABASE compare_competencies OWNER postgres;

-- ─── 2. Расширения ─────────────────────────────────────────────────────────
-- pgcrypto: gen_random_uuid(), crypt(), gen_salt()
-- vector:   pgvector (VECTOR(384) для эмбеддингов навыков)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── 3. Тип-перечисление для KSA ───────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE ksa_type AS ENUM ('knowledge', 'abilities', 'skills');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ─── 4. Функция проверки пароля (использует pgcrypto crypt/blowfish) ──────
CREATE OR REPLACE FUNCTION check_password(
    input_password TEXT,
    stored_hash TEXT
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN crypt(input_password, stored_hash) = stored_hash;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ─── 5. Направление (09.03.02, 09.03.01, ...) ──────────────────────────────
-- UNIQUE(code) был заменён на UNIQUE(code, profile) для поддержки нескольких
-- профилей с одинаковым кодом (например 09.03.01 с двумя профилями).
CREATE TABLE IF NOT EXISTS directions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code        VARCHAR(20) NOT NULL,                -- "09.03.02"
    name        TEXT NOT NULL,                        -- "Информационные системы и технологии"
    profile     TEXT,                                 -- "Перспективные информационные технологии"
    supervisor  VARCHAR(255),                         -- ФИО руководителя направления
    opop_year   INTEGER,                              -- год OPOP (2024, 2025...)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX uq_directions_code_profile ON directions(code, profile);
CREATE INDEX idx_directions_opop ON directions(opop_year);

-- ─── 6. Дисциплина ────────────────────────────────────────────────────────
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
    control_form    VARCHAR(20)
                    CHECK (control_form IN ('exam', 'test', 'coursework', 'diff_pass', 'pass')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disciplines_direction ON disciplines(direction_id);

-- ─── 7. PDF-источники ──────────────────────────────────────────────────────
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

-- ─── 8. Версии парсинга ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS parse_versions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_id        UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    version             VARCHAR(50) NOT NULL,          -- "20260604_v1"
    opop_year           INTEGER,                       -- для какой версии OPOP
    total_disciplines   INTEGER NOT NULL DEFAULT 0,
    total_competencies  INTEGER NOT NULL DEFAULT 0,
    total_skills        INTEGER NOT NULL DEFAULT 0,
    total_ksa_items     INTEGER NOT NULL DEFAULT 0,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parse_versions_direction ON parse_versions(direction_id);

-- ─── 9. Компетенция ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS competencies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    code            VARCHAR(20) NOT NULL,              -- "ОПК-2"
    category        VARCHAR(10) NOT NULL
                    CHECK (category IN ('УК', 'ОПК', 'ПК', 'ППК', 'ИП')),
    number          VARCHAR(10) NOT NULL,              -- "1", "2", "3"...
    name            TEXT,                              -- формальное название
    description     TEXT,                              -- подробное описание
    development_level VARCHAR(10)                      -- КС-1 / КС-2 / КС-3
                    CHECK (development_level IN ('КС-1', 'КС-2', 'КС-3')),
    parent_id       UUID REFERENCES competencies(id) ON DELETE CASCADE,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    embedding       VECTOR(768),                           -- pgvector (mean-pool навыков)
    parse_version_id UUID REFERENCES parse_versions(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_competencies_discipline ON competencies(discipline_id);
CREATE INDEX idx_competencies_category ON competencies(category);
CREATE INDEX idx_competencies_parent ON competencies(parent_id);

-- ─── 10. KSA-записи (знания / умения / навыки из RPD) ──────────────────────
CREATE TABLE IF NOT EXISTS ksa_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competency_id   UUID NOT NULL REFERENCES competencies(id) ON DELETE CASCADE,
    ksa_type        ksa_type NOT NULL,
    original_text   TEXT NOT NULL,
    cleaned_text    TEXT,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    parse_version_id UUID REFERENCES parse_versions(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ksa_competency ON ksa_entries(competency_id);
CREATE INDEX idx_ksa_type ON ksa_entries(ksa_type);

-- ─── 11. Таксономия навыков (it_skills + rpd_skills) ───────────────────────
CREATE TABLE IF NOT EXISTS skills (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    name_en     TEXT,
    description TEXT,
    source      VARCHAR(20) NOT NULL DEFAULT 'it_skills'
                CHECK (source IN ('it_skills', 'rpd_skills', 'market')),
    category    VARCHAR(100),
    embedding   VECTOR(768),                           -- pgvector для семантического поиска
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_skills_name ON skills(LOWER(name));
CREATE INDEX idx_skills_source ON skills(source);
CREATE INDEX idx_skills_category ON skills(category);

-- ─── 12. Связь компетенция ↔ навык (с типом KSA) ──────────────────────────
CREATE TABLE IF NOT EXISTS competency_skills (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competency_id     UUID NOT NULL REFERENCES competencies(id) ON DELETE CASCADE,
    skill_id          UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    ksa_type          VARCHAR(20) NOT NULL
                      CHECK (ksa_type IN ('knowledge', 'abilities', 'skills', 'flat')),
    source_text       TEXT,
    match_type        VARCHAR(20) NOT NULL DEFAULT 'fuzzy'
                      CHECK (match_type IN ('exact', 'fuzzy', 'stem')),
    required_level    VARCHAR(10)
                      CHECK (required_level IN ('КС-1', 'КС-2', 'КС-3')),
    parse_version_id  UUID REFERENCES parse_versions(id),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cs_competency ON competency_skills(competency_id);
CREATE INDEX idx_cs_skill ON competency_skills(skill_id);
CREATE UNIQUE INDEX idx_cs_unique
    ON competency_skills(competency_id, skill_id, ksa_type,
        COALESCE(parse_version_id, '00000000-0000-0000-0000-000000000000'));

-- ─── 13. Пользователи (преподаватели, администраторы) ──────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,                     -- crypt(пароль, gen_salt('bf'))
    full_name       TEXT NOT NULL,
    role            VARCHAR(20) NOT NULL DEFAULT 'teacher'
                    CHECK (role IN ('admin', 'teacher')),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- ─── 14. Рекомендации преподавателя ────────────────────────────────────────
-- llm_request_id FK добавляется ALTER TABLE после создания llm_recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    competency_id   UUID REFERENCES competencies(id) ON DELETE SET NULL,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    direction_id    UUID REFERENCES directions(id) ON DELETE CASCADE,
    suggestion      TEXT NOT NULL,
    suggestion_type VARCHAR(20) NOT NULL
                    CHECK (suggestion_type IN ('modify', 'add', 'remove')),
    source          VARCHAR(20)
                    CHECK (source IN ('shap', 'llm', 'llm_fallback')),
    confidence      FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_recommendations_discipline ON recommendations(discipline_id);
CREATE INDEX idx_recommendations_user ON recommendations(user_id);
CREATE INDEX idx_rec_direction ON recommendations(direction_id);

-- ─── 15. Учебные группы ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_groups (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    direction_id    UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
    name            VARCHAR(100) NOT NULL,              -- "ИСИТ-31"
    year            INTEGER NOT NULL,                    -- год поступления
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_student_groups_name ON student_groups(name);
CREATE INDEX idx_student_groups_direction ON student_groups(direction_id);

-- ─── 16. Студенты ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS students (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id        UUID NOT NULL REFERENCES student_groups(id) ON DELETE CASCADE,
    full_name       TEXT NOT NULL,
    email           VARCHAR(255),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_students_group ON students(group_id);

-- ─── 17. Навыки студентов ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS student_skills (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    skill_id        UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    direction_id    UUID REFERENCES directions(id) ON DELETE SET NULL,
    competency_id   UUID REFERENCES competencies(id) ON DELETE SET NULL,
    source          VARCHAR(30) NOT NULL DEFAULT 'self_assessment'
                    CHECK (source IN ('self_assessment', 'auto_extracted', 'expert', 'test')),
    proficiency     REAL NOT NULL DEFAULT 0.0
                    CHECK (proficiency >= 0.0 AND proficiency <= 1.0),
    achieved_level  VARCHAR(10)
                    CHECK (achieved_level IN ('КС-1', 'КС-2', 'КС-3')),
    assessed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_student_skills_student ON student_skills(student_id);
CREATE INDEX idx_student_skills_skill ON student_skills(skill_id);
CREATE UNIQUE INDEX idx_student_skills_unique
    ON student_skills(student_id, skill_id, source);

CREATE INDEX idx_student_skills_proficiency
    ON student_skills(proficiency DESC);

-- ─── 18. Анализ покрытия ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS coverage_analyses (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discipline_id         UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    competency_id         UUID REFERENCES competencies(id) ON DELETE CASCADE,
    direction_id          UUID REFERENCES directions(id) ON DELETE CASCADE,
    total_skills          INTEGER NOT NULL DEFAULT 0,
    market_matched_skills INTEGER NOT NULL DEFAULT 0,
    coverage_ratio        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    analysis_date         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ca_discipline ON coverage_analyses(discipline_id);
CREATE INDEX idx_ca_coverage ON coverage_analyses(coverage_ratio DESC);
CREATE INDEX idx_ca_direction ON coverage_analyses(direction_id);

-- ─── 19. Запуски пайплайна ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action          VARCHAR(50) NOT NULL
                    CHECK (action IN (
                        'full-cycle', 'rebuild', 'train-clusters', 'train-model',
                        'gap-analysis', 'teacher-analysis', 'data-collection'
                    )),
    status          VARCHAR(20) NOT NULL DEFAULT 'started'
                    CHECK (status IN ('started', 'completed', 'failed')),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    stats           JSONB
);

CREATE INDEX idx_pipeline_runs_action ON pipeline_runs(action);
CREATE INDEX idx_pipeline_runs_started ON pipeline_runs(started_at DESC);

-- ─── 20. Результаты анализов ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    analysis_type   VARCHAR(50) NOT NULL
                    CHECK (analysis_type IN ('gap', 'coverage', 'cluster', 'trend', 'teacher-analysis')),
    direction_id    UUID REFERENCES directions(id) ON DELETE CASCADE,
    discipline_id   UUID REFERENCES disciplines(id) ON DELETE CASCADE,
    competency_id   UUID REFERENCES competencies(id) ON DELETE CASCADE,
    data            JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ar_pipeline ON analysis_results(pipeline_run_id);
CREATE INDEX idx_ar_type ON analysis_results(analysis_type);
CREATE INDEX idx_ar_discipline ON analysis_results(discipline_id);
CREATE INDEX idx_ar_direction ON analysis_results(direction_id);

-- ─── 21. Снимки трендов ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trend_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    snapshot_date   DATE NOT NULL,
    skill_freq      JSONB NOT NULL DEFAULT '{}',
    source          VARCHAR(50) NOT NULL DEFAULT 'hh_vacancies',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ts_date ON trend_snapshots(snapshot_date DESC);

-- ─── 22. Сессии пользователей ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash      TEXT NOT NULL,                      -- HMAC-SHA256 хэш токена
    ip_address      VARCHAR(45),                         -- IPv4 или IPv6
    user_agent      TEXT,
    logged_in_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    logged_out_at   TIMESTAMPTZ
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token_hash);
CREATE INDEX idx_sessions_active ON sessions(logged_out_at) WHERE logged_out_at IS NULL;

-- ─── 23. Вакансии ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS vacancies (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hh_id               INTEGER NOT NULL,                    -- hh.ru id
    name                TEXT NOT NULL,                       -- название вакансии
    experience          VARCHAR(50),                         -- "between1and3"
    salary_from         INTEGER,                             -- зарплата от
    salary_to           INTEGER,                             -- зарплата до
    salary_currency     VARCHAR(10),                         -- "RUR"
    employer_name       TEXT,                                -- работодатель
    employer_id         INTEGER,                             -- id работодателя
    area_name           TEXT,                                -- регион
    snippet_requirement TEXT,                                -- требования (фрагмент)
    snippet_responsibility TEXT,                             -- обязанности (фрагмент)
    description         TEXT,                                -- полное описание
    key_skills          JSONB,                               -- ["Python", "SQL", ...]
    parsed_skills       JSONB,                               -- нормализованные навыки (из pipeline)
    published_at        TIMESTAMPTZ,                         -- дата публикации
    alternate_url       TEXT,                                -- ссылка на hh.ru
    pipeline_run_id     UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    raw                 JSONB,                               -- исходный ответ HH API
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_vacancies_hh_id ON vacancies(hh_id);
CREATE INDEX idx_vacancies_published ON vacancies(published_at DESC)
    WHERE parsed_skills IS NOT NULL;
CREATE INDEX idx_vacancies_employer ON vacancies(employer_name);
CREATE INDEX idx_vacancies_pipeline ON vacancies(pipeline_run_id);

-- ─── 24. Логи запросов (бэкенд + фронтенд) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS request_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    method          VARCHAR(10) NOT NULL,
    path            TEXT NOT NULL,
    status          INTEGER NOT NULL DEFAULT 0,
    duration_ms     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    user_email      VARCHAR(255),
    source          VARCHAR(20) NOT NULL DEFAULT 'backend'
                    CHECK (source IN ('backend', 'frontend')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_request_logs_created ON request_logs(created_at DESC);
CREATE INDEX idx_request_logs_user ON request_logs(user_email);
CREATE INDEX idx_request_logs_source ON request_logs(source);

-- ─── 25. Триггеры автообновления updated_at ───────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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

CREATE TRIGGER trg_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ─── 26. Триггер: нормализация имени навыка ───────────────────────────────
CREATE OR REPLACE FUNCTION normalize_skill_name()
RETURNS TRIGGER AS $$
BEGIN
    NEW.name = LOWER(TRIM(NEW.name));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_skills_name_lower
    BEFORE INSERT OR UPDATE ON skills
    FOR EACH ROW EXECUTE FUNCTION normalize_skill_name();

-- ─── 27. Триггер: автосборка кода компетенции ─────────────────────────────
CREATE OR REPLACE FUNCTION auto_build_competency_code()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.code IS NULL OR NEW.code = '' THEN
        NEW.code := NEW.category || '-' || NEW.number;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_competencies_code_auto
    BEFORE INSERT ON competencies
    FOR EACH ROW EXECUTE FUNCTION auto_build_competency_code();

-- ─── 28. Триггер: upsert навыка студента ──────────────────────────────────
-- При повторной вставке (student_id, skill_id, source) обновляет proficiency
CREATE OR REPLACE FUNCTION upsert_student_skill()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM student_skills
        WHERE student_id = NEW.student_id
          AND skill_id = NEW.skill_id
          AND source = NEW.source
    ) THEN
        UPDATE student_skills
        SET proficiency = NEW.proficiency,
            achieved_level = COALESCE(NEW.achieved_level, achieved_level),
            assessed_at = NOW()
        WHERE student_id = NEW.student_id
          AND skill_id = NEW.skill_id
          AND source = NEW.source;
        RETURN NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_student_skills_upsert
    BEFORE INSERT ON student_skills
    FOR EACH ROW EXECUTE FUNCTION upsert_student_skill();

-- ─── 29. LLM-взаимодействия (аудит) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_run_id UUID REFERENCES pipeline_runs(id),
    direction_id UUID REFERENCES directions(id),
    discipline_id UUID REFERENCES disciplines(id),
    prompt_tokens INT DEFAULT 0,
    response_tokens INT DEFAULT 0,
    prompt_hash TEXT,
    response_summary TEXT,
    model VARCHAR(50) DEFAULT 'yandexgpt',
    duration_ms INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_interactions(analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_llm_dir ON llm_interactions(direction_id);

-- ─── 30. Кэш запросов к LLM ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_hash    TEXT NOT NULL,
    prompt_text     TEXT NOT NULL,
    model_used      VARCHAR(20) NOT NULL
                    CHECK (model_used IN ('qwen3.6', 'gemma4', 'qwen_local', 'deepseek_local')),
    response_json   JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_llm_req_hash ON llm_recommendations(request_hash);

COMMENT ON TABLE llm_recommendations IS
    'Кэш и аудит запросов к LLM. request_hash — для dedup повторных запросов.';
COMMENT ON COLUMN llm_recommendations.model_used IS
    'Какая модель ответила: qwen3.6/gemma4 на H100 или qwen_local/deepseek_local (fallback)';
COMMENT ON COLUMN llm_recommendations.response_json IS
    'Полный JSON-ответ LLM (рекомендации, confidence, объяснения)';

-- ─── 31. Связь рекомендаций с LLM-запросами (FK, отложенная из-за цикла) ──
ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS llm_request_id UUID
        REFERENCES llm_recommendations(id) ON DELETE SET NULL;

-- ─── 32. История оценок профиля ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS profile_evaluations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID REFERENCES users(id) ON DELETE SET NULL,
    discipline_id     UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
    evaluation_type   VARCHAR(20) NOT NULL
                      CHECK (evaluation_type IN ('gap', 'coverage', 'full')),
    input_summary     JSONB NOT NULL DEFAULT '{}',
    result_summary    JSONB NOT NULL DEFAULT '{}',
    llm_request_id    UUID REFERENCES llm_recommendations(id) ON DELETE SET NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_prof_eval_user ON profile_evaluations(user_id);
CREATE INDEX idx_prof_eval_discipline ON profile_evaluations(discipline_id);

COMMENT ON TABLE profile_evaluations IS
    'История оценок профиля: gap, coverage или полный анализ с привязкой к LLM-запросу';

-- ─── 33. Начальные данные ────────────────────────────────────────────────
-- Направления (используется ON CONFLICT на (code, profile))
INSERT INTO directions (code, name, profile, supervisor, opop_year) VALUES
    ('09.03.02', 'Перспективные информационные технологии',
     'Перспективные информационные технологии', 'Хусаинов Н.Ш.', 2025),
    ('01.03.01', 'Математика, механика и математическое моделирование',
     'Математика, механика и математическое моделирование', 'Карапетянц А.Н.', 2025),
    ('01.03.02', 'Математическое моделирование и искусственный интеллект',
     'Математическое моделирование и искусственный интеллект', 'Махно В.В.', 2025),
    ('09.03.04', 'Методы и средства разработки программного обеспечения',
     'Методы и средства разработки программного обеспечения', 'Хусаинов Н.Ш.', 2025),
    ('09.03.01', 'Программирование и системная интеграция ИТ-решений',
     'Программирование и системная интеграция ИТ-решений', 'Хусаинов Н.Ш.', 2025),
    ('09.03.01', 'Технологии искусственного интеллекта',
     'Технологии искусственного интеллекта', 'Хусаинов Н.Ш.', 2025),
    ('02.03.02', 'Фундаментальная информатика и информационные технологии',
     'Фундаментальная информатика и информационные технологии', 'Михалкович С.С.', 2025)
ON CONFLICT (code, profile) DO UPDATE SET
    name = EXCLUDED.name,
    supervisor = EXCLUDED.supervisor;

-- Пользователи по умолчанию (пароль через pgcrypt bcrypt)
INSERT INTO users (email, password_hash, full_name, role) VALUES
    ('admin@compare-competencies.local', crypt('admin', gen_salt('bf')), 'Администратор', 'admin'),
    ('teacher@compare-competencies.local', crypt('teacher123', gen_salt('bf')), 'Преподаватель', 'teacher')
ON CONFLICT (email) DO NOTHING;

INSERT INTO users (email, password_hash, full_name, role) VALUES
    ('karapetyants@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Карапетянц Алексей Николаевич', 'teacher'),
    ('vvmakhno@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Махно Виктория Викторовна', 'teacher'),
    ('khusainov@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Хусаинов Наиль Шавяктович', 'teacher'),
    ('miks@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Михалкович Станислав Станиславович', 'teacher'),
    ('asviridov@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Свиридов Александр Славьевич', 'teacher'),
    ('skucherov@sfedu.ru', crypt('teacher123', gen_salt('bf')), 'Кучеров Сергей Александрович', 'teacher')
ON CONFLICT (email) DO UPDATE SET
    full_name = EXCLUDED.full_name,
    role = 'teacher',
    is_active = TRUE;

-- Удаление не-IT дисциплин (общеобразовательные, не влияющие на анализ рынка)
DELETE FROM disciplines WHERE name IN (
    'Безопасность жизнедеятельности',
    'Дисциплины по ФКиС',
    'История России',
    'Иностранный язык (англ. яз., уровень А1)',
    'Иностранный язык (англ. яз., уровень А2)',
    'Иностранный язык (англ. яз., уровень В1)',
    'Иностранный язык (англ. яз., уровень В2)',
    'Иностранный язык (англ. яз., уровень С1)',
    'Иностранный язык (русский язык)',
    'Иностранный язык для деловой коммуникации',
    'Воображение, изображение, реальность',
    'Основы проектной деятельности',
    'Практикум по подготовке инженерной документации'
);

-- =============================================================================
-- Готово. Проверка:
--   SELECT table_name FROM information_schema.tables
--   WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
--   ORDER BY table_name;
-- Ожидается 25 таблиц (всего).
-- Добавлены: llm_recommendations, profile_evaluations (из sql/004_llm_recommendations.sql)
-- Изменены: directions (supervisor + composite UNIQUE), recommendations (4 cols),
-- student_skills (2 cols), pipeline_runs (user_id + expanded CHECK),
-- analysis_results (direction_id + teacher-analysis), coverage_analyses (direction_id),
-- vacancies (parsed_skills).
-- Исключены: market_skill_mappings (dead, дропнута 006_market_skill_mappings_drop.sql).
-- =============================================================================
