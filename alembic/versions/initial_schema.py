"""Initial schema — replicates sql/001_create_krm_schema.sql.

Extensions, ENUM, functions, 21 tables, indexes, triggers, seed data.
All operations use IF NOT EXISTS / CREATE OR REPLACE / ON CONFLICT DO NOTHING.
Triggers are wrapped in DO $$ EXCEPTION blocks for idempotent re-runs.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Extensions ──────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ─── ENUM ksa_type ───────────────────────────────────────────────────────
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE ksa_type AS ENUM ('knowledge', 'abilities', 'skills');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
    """)

    # ─── Functions ───────────────────────────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION check_password(
            input_password TEXT, stored_hash TEXT
        ) RETURNS BOOLEAN AS $$
        BEGIN
            RETURN crypt(input_password, stored_hash) = stored_hash;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION normalize_skill_name()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.name = LOWER(TRIM(NEW.name));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE OR REPLACE FUNCTION auto_build_competency_code()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.code IS NULL OR NEW.code = '' THEN
                NEW.code := NEW.category || '-' || NEW.number;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
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
    """)

    # ─── directions ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS directions (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            code        VARCHAR(20) UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            profile     TEXT,
            opop_year   INTEGER,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_directions_code ON directions(code)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_directions_opop ON directions(opop_year)")

    # ─── disciplines ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS disciplines (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            direction_id    UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            name_en         TEXT,
            description     TEXT,
            semester        INTEGER,
            hours_total     INTEGER,
            hours_lecture   INTEGER,
            hours_practice  INTEGER,
            hours_lab       INTEGER,
            hours_self      INTEGER,
            control_form    VARCHAR(20)
                CHECK (control_form IN ('exam', 'test', 'coursework', 'diff_pass', 'pass')),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_disciplines_direction ON disciplines(direction_id)")

    # ─── pdf_sources ─────────────────────────────────────────────────────────
    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pdf_sources_discipline ON pdf_sources(discipline_id)")

    # ─── parse_versions ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS parse_versions (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            direction_id        UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
            version             VARCHAR(50) NOT NULL,
            opop_year           INTEGER,
            total_disciplines   INTEGER NOT NULL DEFAULT 0,
            total_competencies  INTEGER NOT NULL DEFAULT 0,
            total_skills        INTEGER NOT NULL DEFAULT 0,
            total_ksa_items     INTEGER NOT NULL DEFAULT 0,
            notes               TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_parse_versions_direction ON parse_versions(direction_id)")

    # ─── competencies ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS competencies (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
            code            VARCHAR(20) NOT NULL,
            category        VARCHAR(10) NOT NULL
                CHECK (category IN ('УК', 'ОПК', 'ПК', 'ППК', 'ИП')),
            number          VARCHAR(10) NOT NULL,
            name            TEXT,
            description     TEXT,
            development_level VARCHAR(10)
                CHECK (development_level IN ('КС-1', 'КС-2', 'КС-3')),
            parent_id       UUID REFERENCES competencies(id) ON DELETE CASCADE,
            sort_order      INTEGER NOT NULL DEFAULT 0,
            embedding       VECTOR(768),
            parse_version_id UUID REFERENCES parse_versions(id),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_competencies_discipline ON competencies(discipline_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_competencies_category ON competencies(category)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_competencies_parent ON competencies(parent_id)")

    # ─── ksa_entries ─────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS ksa_entries (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            competency_id   UUID NOT NULL REFERENCES competencies(id) ON DELETE CASCADE,
            ksa_type        ksa_type NOT NULL,
            original_text   TEXT NOT NULL,
            cleaned_text    TEXT,
            sort_order      INTEGER NOT NULL DEFAULT 0,
            parse_version_id UUID REFERENCES parse_versions(id),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ksa_competency ON ksa_entries(competency_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ksa_type ON ksa_entries(ksa_type)")

    # ─── skills ──────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT NOT NULL,
            name_en     TEXT,
            description TEXT,
            source      VARCHAR(20) NOT NULL DEFAULT 'it_skills'
                CHECK (source IN ('it_skills', 'rpd_skills', 'market')),
            category    VARCHAR(100),
            embedding   VECTOR(768),
            is_active   BOOLEAN NOT NULL DEFAULT TRUE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_skills_name ON skills(LOWER(name))")
    op.execute("CREATE INDEX IF NOT EXISTS idx_skills_source ON skills(source)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category)")

    # ─── competency_skills ───────────────────────────────────────────────────
    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_cs_competency ON competency_skills(competency_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_cs_skill ON competency_skills(skill_id)")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cs_unique
        ON competency_skills(competency_id, skill_id, ksa_type,
            COALESCE(parse_version_id, '00000000-0000-0000-0000-000000000000'))
    """)

    # ─── users ───────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email           VARCHAR(255) UNIQUE NOT NULL,
            password_hash   TEXT NOT NULL,
            full_name       TEXT NOT NULL,
            role            VARCHAR(20) NOT NULL DEFAULT 'teacher'
                CHECK (role IN ('admin', 'teacher')),
            is_active       BOOLEAN NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")

    # ─── recommendations ─────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            discipline_id   UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
            competency_id   UUID REFERENCES competencies(id) ON DELETE SET NULL,
            user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
            suggestion      TEXT NOT NULL,
            suggestion_type VARCHAR(20) NOT NULL
                CHECK (suggestion_type IN ('modify', 'add', 'remove')),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_discipline ON recommendations(discipline_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)")

    # ─── student_groups ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS student_groups (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            direction_id    UUID NOT NULL REFERENCES directions(id) ON DELETE CASCADE,
            name            VARCHAR(100) NOT NULL,
            year            INTEGER NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_student_groups_name ON student_groups(name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_student_groups_direction ON student_groups(direction_id)")

    # ─── students ────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            group_id        UUID NOT NULL REFERENCES student_groups(id) ON DELETE CASCADE,
            full_name       TEXT NOT NULL,
            email           VARCHAR(255),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_students_group ON students(group_id)")

    # ─── student_skills ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS student_skills (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            student_id      UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            skill_id        UUID NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
            source          VARCHAR(30) NOT NULL DEFAULT 'self_assessment'
                CHECK (source IN ('self_assessment', 'auto_extracted', 'expert', 'test')),
            proficiency     REAL NOT NULL DEFAULT 0.0
                CHECK (proficiency >= 0.0 AND proficiency <= 1.0),
            achieved_level  VARCHAR(10)
                CHECK (achieved_level IN ('КС-1', 'КС-2', 'КС-3')),
            assessed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_student_skills_student ON student_skills(student_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_student_skills_skill ON student_skills(skill_id)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_student_skills_unique ON student_skills(student_id, skill_id, source)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_student_skills_proficiency ON student_skills(proficiency DESC)")

    # ─── coverage_analyses ───────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS coverage_analyses (
            id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            discipline_id         UUID NOT NULL REFERENCES disciplines(id) ON DELETE CASCADE,
            competency_id         UUID REFERENCES competencies(id) ON DELETE CASCADE,
            total_skills          INTEGER NOT NULL DEFAULT 0,
            market_matched_skills INTEGER NOT NULL DEFAULT 0,
            coverage_ratio        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            analysis_date         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ca_discipline ON coverage_analyses(discipline_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ca_coverage ON coverage_analyses(coverage_ratio DESC)")

    # ─── pipeline_runs ───────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            action          VARCHAR(50) NOT NULL,
            status          VARCHAR(20) NOT NULL DEFAULT 'started'
                CHECK (status IN ('started', 'completed', 'failed')),
            started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at    TIMESTAMPTZ,
            error_message   TEXT,
            stats           JSONB
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_action ON pipeline_runs(action)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started ON pipeline_runs(started_at DESC)")

    # ─── analysis_results ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
            analysis_type   VARCHAR(50) NOT NULL
                CHECK (analysis_type IN ('gap', 'coverage', 'cluster', 'trend', 'teacher')),
            discipline_id   UUID REFERENCES disciplines(id) ON DELETE CASCADE,
            competency_id   UUID REFERENCES competencies(id) ON DELETE CASCADE,
            data            JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ar_pipeline ON analysis_results(pipeline_run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ar_type ON analysis_results(analysis_type)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_ar_discipline ON analysis_results(discipline_id)")

    # ─── trend_snapshots ─────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS trend_snapshots (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pipeline_run_id UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
            snapshot_date   DATE NOT NULL,
            skill_freq      JSONB NOT NULL DEFAULT '{}',
            source          VARCHAR(50) NOT NULL DEFAULT 'hh_vacancies',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_ts_date ON trend_snapshots(snapshot_date DESC)")

    # ─── sessions ────────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash      TEXT NOT NULL,
            ip_address      VARCHAR(45),
            user_agent      TEXT,
            logged_in_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_activity   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            logged_out_at   TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(logged_out_at) WHERE logged_out_at IS NULL")

    # ─── vacancies ───────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS vacancies (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            hh_id               INTEGER NOT NULL,
            name                TEXT NOT NULL,
            experience          VARCHAR(50),
            salary_from         INTEGER,
            salary_to           INTEGER,
            salary_currency     VARCHAR(10),
            employer_name       TEXT,
            employer_id         INTEGER,
            area_name           TEXT,
            snippet_requirement TEXT,
            snippet_responsibility TEXT,
            description         TEXT,
            key_skills          JSONB,
            published_at        TIMESTAMPTZ,
            alternate_url       TEXT,
            pipeline_run_id     UUID REFERENCES pipeline_runs(id) ON DELETE SET NULL,
            raw                 JSONB,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vacancies_hh_id ON vacancies(hh_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vacancies_published ON vacancies(published_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vacancies_employer ON vacancies(employer_name)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_vacancies_pipeline ON vacancies(pipeline_run_id)")

    # ─── request_logs ────────────────────────────────────────────────────────
    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_user ON request_logs(user_email)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_source ON request_logs(source)")

    # ─── llm_interactions ────────────────────────────────────────────────────
    op.execute("""
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
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_interactions(analysis_run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_dir ON llm_interactions(direction_id)")

    # ─── Triggers ────────────────────────────────────────────────────────────
    trigger_pairs = [
        ("trg_directions_updated", "directions", "update_updated_at()"),
        ("trg_disciplines_updated", "disciplines", "update_updated_at()"),
        ("trg_competencies_updated", "competencies", "update_updated_at()"),
        ("trg_skills_updated", "skills", "update_updated_at()"),
        ("trg_users_updated", "users", "update_updated_at()"),
        ("trg_skills_name_lower", "skills", "normalize_skill_name()"),
        ("trg_competencies_code_auto", "competencies", "auto_build_competency_code()"),
        ("trg_student_skills_upsert", "student_skills", "upsert_student_skill()"),
    ]
    for name, table, func in trigger_pairs:
        is_before = "BEFORE INSERT OR UPDATE" if "updated" in name or "name_lower" in name else "BEFORE INSERT"
        # upsert trigger is BEFORE INSERT
        actual_mode = "BEFORE INSERT OR UPDATE" if "updated" in name or "name_lower" in name else "BEFORE INSERT"
        if "upsert" in name:
            actual_mode = "BEFORE INSERT"
        op.execute(f"""
            DO $$ BEGIN
                CREATE TRIGGER {name}
                    {actual_mode} ON {table}
                    FOR EACH ROW EXECUTE FUNCTION {func};
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$;
        """)

    # ─── Seed data ───────────────────────────────────────────────────────────
    op.execute("""
        INSERT INTO directions (code, name, profile, opop_year)
        VALUES (
            '09.03.02',
            'Информационные системы и технологии',
            'Перспективные информационные технологии',
            2025
        ) ON CONFLICT (code) DO NOTHING;
    """)
    op.execute("""
        INSERT INTO users (email, password_hash, full_name, role) VALUES
            ('admin@compare-competencies.local', crypt('admin', gen_salt('bf')), 'Администратор', 'admin'),
            ('teacher@compare-competencies.local', crypt('prepod', gen_salt('bf')), 'Преподаватель', 'teacher'),
            ('student@compare-competencies.local', crypt('student', gen_salt('bf')), 'Студент', 'teacher')
        ON CONFLICT (email) DO NOTHING;
    """)


def downgrade() -> None:
    tables = [
        "llm_interactions", "request_logs", "vacancies", "sessions",
        "trend_snapshots", "analysis_results", "pipeline_runs",
        "coverage_analyses", "student_skills", "students", "student_groups",
        "recommendations", "users", "competency_skills", "skills",
        "ksa_entries", "competencies", "parse_versions", "pdf_sources",
        "disciplines", "directions",
    ]
    for t in tables:
        op.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    op.execute("DROP TYPE IF EXISTS ksa_type")
