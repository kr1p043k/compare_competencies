-- =============================================================================
-- Vacancies table — 2026-06-07
-- Переносит сырые вакансии из JSON-файлов в БД для исторического анализа.
-- =============================================================================

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
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_vacancies_hh_id ON vacancies(hh_id);
CREATE INDEX IF NOT EXISTS idx_vacancies_published ON vacancies(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_vacancies_employer ON vacancies(employer_name);
CREATE INDEX IF NOT EXISTS idx_vacancies_pipeline ON vacancies(pipeline_run_id);
