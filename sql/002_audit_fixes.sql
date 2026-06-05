-- =============================================================================
-- DB Schema Audit Fixes — 2026-06-05
-- Добавляет direction_id, расширяет CHECK-констрейнты, нормализует связи
-- =============================================================================

-- ─── 1. analysis_results: добавить direction_id + расширить type ──────────
ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE,
    DROP CONSTRAINT IF EXISTS analysis_results_analysis_type_check,
    ADD CONSTRAINT analysis_results_analysis_type_check
        CHECK (analysis_type IN ('gap', 'coverage', 'cluster', 'trend', 'teacher-analysis'));

CREATE INDEX IF NOT EXISTS idx_ar_direction ON analysis_results(direction_id);

-- ─── 2. pipeline_runs: расширить action ───────────────────────────────────
ALTER TABLE pipeline_runs
    DROP CONSTRAINT IF EXISTS pipeline_runs_action_check,
    ADD CONSTRAINT pipeline_runs_action_check
        CHECK (action IN (
            'full-cycle', 'rebuild', 'train-clusters', 'train-model',
            'gap-analysis', 'teacher-analysis', 'data-collection'
        ));

-- ─── 3. market_skill_mappings: связь с направлением ───────────────────────
ALTER TABLE market_skill_mappings
    ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_msm_direction ON market_skill_mappings(direction_id);

-- ─── 4. student_skills: аналитические поля ────────────────────────────────
ALTER TABLE student_skills
    ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS competency_id UUID REFERENCES competencies(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_ss_direction ON student_skills(direction_id);

-- ─── 5. coverage_analyses: добавить direction_id ──────────────────────────
ALTER TABLE coverage_analyses
    ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_ca_direction ON coverage_analyses(direction_id);

-- ─── 6. recommendations: добавить direction_id ────────────────────────────
ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS direction_id UUID REFERENCES directions(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_rec_direction ON recommendations(direction_id);
