-- =============================================================================
-- LLM Integration — PostgreSQL
-- Проект: compare_competencies
-- Содержит: кэш LLM-запросов, расширение recommendations, история оценок
--
-- Создано: 2026-06-07
-- Использование: выполнить после 001_create_krm_schema.sql
-- =============================================================================

-- ─── 1. Кэш запросов к LLM ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS llm_recommendations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_hash    TEXT NOT NULL,
    prompt_text     TEXT NOT NULL,
    model_used      VARCHAR(20) NOT NULL
                    CHECK (model_used IN (
                        'qwen3.6', 'gemma4',
                        'qwen_local', 'deepseek_local'
                    )),
    response_json   JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_req_hash ON llm_recommendations(request_hash);

COMMENT ON TABLE llm_recommendations IS
    'Кэш и аудит запросов к LLM. request_hash — для dedup повторных запросов.';
COMMENT ON COLUMN llm_recommendations.model_used IS
    'Какая модель ответила: qwen3.6/gemma4 на H100 или qwen_local/deepseek_local (fallback)';
COMMENT ON COLUMN llm_recommendations.response_json IS
    'Полный JSON-ответ LLM (рекомендации, confidence, объяснения)';

-- ─── 2. Расширение таблицы recommendations ───────────────────────────────────

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS source         VARCHAR(20)
        CHECK (source IN ('shap', 'llm', 'llm_fallback'));

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS llm_request_id UUID
        REFERENCES llm_recommendations(id) ON DELETE SET NULL;

ALTER TABLE recommendations
    ADD COLUMN IF NOT EXISTS confidence     FLOAT
        CHECK (confidence >= 0 AND confidence <= 1);

COMMENT ON COLUMN recommendations.source IS
    'Откуда получена рекомендация: shap (локально), llm (H100), llm_fallback (локальная LLM)';
COMMENT ON COLUMN recommendations.llm_request_id IS
    'Ссылка на исходный запрос к LLM (если источник — LLM)';
COMMENT ON COLUMN recommendations.confidence IS
    'Уверенность рекомендации (0..1). Для SHAP — вклад признака, для LLM — score из ответа';

-- ─── 3. История оценок профиля ──────────────────────────────────────────────

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

CREATE INDEX IF NOT EXISTS idx_prof_eval_user
    ON profile_evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_prof_eval_discipline
    ON profile_evaluations(discipline_id);

COMMENT ON TABLE profile_evaluations IS
    'История оценок профиля: gap, coverage или полный анализ с привязкой к LLM-запросу';
