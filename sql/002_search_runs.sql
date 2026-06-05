-- Search run tracking: link market data to pipeline parameters

ALTER TABLE market_skill_mappings
  ADD COLUMN IF NOT EXISTS pipeline_run_id UUID REFERENCES pipeline_runs(id)
  ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_msm_pipeline_run
  ON market_skill_mappings (pipeline_run_id);
