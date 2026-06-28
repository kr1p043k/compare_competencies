-- SQL migration: populate competency hierarchy parent_id from dot notation
-- Run after 001_create_krm_schema.sql on existing data.
-- Creates parent competency records for dotted children if missing,
-- then links children to parents via parent_id.

DO $$ BEGIN

-- Step 1: Create parent competency records for dotted competencies
-- that don't have a parent yet (e.g. ОПК-2.1 needs parent ОПК-2)
INSERT INTO competencies (discipline_id, code, category, number, name, description, sort_order)
SELECT DISTINCT
    c.discipline_id,
    LEFT(c.code, POSITION('.' IN c.code) - 1) AS parent_code,
    LEFT(c.code, POSITION('-' IN c.code) - 1) AS category,
    SUBSTRING(c.code FROM POSITION('-' IN c.code) + 1 FOR POSITION('.' IN c.code) - POSITION('-' IN c.code) - 1) AS number,
    'Группа ' || LEFT(c.code, POSITION('.' IN c.code) - 1) AS name,
    'Автоматически созданный родитель для компетенций ' || c.code,
    0
FROM competencies c
WHERE c.code LIKE '%-%.%'
  AND c.parent_id IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM competencies p
      WHERE p.code = LEFT(c.code, POSITION('.' IN c.code) - 1)
        AND p.discipline_id = c.discipline_id
  );

-- Step 2: Link children to parents where parent exists
UPDATE competencies c
SET parent_id = p.id
FROM competencies p
WHERE c.parent_id IS NULL
  AND c.code LIKE '%-%.%'
  AND p.code = LEFT(c.code, POSITION('.' IN c.code) - 1)
  AND p.discipline_id = c.discipline_id;

END $$;
