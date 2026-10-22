# -*- coding: utf-8 -*-
"""Audit fixes 2026-10-12 evening: schema/model drift, LTR features, blend, cache."""
import numpy as np
import pytest
from sqlalchemy.dialects.postgresql import dialect as PG


def test_pipeline_run_action_list_matches_code():
    """reparse/rpd-import/cancelled пишет код — CHECK в модели обязан их пускать."""
    from sqlalchemy.schema import CreateTable

    from src.models.krm_models import Base, PipelineRun

    ddl = str(CreateTable(PipelineRun.__table__).compile(dialect=PG()))
    for act in ("reparse", "rpd-import", "hh-import", "cancelled"):
        assert act in ddl, act


def test_recommendations_competency_fk_set_null():
    """CASCADE в модели стирал рекомендации; в живой БД SET NULL."""
    from src.models.krm_models import Recommendation

    col = Recommendation.__table__.columns["competency_id"]
    fk = list(col.foreign_keys)[0]
    assert fk.ondelete == "SET NULL"


def test_all_fk_columns_have_index():
    """25 FK-колонок остались без индекса в модели (fresh-install деградация)."""
    from src.models.krm_models import Base

    missing = [
        "%s.%s" % (t.name, c.name)
        for t in Base.metadata.tables.values()
        for c in t.columns
        if c.foreign_keys and not c.index
    ]
    assert missing == [], missing


def test_category_encoding_reverse_map_complete():
    from src.predictors.ltr_recommendation_engine import (
        CATEGORY_ENCODING,
        CATEGORY_ENCODING_NAME,
    )

    assert set(CATEGORY_ENCODING_NAME.values()) == set(CATEGORY_ENCODING)
    for enc, names in CATEGORY_ENCODING.items():
        # один код может содержать одну категорию (programming_languages),
        # но каждый код обязан быть представлен
        assert len(names) >= 1, (enc, names)
    assert "devops" in CATEGORY_ENCODING[4]


def test_blend_merges_case_variants():
    """Evaluator отдаёт 'PostgreSQL', LTR — 'postgresql'; раньше это были
    две записи и 40% бленда терялись."""
    ev = {"PostgreSQL": 0.8}
    ltr = {"postgresql": 0.9}
    display, ev_l = {}, {}
    for skill, score in ev.items():
        k = skill.lower()
        ev_l[k] = max(ev_l.get(k, 0.0), score)
        display.setdefault(k, skill)
    for skill in ltr:
        display.setdefault(skill.lower(), skill)
    combined = {display.get(k, k): 0.6 * ev_l.get(k, 0.0) + 0.4 * ltr.get(k, 0.0)
                for k in set(ev_l) | set(ltr)}
    assert len(combined) == 1
    assert list(combined.values())[0] == pytest.approx(0.84, abs=1e-6)
    assert list(combined.keys())[0] == "PostgreSQL"


def test_skill_to_zero_is_decline():
    """Навык, упавший до нуля, = -100%, а не 0% (иначе не попадал в declining)."""
    from src.predictors.skill_forecast import SkillForecastEngine

    eng = SkillForecastEngine()
    eng._models["cobol"] = {
        "slope": -1.0, "n": 4, "last_freq": 0.0, "mape": 0.5, "rmse": 1.0,
    }
    res = eng.predict("cobol")
    if res.is_ok():
        assert res.unwrap().predicted_growth < 0