"""SHAP top-k: строки shap_values выровнены с возвращёнными impacts.

До фикса раннер индексировал shap_vals позицией в ОТСОРТИРОВАННОМ топ-15,
а строки шли в порядке valid_missing — persisted shap_per_skill был перепутан.
Плюс SHAP считался на всех ~200 кандидатах вместо показанных 15.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

sys.modules.setdefault("shap", MagicMock())
sys.modules.setdefault("cv2", MagicMock())

from tests.predictors.test_ltr_recommendation_engine import (  # noqa: E402,F811
    engine_with_mocks,  # noqa: F401,F811  (re-export fixture)
    mock_embedding_model,  # noqa: F401
    mock_level_analyzer,  # noqa: F401
    mock_skill_filter,  # noqa: F401
    mock_vacancy_parser,  # noqa: F401
)


def test_shap_rows_align_with_returned_impacts(engine_with_mocks):  # noqa: F811
    engine = engine_with_mocks
    engine.is_fitted = True
    engine.model = MagicMock()
    # docker явно выше sql
    engine.model.predict.return_value = np.array([0.2, 0.9])
    seen = {}

    def fake_shap_values(frame):
        seen["n_rows"] = len(frame)
        # строка i помечена значением i — проверяем порядок
        return np.array([[float(i)] * len(engine.feature_names)
                          for i in range(len(frame))])

    with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer") as mock_exp:
        mock_exp.return_value.shap_values.side_effect = fake_shap_values
        result = engine.predict_skill_impact_with_shap(
            ["python"], ["sql", "docker"], compute_shap=True)
    assert result.is_ok()
    recs, shap_vals, xout = result.unwrap()
    assert [r[0] for r in recs] == ["docker", "sql"]
    # SHAP посчитан только на показанных строках, в порядке impacts
    assert seen["n_rows"] == 2
    assert shap_vals is not None and len(shap_vals) == 2
    # первая строка SHAP соответствует первому impact (docker)
    assert float(shap_vals[0][0]) == 0.0
    assert len(xout) == 2 and list(xout.index) == [1, 0]


def test_shap_topk_limits_rows(engine_with_mocks):  # noqa: F811
    engine = engine_with_mocks
    engine.is_fitted = True
    engine.model = MagicMock()
    engine.model.predict.return_value = np.array([0.2, 0.9])
    seen = {}

    def fake2(frame):
        seen["n_rows"] = len(frame)
        return np.zeros((len(frame), 1))

    with patch("src.predictors.ltr_recommendation_engine.shap.TreeExplainer") as mock_exp:
        mock_exp.return_value.shap_values.side_effect = fake2
        result = engine.predict_skill_impact_with_shap(
            ["python"], ["sql", "docker"], compute_shap=True, shap_top_k=1)
    assert result.is_ok()
    recs, shap_vals, _x = result.unwrap()
    assert seen["n_rows"] == 1
    assert len(shap_vals) == 1
    assert recs[0][0] == "docker"
