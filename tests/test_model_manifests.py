"""Проверка консистентности манифестов моделей."""
from __future__ import annotations

from scripts.check_model_manifests import validate

RUNTIME = ("a", "b", "c")
INCOMPAT = {"legacy10.joblib": "10-признаковый feature-space"}


def test_ok_when_features_and_metric_match():
    problems = validate("good.joblib", RUNTIME, {"metrics": {"ndcg@10": 0.5}}, RUNTIME, INCOMPAT)
    assert problems == []


def test_feature_mismatch_flagged():
    problems = validate("m.joblib", ("a", "b"), {"metrics": {"human_ndcg_at_10": 0.4}}, RUNTIME, INCOMPAT)
    assert any("feature_names не совпадают" in p for p in problems)


def test_missing_feature_names_flagged():
    problems = validate("m.joblib", None, {"metrics": {"ndcg_at_10": 0.3}}, RUNTIME, INCOMPAT)
    assert any("нет feature_names" in p for p in problems)


def test_missing_ndcg_flagged():
    problems = validate("m.joblib", RUNTIME, {"metrics": {}}, RUNTIME, INCOMPAT)
    assert any("NDCG" in p for p in problems)


def test_incompatible_model_reports_reason():
    problems = validate("legacy10.joblib", RUNTIME, {"metrics": {"ndcg@10": 0.9}}, RUNTIME, INCOMPAT)
    assert problems == ["в списке несовместимых: 10-признаковый feature-space"]


def test_human_ndcg_key_accepted():
    assert validate("m.joblib", RUNTIME, {"metrics": {"human_ndcg_at_10": 0.6}}, RUNTIME, INCOMPAT) == []
