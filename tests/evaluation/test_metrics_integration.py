from src.evaluation.metrics import RetrievalMetrics, ClassificationMetrics
import pytest


def test_no_regression_retrieval():
    rel = {"python", "sql", "docker"}
    ret = ["python", "kubernetes", "sql", "aws", "docker"]
    r = RetrievalMetrics.report(rel, ret, ks=[1, 3, 5])
    assert r["P@1"] == 1.0
    assert r["R@3"] == pytest.approx(2 / 3)
    assert r["P@5"] == pytest.approx(3 / 5)
    assert r["AP"] > 0
    assert r["MRR"] == 1.0


def test_no_regression_classification():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]
    r = ClassificationMetrics.report(y_true, y_pred)
    assert r["accuracy"] == pytest.approx(4 / 5)
    assert r["precision"] == pytest.approx(2 / 2)
    assert r["recall"] == pytest.approx(2 / 3)
