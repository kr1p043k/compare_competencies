# tests/analyzers/comparison/test_engines.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src import Err, Ok, Result
from src.analyzers.comparison.engines import JaccardEngine, EnsembleEngine, BM25Engine, SimilarityEngine


def _unwrap(res: Result) -> dict:
    match res:
        case Ok(d):
            return d
        case _:
            raise AssertionError(f"Expected Ok, got {res}")


class TestJaccardEngine:
    def test_init_default_threshold(self):
        e = JaccardEngine()
        assert e.threshold == 0.6

    def test_compare_exact_match(self):
        e = JaccardEngine()
        res = _unwrap(e.compare(["python developer"], ["python developer"]))
        assert res["score"] == 1.0
        assert len(res["matches"]) == 1

    def test_compare_fuzzy_match(self):
        e = JaccardEngine(threshold=0.5)
        res = _unwrap(e.compare(["python dev"], ["python developer"]))
        assert res["score"] > 0.5
        assert len(res["matches"]) == 1

    def test_compare_no_match_below_threshold(self):
        e = JaccardEngine(threshold=0.9)
        res = _unwrap(e.compare(["python"], ["golang"]))
        assert res["score"] == 0.0
        assert res["matches"] == []

    def test_compare_multiple_skills(self):
        e = JaccardEngine(threshold=0.3)
        res = _unwrap(e.compare(
            ["python", "sql"],
            ["python developer", "sql database", "golang"],
        ))
        assert len(res["matches"]) >= 2
        assert any("python" in m["skill"] for m in res["matches"])
        assert any("sql" in m["skill"] for m in res["matches"])

    def test_compare_empty_student_skills(self):
        e = JaccardEngine()
        res = _unwrap(e.compare([], ["python"]))
        assert res["score"] == 0.0

    def test_compare_empty_market_skills(self):
        e = JaccardEngine()
        res = _unwrap(e.compare(["python"], []))
        assert res["score"] == 0.0

    def test_compare_limits_to_15_matches(self):
        e = JaccardEngine(threshold=0.0)
        market = [f"skill_{i}" for i in range(30)]
        res = _unwrap(e.compare(["skill"], market))
        assert len(res["matches"]) <= 15

    def test_compare_result_keys(self):
        e = JaccardEngine()
        res = _unwrap(e.compare(["python"], ["python"]))
        assert "score" in res
        assert "weighted_coverage" in res
        assert "avg_similarity" in res
        assert "matches" in res
        assert "missing" in res


class TestBM25Engine:
    def test_fit_and_compare(self):
        e = BM25Engine()
        e.fit(["python developer", "sql database", "machine learning"])
        res = _unwrap(e.compare(["python", "sql"], ["python developer", "sql database"]))
        assert res["score"] > 0
        assert len(res["matches"]) == 2

    def test_compare_without_fit_returns_zero(self):
        e = BM25Engine()
        res = _unwrap(e.compare(["python"], ["python developer"]))
        assert res["score"] == 0.0

    def test_compare_empty_student(self):
        e = BM25Engine()
        e.fit(["python"])
        res = _unwrap(e.compare([], ["python"]))
        assert res["score"] == 0.0

    def test_compare_no_query_tokens(self):
        e = BM25Engine()
        e.fit(["python"])
        res = _unwrap(e.compare([""], ["python"]))
        assert res["score"] == 0.0

    def test_compare_no_matches(self):
        e = BM25Engine()
        e.fit(["golang", "rust"])
        res = _unwrap(e.compare(["python"], ["golang"]))
        assert "score" in res

    def test_normalization(self):
        e = BM25Engine()
        e.fit(["python", "python advanced", "python expert"])
        res = _unwrap(e.compare(["python expert"], ["python expert"]))
        assert res["matches"][0]["similarity"] == 1.0

    def test_result_keys(self):
        e = BM25Engine()
        e.fit(["python"])
        res = _unwrap(e.compare(["python"], ["python"]))
        assert "score" in res
        assert "matches" in res
        assert "missing" in res


class TestEnsembleEngine:
    def test_init_empty_engines_raises(self):
        with pytest.raises(ValueError, match="At least one engine required"):
            EnsembleEngine({})

    def test_compare_single_engine(self):
        mock_engine = MagicMock(spec=SimilarityEngine)
        mock_engine.compare.return_value = Ok({
            "score": 0.8, "matches": [{"skill": "py", "similarity": 0.8}],
            "weighted_coverage": 0.8, "avg_similarity": 0.8, "missing": [],
        })
        ensemble = EnsembleEngine({"mock": (mock_engine, 1.0)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        assert res["score"] == 0.8
        mock_engine.compare.assert_called_once()

    def test_compare_multi_engine_weighted(self):
        e1 = MagicMock(spec=SimilarityEngine)
        e1.compare.return_value = Ok({"score": 0.8, "matches": [{"skill": "py", "similarity": 0.8}]})
        e2 = MagicMock(spec=SimilarityEngine)
        e2.compare.return_value = Ok({"score": 0.4, "matches": [{"skill": "py", "similarity": 0.4}]})
        ensemble = EnsembleEngine({"e1": (e1, 0.7), "e2": (e2, 0.3)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        expected = (0.8 * 0.7 + 0.4 * 0.3) / 1.0
        assert res["score"] == pytest.approx(expected, 0.01)

    def test_compare_zero_total_weight(self):
        e1 = MagicMock(spec=SimilarityEngine)
        e1.compare.return_value = Ok({"score": 0.8, "matches": []})
        ensemble = EnsembleEngine({"e1": (e1, 0.0)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        assert res["score"] == 0.0

    def test_compare_merges_matches(self):
        e1 = MagicMock(spec=SimilarityEngine)
        e1.compare.return_value = Ok({"score": 0.8, "matches": [{"skill": "py", "similarity": 0.8}]})
        e2 = MagicMock(spec=SimilarityEngine)
        e2.compare.return_value = Ok({"score": 0.6, "matches": [{"skill": "py", "similarity": 0.6}]})
        ensemble = EnsembleEngine({"e1": (e1, 0.5), "e2": (e2, 0.5)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        assert len(res["matches"]) == 1
        assert res["matches"][0]["similarity"] == 0.8  # max

    def test_compare_result_has_details(self):
        e1 = MagicMock(spec=SimilarityEngine)
        e1.compare.return_value = Ok({"score": 0.7, "matches": []})
        ensemble = EnsembleEngine({"e1": (e1, 1.0)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        assert "details" in res
        assert "e1" in res["details"]

    def test_compare_limits_15_matches(self):
        engines = {}
        for i in range(3):
            m = MagicMock(spec=SimilarityEngine)
            m.compare.return_value = Ok({"score": 0.5, "matches": [{"skill": f"s{j}", "similarity": 0.5} for j in range(10)]})
            engines[f"e{i}"] = (m, 1.0)
        ensemble = EnsembleEngine(engines)
        res = _unwrap(ensemble.compare(["a"], [f"s{j}" for j in range(30)]))
        assert len(res["matches"]) <= 15

    def test_returns_required_keys(self):
        e1 = MagicMock(spec=SimilarityEngine)
        e1.compare.return_value = Ok({"score": 0.7, "matches": []})
        ensemble = EnsembleEngine({"e1": (e1, 1.0)})
        res = _unwrap(ensemble.compare(["python"], ["py"]))
        for key in ("score", "weighted_coverage", "avg_similarity", "matches", "missing", "details"):
            assert key in res
