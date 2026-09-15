"""Unit (v32): market synonym fold is deterministic and sums frequencies."""
from src.analyzers.skill_matcher import MARKET_SYNONYMS, fold_market_synonyms


def test_map_minimal_and_grounded():
    assert MARKET_SYNONYMS == {"python3": "python", "python 3": "python"}


def test_fold_merges_and_sums():
    market = {"python": 500, "python3": 1, "python 3": 8, "sql": 100}
    out = fold_market_synonyms(market)
    assert out["python"] == 509
    assert "python3" not in out and "python 3" not in out
    assert out["sql"] == 100
    # input untouched, canonical created when missing
    assert market["python"] == 500
    assert fold_market_synonyms({"python3": 2}) == {"python": 2}


def test_fold_noop_without_aliases():
    market = {"python": 500, "sql": 100}
    assert fold_market_synonyms(market) == market
