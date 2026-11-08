"""Unit tests for SnapshotTrendAnalyzer.get_rising / get_declining.

Synthetic snapshots are injected through the constructor (``snapshot_records``),
the only data seam of the analyzer, so nothing here touches a database,
file or network.

Both methods compare skill *shares* between the two freshest same-source
snapshots. The fixtures below keep snapshot totals equal, so the expected
share change equals the raw count change (100 -> 150 over 200 -> 200 totals
is exactly +50%).

Edge contract: with fewer than two snapshots the methods do not crash - they
return ``Err(TrendError(reason="insufficient_data"))``, the Result-typed
equivalent of "no data" (trend_analyzer.py get_rising/get_declining guards).
"""

from __future__ import annotations

import pytest

from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer
from src.errors import TrendError


def _analyzer(*freq_dicts: dict[str, int]) -> SnapshotTrendAnalyzer:
    """Analyzer over synthetic snapshots, one snapshot per freq dict."""
    return SnapshotTrendAnalyzer([
        {"skill_freq": dict(freqs), "source": "hh_vacancies"} for freqs in freq_dicts
    ])


def _skills(rows: list[dict]) -> set[str]:
    return {row["skill"] for row in rows}


class TestGetRisingDeclining:
    def test_rising_skill_reported_by_rising_not_declining(self) -> None:
        # Totals 200 -> 200: docker +50% (rising), nginx -50% (declining).
        ta = _analyzer({"docker": 100, "nginx": 100}, {"docker": 150, "nginx": 50})
        rising_result = ta.get_rising(top_n=10)
        declining_result = ta.get_declining(top_n=10)
        assert rising_result.is_ok()
        assert declining_result.is_ok()
        assert _skills(rising_result.ok()) == {"docker"}
        assert "docker" not in _skills(declining_result.ok())

    def test_declining_skill_reported_by_declining_not_rising(self) -> None:
        # Totals 200 -> 200: perl -40% (declining), redis +40% (rising).
        ta = _analyzer({"perl": 100, "redis": 100}, {"perl": 60, "redis": 140})
        rising_result = ta.get_rising(top_n=10)
        declining_result = ta.get_declining(top_n=10)
        assert rising_result.is_ok()
        assert declining_result.is_ok()
        assert _skills(declining_result.ok()) == {"perl"}
        assert "perl" not in _skills(rising_result.ok())

    def test_top_n_limits_rising(self) -> None:
        # Totals 500 -> 500: spark +40%, airflow +20%, kafka +10% rise.
        # top_n=2 must keep the two strongest and drop kafka.
        ta = _analyzer(
            {"spark": 100, "airflow": 100, "kafka": 100, "flask": 100, "cobol": 100},
            {"spark": 140, "airflow": 120, "kafka": 110, "flask": 40, "cobol": 90},
        )
        result = ta.get_rising(top_n=2)
        assert result.is_ok()
        rising = result.ok()
        assert len(rising) == 2
        assert _skills(rising) == {"spark", "airflow"}
        # Sorted by change_pct descending: the +40% leader comes first.
        assert [item["change_pct"] for item in rising] == [40.0, 20.0]

    def test_top_n_limits_declining(self) -> None:
        # Totals 500 -> 500: flask -60%, cobol -10% decline.
        # top_n=1 must keep only the steepest fall.
        ta = _analyzer(
            {"spark": 100, "airflow": 100, "kafka": 100, "flask": 100, "cobol": 100},
            {"spark": 140, "airflow": 120, "kafka": 110, "flask": 40, "cobol": 90},
        )
        result = ta.get_declining(top_n=1)
        assert result.is_ok()
        declining = result.ok()
        assert len(declining) == 1
        assert _skills(declining) == {"flask"}
        assert declining[0]["change_pct"] == -60.0

    def test_change_pct_computed_for_known_input(self) -> None:
        # Totals 200 -> 200, so count change == share change:
        # python 100 -> 150 is exactly +50%, java 100 -> 50 is exactly -50%.
        ta = _analyzer({"python": 100, "java": 100}, {"python": 150, "java": 50})
        rising = ta.get_rising(top_n=10).ok()
        python_item = next(item for item in rising if item["skill"] == "python")
        assert python_item["change_pct"] == 50.0
        assert python_item["change_pct_raw"] == 50.0
        assert python_item["capped"] is False
        assert python_item["frequency"] == 150
        declining = ta.get_declining(top_n=10).ok()
        java_item = next(item for item in declining if item["skill"] == "java")
        assert java_item["change_pct"] == -50.0

    def test_zero_change_skill_in_neither_list(self) -> None:
        # Totals 300 -> 300: go is flat (100 -> 100, 0%) - strictly positive
        # / negative filters must keep it out of both lists.
        ta = _analyzer(
            {"python": 100, "java": 100, "go": 100},
            {"python": 150, "java": 50, "go": 100},
        )
        rising = _skills(ta.get_rising(top_n=10).ok())
        declining = _skills(ta.get_declining(top_n=10).ok())
        assert "go" not in rising
        assert "go" not in declining

    @pytest.mark.parametrize("getter", ["get_rising", "get_declining"])
    @pytest.mark.parametrize("freq_dicts", [[], [{"python": 100}]], ids=["empty", "single"])
    def test_insufficient_snapshots_do_not_crash(
        self, getter: str, freq_dicts: list[dict[str, int]]
    ) -> None:
        # Fewer than two snapshots yield a typed Err, not a raised exception
        # and not an Ok list.
        ta = _analyzer(*freq_dicts)
        result = getattr(ta, getter)(top_n=10)
        assert isinstance(result.err(), TrendError)
        assert result.err().reason == "insufficient_data"
        assert result.ok() is None

    def test_token_subset_alias_counts_stay_distinct(self) -> None:
        # Regression: a token-subset alias used to overwrite the history of
        # the base skill with the specific one (previous["api"] took the
        # "api gateway" count instead of its own 799), so growth was computed
        # from a wrong base. "api" and "api gateway" must remain separate
        # skills with their own counts and their own percentages.
        # Totals 1011 -> 1011: api 799 -> 800 (+0.1%), api gateway 12 -> 15
        # (+25.0%), python 200 -> 196 (-2.0%).
        ta = _analyzer(
            {"api": 799, "api gateway": 12, "python": 200},
            {"api": 800, "api gateway": 15, "python": 196},
        )
        rising = {item["skill"]: item for item in ta.get_rising(top_n=10).ok()}
        # Each skill is reported separately, with its own frequency.
        assert set(rising) == {"api", "api gateway"}
        assert rising["api"]["frequency"] == 800
        assert rising["api gateway"]["frequency"] == 15
        assert rising["api"]["frequency"] != rising["api gateway"]["frequency"]
        # Percentages come from each skill's own base (799 and 12), not from
        # the sibling's count: the alias bug would show api at +200% (capped,
        # base 12) instead of +0.1%.
        assert rising["api"]["change_pct"] == 0.1
        assert rising["api gateway"]["change_pct"] == 25.0
        # Counts must not bleed across skills and a skill is never both
        # rising and declining.
        declining = _skills(ta.get_declining(top_n=10).ok())
        assert declining == {"python"}
        assert not (declining & set(rising))
