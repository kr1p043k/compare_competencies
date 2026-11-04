"""Регрессия: базовый навык не подменяется историей специфичного.

До фикса в ``get_rising``/``get_declining`` работал token-subset alias:
для нового ключа подбирался старый, чьи слова содержат его слова, и в
историю записывалась его частота::

    previous["api"] = previous["api gateway"]   # 3 вместо 799
    previous["sql"] = previous["ms sql"]        # 97 вместо 1231

Рост после этого считался от ложной базы, и на живом снапшоте базовые
навыки либо пропадали из списка, либо показывали выдуманные проценты.

Тест использует реальные значения снапшота: sql 1231 -> 1851,
ms sql 97 -> 245, api 799 -> 1456, api gateway 3 -> 16.
"""

from __future__ import annotations

from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer


def _analyzer(previous: dict[str, int], latest: dict[str, int]) -> SnapshotTrendAnalyzer:
    return SnapshotTrendAnalyzer([{"skill_freq": previous}, {"skill_freq": latest}])


class TestTokenSubsetAliasRemoved:
    def test_generic_skill_history_not_replaced_by_specific(self) -> None:
        # api: 799 -> 1456. Если бы история подменялась на 'api gateway'
        # (3 -> 16), навык получил бы почти -100% по доле и попал бы в падающие.
        # С корректной историей его доля меняется мало (799/802 -> 1456/1472).
        ta = _analyzer({"api": 799, "api gateway": 3}, {"api": 1456, "api gateway": 16})
        declining = {item["skill"]: item["change_pct"] for item in ta.get_declining(top_n=10).unwrap()}
        assert declining.get("api", 0) > -50

    def test_specific_skill_not_substituted_by_generic(self) -> None:
        ta = _analyzer({"sql": 1231, "ms sql": 97}, {"sql": 1851, "ms sql": 245})
        rising = {item["skill"]: item["change_pct"] for item in ta.get_rising(top_n=10).unwrap()}
        # ms sql растёт быстрее; его процент не должен подменяться процентом sql.
        assert "ms sql" in rising
        assert rising["ms sql"] != rising.get("sql")

    def test_no_alias_collapse_across_lists(self) -> None:
        # Один навык не может одновременно считаться растущим и падающим.
        ta = _analyzer(
            {"sql": 1231, "ms sql": 97, "api": 799, "api gateway": 3},
            {"sql": 1851, "ms sql": 245, "api": 1456, "api gateway": 16},
        )
        rising = {item["skill"] for item in ta.get_rising(top_n=10).unwrap()}
        declining = {item["skill"] for item in ta.get_declining(top_n=10).unwrap()}
        assert not (rising & declining)

    def test_declining_only_negative(self) -> None:
        ta = _analyzer({"a": 100, "b": 100, "c": 100}, {"a": 10, "b": 50, "c": 120})
        declining = ta.get_declining(top_n=10).unwrap()
        assert all(item["change_pct"] < 0 for item in declining)
        assert "c" not in {item["skill"] for item in declining}
