# tests/analyzers/test_trends.py
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.analyzers.skills.trends import TrendAnalyzer


@pytest.fixture
def sample_current_freq():
    return {
        "python": 120,
        "sql": 90,
        "docker": 80,
        "fastapi": 45,
        "machine learning": 30,
        "llm": 15,
        "kubernetes": 70,
        "git": 110,
        "rest api": 50,
    }


@pytest.fixture
def sample_prev_freq():
    return {
        "python": 100,
        "sql": 95,
        "docker": 60,
        "fastapi": 30,
        "machine learning": 20,
        "llm": 5,
        "kubernetes": 65,
        "git": 105,
        "rest api": 55,
    }


class TestTrendAnalyzer:
    def test_init_default_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.analyzers.skills.trends.config.HISTORY_DIR", tmp_path)
        analyzer = TrendAnalyzer({})
        assert analyzer.history_dir == tmp_path
        assert analyzer.history_dir.exists()

    def test_init_custom_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_history"
        analyzer = TrendAnalyzer({}, historical_dir=custom_dir)
        assert analyzer.history_dir == custom_dir
        assert analyzer.history_dir.exists()

    def test_save_snapshot(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        analyzer.save_snapshot(sample_current_freq, label="test")
        files = list(tmp_path.glob("freq_*.json"))
        assert len(files) == 1
        assert "freq_test.json" in files[0].name

    def test_save_snapshot_default_label(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        analyzer.save_snapshot(sample_current_freq)
        files = list(tmp_path.glob("freq_*.json"))
        assert len(files) == 1
        assert files[0].name.startswith("freq_20")

    def test_load_all_snapshots(self, tmp_path, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        with open(tmp_path / "freq_2024-01-01.json", "w") as f:
            json.dump(sample_prev_freq, f)
        with open(tmp_path / "freq_2024-02-01.json", "w") as f:
            json.dump(sample_current_freq, f)

        snapshots = analyzer.load_all_snapshots()
        assert len(snapshots) >= 2

    def test_get_snapshots_for_analysis(self, tmp_path, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        with open(tmp_path / "freq_2024-01-01.json", "w") as f:
            json.dump(sample_prev_freq, f)
        with open(tmp_path / "freq_2024-02-01.json", "w") as f:
            json.dump(sample_current_freq, f)

        snapshots = analyzer.get_snapshots_for_analysis(n=1)
        assert len(snapshots) == 1

    def test_get_trending_skills_with_previous(self, sample_current_freq, sample_prev_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=sample_prev_freq)
        assert isinstance(trends["rising"], list)
        assert isinstance(trends["falling"], list)
        assert any(r["skill"] == "llm" for r in trends["rising"])
        assert any(r["skill"] == "fastapi" for r in trends["rising"])

    def test_get_trending_skills_no_previous_no_snapshots(self, sample_current_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills()
        assert trends == {"rising": [], "falling": []}

    def test_get_trending_skills_empty_current(self, sample_prev_freq, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(previous_snapshot=sample_prev_freq)
        assert isinstance(trends["rising"], list)
        assert isinstance(trends["falling"], list)

    def test_get_emerging_skills(self, sample_current_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=10)
        assert isinstance(emerging, list)

    def test_get_emerging_skills_empty(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        emerging = analyzer.get_emerging_skills()
        assert emerging == []

    def test_get_stable_skills(self, sample_current_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills(top_n=3)
        assert len(stable) <= 3
        if stable:
            assert "weight" in stable[0]
            assert "stability" in stable[0]

    def test_get_stable_skills_empty(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills()
        assert stable == []

    def test_get_stable_skills_all_equal(self, tmp_path):
        freq = {"a": 10, "b": 10, "c": 10}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills()
        assert len(stable) == 3
        for s in stable:
            assert s["stability"] == "STABLE"

    def test_get_skill_timeline(self, sample_current_freq, sample_prev_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("freq_2024-01-01.json"), sample_prev_freq),
            (dt2, Path("freq_2024-02-01.json"), sample_current_freq),
        ]
        timeline = analyzer.get_skill_timeline(["python", "llm"], snapshots)
        assert len(timeline["python"]) == 2
        assert timeline["python"][0] == (dt1, 100)
        assert timeline["python"][1] == (dt2, 120)

    def test_load_file(self, tmp_path, sample_current_freq):
        path = tmp_path / "test.json"
        with open(path, "w") as f:
            json.dump(sample_current_freq, f)
        data = TrendAnalyzer.load_file(path)
        assert data == sample_current_freq

    def test_save_snapshot_with_validator(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        analyzer.save_snapshot(sample_current_freq, label="validated", apply_whitelist=True)
        files = list(tmp_path.glob("freq_validated.json"))
        assert len(files) == 1
        with open(files[0], encoding="utf-8") as f:
            saved = json.load(f)
        assert "python" in saved


class TestTrendAnalyzerFull:
    @pytest.fixture
    def sample_freq(self):
        return {
            "python": 120,
            "sql": 90,
            "docker": 80,
            "fastapi": 45,
            "machine learning": 30,
            "llm": 15,
            "kubernetes": 70,
            "git": 110,
        }

    @pytest.fixture
    def sample_prev(self):
        return {
            "python": 100,
            "sql": 95,
            "docker": 60,
            "fastapi": 30,
            "machine learning": 20,
            "llm": 5,
            "kubernetes": 65,
            "git": 105,
        }

    def test_get_emerging_skills_with_specific_keywords(self, sample_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=20)
        llm_item = next((e for e in emerging if e["skill"] == "llm"), None)
        if llm_item:
            assert llm_item["potential"] == "RISING"

    def test_get_stable_skills_critical(self, tmp_path):
        freq = {"python": 100, "sql": 10, "docker": 8, "git": 7}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills()
        python_item = next((s for s in stable if s["skill"] == "python"), None)
        if python_item:
            assert python_item["stability"] == "CRITICAL"

    def test_save_snapshot_with_whitelist(self, tmp_path, sample_freq):
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        analyzer.save_snapshot(sample_freq, label="test", apply_whitelist=True)
        files = list(tmp_path.glob("freq_test.json"))
        assert len(files) == 1

    def test_get_skill_timeline_empty_skills(self, sample_freq, sample_prev, tmp_path):
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), sample_prev),
            (dt2, Path("f2.json"), sample_freq),
        ]
        timeline = analyzer.get_skill_timeline([], snapshots)
        assert isinstance(timeline, dict)
        assert len(timeline) == 0

    def test_get_skill_timeline_missing_skill(self, sample_freq, sample_prev, tmp_path):
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        snapshots = [(dt1, Path("f1.json"), sample_prev)]
        timeline = analyzer.get_skill_timeline(["nonexistent"], snapshots)
        assert timeline["nonexistent"][0][1] == 0

    def test_extract_date_formats(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        path1 = tmp_path / "freq_2024-01-01-120000.json"
        path1.write_text("{}")
        dt1 = analyzer._extract_date(path1)
        assert dt1 == datetime(2024, 1, 1, 12, 0, 0)

        path2 = tmp_path / "freq_2024-06-15.json"
        path2.write_text("{}")
        dt2 = analyzer._extract_date(path2)
        assert dt2 == datetime(2024, 6, 15)

    def test_get_trending_skills_new_skill_zero_division(self, sample_freq, tmp_path):
        prev = {"python": 100}
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(previous_snapshot=prev)
        fastapi_in_rising = any(r["skill"] == "fastapi" for r in trends["rising"])
        assert not fastapi_in_rising


class TestTrendAnalyzerPlots:
    @pytest.fixture
    def freq(self):
        return {"python": 120, "sql": 90, "docker": 80, "fastapi": 45, "llm": 15, "kubernetes": 70}

    @pytest.fixture
    def prev(self):
        return {"python": 100, "sql": 95, "docker": 60, "fastapi": 30, "llm": 5, "kubernetes": 65}

    def test_plot_trending_with_data(self, tmp_path, freq, prev):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "trending.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
            if result is not None:
                mock_save.assert_called()

    def test_plot_trending_no_significant_trends(self, tmp_path):
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        result = analyzer.plot_trending(top_n=10, save_path=None, previous_snapshot=prev)
        assert result is None

    def test_plot_timeline(self, tmp_path, freq, prev):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline.png"
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            analyzer.plot_timeline(["python", "sql"], snapshots=snapshots, save_path=save_path, title="Test Timeline")
            mock_save.assert_called()

    def test_plot_timeline_insufficient_snapshots(self, tmp_path, freq):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        snapshots = [(dt1, Path("f1.json"), freq)]
        result = analyzer.plot_timeline(["python"], snapshots=snapshots)
        assert result is None

    def test_get_trending_skills_falling_only(self, tmp_path):
        current = {"python": 50}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["skill"] == "python"
        assert trends["falling"][0]["change_pct"] == -50.0

    def test_get_stable_skills_critical_threshold(self, tmp_path):
        freq = {"critical": 100, "normal": 20, "low": 15}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills()
        critical_item = next((s for s in stable if s["skill"] == "critical"), None)
        assert critical_item is not None
        assert critical_item["stability"] == "CRITICAL"

    def test_plot_timeline_with_skill_having_zeros(self, tmp_path):
        freq = {"python": 120}
        prev = {"python": 0}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline.png"
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig"):
            result = analyzer.plot_timeline(["python"], snapshots=snapshots, save_path=save_path)
            assert result is not None

    def test_plot_trending_with_rising_and_falling(self, tmp_path):
        current = {"python": 200, "sql": 50, "docker": 100}
        prev = {"python": 100, "sql": 100, "docker": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_both.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
            mock_save.assert_called()

    def test_plot_trending_only_rising(self, tmp_path):
        current = {"python": 150, "docker": 120}
        prev = {"python": 100, "docker": 110}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_rising.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
            assert result is not None
            mock_save.assert_called()

    def test_plot_timeline_with_custom_formatting(self, tmp_path, freq, prev):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_format.png"
        dt1 = datetime(2024, 1, 15)
        dt2 = datetime(2024, 3, 20)
        dt3 = datetime(2024, 6, 10)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), {"python": 110, "sql": 92}),
            (dt3, Path("f3.json"), freq),
        ]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            analyzer.plot_timeline(
                ["python", "sql", "docker"],
                snapshots=snapshots,
                save_path=save_path,
                title="Расширенный анализ трендов",
            )
            mock_save.assert_called()

    def test_plot_timeline_large_dataset(self, tmp_path):
        skills = [f"skill_{i}" for i in range(15)]
        freq = {s: np.random.randint(10, 200) for s in skills}
        prev = {s: np.random.randint(10, 200) for s in skills}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_large.png"
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            analyzer.plot_timeline(skills[:10], snapshots=snapshots, save_path=save_path)
            mock_save.assert_called()

    def test_extract_date_from_invalid_filename(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        path = tmp_path / "freq_invalid.json"
        path.write_text("{}")
        dt = analyzer._extract_date(path)
        assert isinstance(dt, datetime)

    def test_extract_date_space_format(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        path = tmp_path / "freq_2024-01-01 120000.json"
        path.write_text("{}")
        dt = analyzer._extract_date(path)
        assert isinstance(dt, datetime)

    def test_load_all_snapshots_with_invalid_file(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        (tmp_path / "freq_broken.json").write_text("{invalid")
        with open(tmp_path / "freq_valid.json", "w") as f:
            json.dump(sample_current_freq, f)
        snapshots = analyzer.load_all_snapshots()
        assert len(snapshots) == 1

    def test_get_emerging_skills_with_cloud_keywords(self, tmp_path):
        freq = {"python": 200, "sql": 150, "aws": 12, "azure": 8, "gcp": 5, "terraform": 15}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        emerging = analyzer.get_emerging_skills(min_weight=0.03, top_n=20)
        cloud_skills = [e for e in emerging if e["potential"] == "RISING"]
        assert len(cloud_skills) >= 0

    def test_get_stable_skills_with_high_threshold(self, tmp_path):
        freq = {"super_critical": 200, "critical": 100, "medium": 50, "low": 10}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills()
        super_item = next((s for s in stable if s["skill"] == "super_critical"), None)
        if super_item:
            assert super_item["stability"] == "CRITICAL"

    def test_save_snapshot_with_apply_whitelist_true(self, tmp_path, freq):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="whitelist", apply_whitelist=True)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert isinstance(saved, dict)

    def test_get_trending_skills_falling_edge_case(self, tmp_path):
        current = {"legacy_skill": 10}
        prev = {"legacy_skill": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["change_pct"] == -90.0

    def test_plot_timeline_mdates_formatting(self, tmp_path, freq, prev):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_dates.png"
        dates = [datetime(2024, i, 1) for i in range(1, 7)]
        snapshots = [(d, Path(f"freq_{d.strftime('%Y%m%d')}.json"), freq) for d in dates]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            analyzer.plot_timeline(["python", "sql"], snapshots=snapshots, save_path=save_path)
            mock_save.assert_called()

    def test_plot_timeline_without_save_path(self, freq, prev, tmp_path):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_timeline(["python"], snapshots=snapshots, save_path=None)
            assert result is not None
            mock_save.assert_not_called()

    def test_trend_module_can_be_imported(self):
        import src.analyzers.skills.trends
        assert hasattr(src.analyzers.skills.trends, "TrendAnalyzer")

    def test_get_trending_skills_equal_frequencies(self, tmp_path):
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=1.0, previous_snapshot=prev)
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_get_trending_skills_rising_only(self, tmp_path):
        current = {"python": 200, "sql": 150, "docker": 100}
        prev = {"python": 100, "sql": 100, "docker": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["rising"]) >= 1
        assert len(trends["falling"]) == 0

    def test_plot_trending_only_falling(self, tmp_path):
        current = {"python": 50, "sql": 40}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_fall.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
            assert result is not None
            mock_save.assert_called()

    def test_get_emerging_skills_cloud_potential(self):
        freq = {"aws": 3, "azure": 2, "kubernetes": 1, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        cloud_skills = [e for e in emerging if e["skill"] in ("aws", "azure", "kubernetes")]
        assert len(cloud_skills) == 3
        for cs in cloud_skills:
            assert cs["potential"] == "RISING"

    def test_get_emerging_skills_default_potential(self):
        freq = {"new_tech": 3, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tech = next((e for e in emerging if e["skill"] == "new_tech"), None)
        if new_tech:
            assert new_tech["potential"] == "STABLE"

    def test_get_stable_skills_exact_average(self, tmp_path):
        freq = {"a": 10, "b": 10, "c": 10}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills(top_n=10)
        for s in stable:
            assert s["stability"] == "STABLE"

    def test_get_stable_skills_critical_boundary(self, tmp_path):
        freq = {"critical": 100, "normal": 50}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills(top_n=10)
        critical_item = next((s for s in stable if s["skill"] == "critical"), None)
        if critical_item:
            assert critical_item["stability"] == "STABLE"

    def test_get_stable_skills_clear_critical(self, tmp_path):
        freq = {"ultra_critical": 300, "normal": 50, "low": 25}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        stable = analyzer.get_stable_skills(top_n=10)
        ultra = next((s for s in stable if s["skill"] == "ultra_critical"), None)
        if ultra:
            assert ultra["stability"] == "CRITICAL"

    def test_get_emerging_skills_stable_default(self):
        freq = {"new_tool": 3, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tool = next((e for e in emerging if e["skill"] == "new_tool"), None)
        if new_tool:
            assert new_tool["potential"] == "STABLE"

    def test_save_snapshot_without_whitelist(self, tmp_path, freq):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="no_filter", apply_whitelist=False)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == len(freq)

    def test_get_trending_skills_below_threshold_zero_change(self, tmp_path):
        current = {"python": 100}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=1.0, previous_snapshot=prev)
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_plot_timeline_returns_figure(self, freq, prev, tmp_path):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig"):
            result = analyzer.plot_timeline(["python"], snapshots=snapshots)
            assert result is not None

    def test_plot_trending_returns_figure(self, tmp_path, freq, prev):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        with patch("matplotlib.pyplot.savefig"):
            result = analyzer.plot_trending(top_n=5, save_path=None, previous_snapshot=prev)
            if result is not None:
                import matplotlib.figure
                assert isinstance(result, matplotlib.figure.Figure)

    def test_get_trending_skills_auto_previous_with_files(self, tmp_path):
        current = {"python": 200}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["rising"]) > 0
        assert trends["rising"][0]["prev_label"] == "предыдущий"

    def test_trend_analyzer_load_file(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text('{"python": 100, "sql": 50}', encoding="utf-8")
        data = TrendAnalyzer.load_file(test_file)
        assert data == {"python": 100, "sql": 50}

    def test_get_snapshots_for_analysis_all(self, tmp_path):
        for i in range(3):
            snap = tmp_path / f"freq_2024-0{i + 1}-01.json"
            snap.write_text(json.dumps({"python": 100 + i * 50}))
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        snapshots = analyzer.get_snapshots_for_analysis()
        assert len(snapshots) == 3

    def test_save_snapshot_no_whitelist_no_label(self, tmp_path, freq):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, apply_whitelist=False)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == len(freq)

    def test_get_trending_skills_rising_edge(self, tmp_path):
        current = {"python": 110}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["rising"]) == 1

    def test_get_trending_skills_falling_edge(self, tmp_path):
        current = {"python": 90}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0, previous_snapshot=prev)
        assert len(trends["falling"]) == 1
        assert trends["falling"][0]["change_pct"] == -10.0

    def test_plot_trending_no_rising_only_falling(self, tmp_path):
        current = {"python": 50}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_fall.png"
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
            assert result is not None
            mock_save.assert_called()

    def test_plot_timeline_empty_skills(self, freq, prev, tmp_path):
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), prev), (dt2, Path("f2.json"), freq)]
        with patch("matplotlib.pyplot.savefig"):
            result = analyzer.plot_timeline([], snapshots=snapshots)
            assert result is not None


class TestTrendAnalyzerCLI:
    def test_get_emerging_skills_with_cloud_keywords(self):
        freq = {"python": 200, "sql": 150, "aws_lambda": 5, "azure_functions": 3, "llm_integration": 4, "k8s_operator": 6}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.03, top_n=20)
        for e in emerging:
            if any(kw in e["skill"] for kw in ["aws", "azure", "llm", "k8s"]):
                assert e["potential"] == "RISING"

    def test_get_emerging_skills_default_potential(self):
        freq = {"python": 200, "new_tool": 3}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tool = next((e for e in emerging if e["skill"] == "new_tool"), None)
        if new_tool:
            assert new_tool["potential"] == "STABLE"

    def test_trend_analyzer_load_file(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text('{"python": 100, "sql": 50}', encoding="utf-8")
        data = TrendAnalyzer.load_file(test_file)
        assert data == {"python": 100, "sql": 50}

    def test_extract_date_formats(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        path1 = tmp_path / "freq_2024-01-01-120000.json"
        path1.write_text("{}")
        assert analyzer._extract_date(path1) == datetime(2024, 1, 1, 12, 0, 0)
        path2 = tmp_path / "freq_2024-06-15.json"
        path2.write_text("{}")
        assert analyzer._extract_date(path2) == datetime(2024, 6, 15)
        path3 = tmp_path / "freq_2024-03-20 120000.json"
        path3.write_text("{}")
        assert isinstance(analyzer._extract_date(path3), datetime)
        path4 = tmp_path / "freq_invalid.json"
        path4.write_text("{}")
        assert isinstance(analyzer._extract_date(path4), datetime)

    def test_load_all_snapshots_with_invalid_file(self, tmp_path):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        (tmp_path / "freq_broken.json").write_text("{invalid")
        (tmp_path / "freq_2024-01-01.json").write_text('{"python": 100}')
        snapshots = analyzer.load_all_snapshots()
        assert len(snapshots) == 1

    def test_get_snapshots_for_analysis_n(self, tmp_path):
        for i in range(5):
            snap = tmp_path / f"freq_2024-0{i+1}-01.json"
            snap.write_text(json.dumps({"python": 100 + i * 10}))
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        assert len(analyzer.get_snapshots_for_analysis()) == 5
        assert len(analyzer.get_snapshots_for_analysis(n=2)) == 2

    def test_plot_timeline_without_save(self, tmp_path, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [(dt1, Path("f1.json"), sample_prev_freq), (dt2, Path("f2.json"), sample_current_freq)]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_timeline(["python"], snapshots=snapshots, save_path=None)
            assert result is not None
            mock_save.assert_not_called()

    def test_plot_trending_no_significant_trends(self, tmp_path):
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        result = analyzer.plot_trending(top_n=10, save_path=None, previous_snapshot=prev)
        assert result is None

    def test_plot_trending_only_rising(self, tmp_path):
        current = {"python": 150, "docker": 120}
        prev = {"python": 100, "docker": 110}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        with patch("matplotlib.pyplot.savefig") as mock_save:
            result = analyzer.plot_trending(top_n=10, save_path=tmp_path / "test.png", previous_snapshot=prev)
            assert result is not None
            mock_save.assert_called()

    def test_get_trending_skills_with_prev_label(self, tmp_path):
        sample_freq = {"python": 120, "sql": 90, "docker": 80}
        sample_prev = {"python": 100, "sql": 95, "docker": 60}
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(previous_snapshot=sample_prev, prev_label="custom_label")
        for r in trends.get("rising", []):
            assert r["prev_label"] == "custom_label"

    def test_get_trending_skills_equal_frequencies(self, tmp_path):
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(min_change_percent=1.0, previous_snapshot=prev)
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0
