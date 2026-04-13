# tests/analyzers/test_trends.py
import json
from pathlib import Path
from datetime import datetime
import pytest
from unittest.mock import patch

from src.analyzers.trends import TrendAnalyzer
from src import config


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
        monkeypatch.setattr(config, "HISTORY_DIR", tmp_path)
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
        # имя содержит дату
        assert files[0].name.startswith("freq_20")

    def test_load_previous_snapshot(self, tmp_path, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        # сохраняем два снимка
        with open(tmp_path / "freq_2024-01-01.json", "w") as f:
            json.dump(sample_prev_freq, f)
        with open(tmp_path / "freq_2024-02-01.json", "w") as f:
            json.dump(sample_current_freq, f)

        prev = analyzer._load_previous_snapshot()
        assert prev == sample_prev_freq

    def test_load_previous_snapshot_insufficient_files(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        with open(tmp_path / "freq_latest.json", "w") as f:
            json.dump(sample_current_freq, f)
        prev = analyzer._load_previous_snapshot()
        assert prev == {}

    def test_get_trending_skills(self, tmp_path, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        # Патчим метод _load_previous_snapshot, чтобы он возвращал sample_prev_freq
        with patch.object(analyzer, '_load_previous_snapshot', return_value=sample_prev_freq):
            trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0)
        
        rising = trends["rising"]
        # Проверяем, что есть навыки с ростом >= 100%
        high_growth = any(r["change_pct"] >= 100 for r in rising)
        assert high_growth

    def test_get_trending_skills_no_prev(self, tmp_path, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills()
        assert trends == {"rising": [], "falling": []}

    def test_get_emerging_skills(self, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=5)
        # llm имеет низкую частоту, но ключевые слова -> потенциал RISING
        llm_item = next((e for e in emerging if e["skill"] == "llm"), None)
        assert llm_item is not None
        assert llm_item["potential"] == "RISING"
        # python не emerging (высокая частота)
        assert not any(e["skill"] == "python" for e in emerging)

    def test_get_emerging_skills_empty(self):
        analyzer = TrendAnalyzer({})
        emerging = analyzer.get_emerging_skills()
        assert emerging == []
        
    def test_get_stable_skills(self, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        stable = analyzer.get_stable_skills(top_n=3)
        # python вес 120, avg=66.6, 120 < 133.2 => STABLE
        python_item = next((s for s in stable if s["skill"] == "python"), None)
        assert python_item is not None
        assert python_item["stability"] == "STABLE"
        # docker 80, avg=66.6 => STABLE
        assert stable[0]["weight"] == 120  # python первый по весу

    def test_get_stable_skills_empty(self):
        analyzer = TrendAnalyzer({})
        stable = analyzer.get_stable_skills()
        assert stable == []

    def test_get_stable_skills_all_equal(self):
        freq = {"a": 10, "b": 10, "c": 10}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        # все равны среднему, должны попасть в stable
        assert len(stable) == 3
        for s in stable:
            assert s["stability"] == "STABLE"
            
    def test_get_trending_skills_empty_current(self, tmp_path, sample_prev_freq):
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        with open(tmp_path / "freq_prev.json", "w") as f:
            json.dump(sample_prev_freq, f)
        with open(tmp_path / "freq_curr.json", "w") as f:
            json.dump({}, f)
        trends = analyzer.get_trending_skills()
        assert trends == {"rising": [], "falling": []}

    def test_get_emerging_skills(self, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=5)
        # llm имеет вес 15, нормированный 15/120=0.125 > 0.05, поэтому не emerging
        llm_item = next((e for e in emerging if e["skill"] == "llm"), None)
        assert llm_item is None
        assert isinstance(emerging, list)

        
    def test_get_stable_skills_single_skill(self):
        freq = {"python": 100}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        assert len(stable) == 1
        assert stable[0]["stability"] == "STABLE"

    def test_get_stable_skills_exact_average(self):
        freq = {"a": 10, "b": 20, "c": 30}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        # avg=20, a не stable, b stable (STABLE), c stable (CRITICAL, т.к. 30 >= 40? нет, 30<40 -> STABLE)
        # Пересчитаем: avg=20, b=20 => STABLE, c=30 => <40 => STABLE.
        assert len(stable) == 2
        for s in stable:
            assert s["stability"] == "STABLE"
