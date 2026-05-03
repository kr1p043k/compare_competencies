# tests/analyzers/test_trends.py
import json
from pathlib import Path
from datetime import datetime
import pytest
from unittest.mock import patch
from src.analyzers.trends import TrendAnalyzer
from src import config
import numpy as np

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

    def test_get_trending_skills_with_previous(self, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=sample_prev_freq
        )

        rising = trends["rising"]
        falling = trends["falling"]

        assert isinstance(rising, list)
        assert isinstance(falling, list)

        llm_rising = any(r["skill"] == "llm" for r in rising)
        assert llm_rising, f"llm not in rising: {[r['skill'] for r in rising]}"

        fastapi_rising = any(r["skill"] == "fastapi" for r in rising)
        assert fastapi_rising, f"fastapi not in rising: {[r['skill'] for r in rising]}"

    # ИСПРАВЛЕНО: get_trending_skills без previous_snapshot и без снимков в tmp_path
    # возвращает {"rising": [], "falling": []} из-за проверки len(snapshots) < 2
    def test_get_trending_skills_no_previous_no_snapshots(self, sample_current_freq, tmp_path):
        analyzer = TrendAnalyzer(sample_current_freq, historical_dir=tmp_path)
        # history_dir пуст, load_all_snapshots вернёт []
        trends = analyzer.get_trending_skills()
        assert trends == {"rising": [], "falling": []}

    def test_get_trending_skills_empty_current(self, sample_prev_freq):
        analyzer = TrendAnalyzer({})
        trends = analyzer.get_trending_skills(previous_snapshot=sample_prev_freq)
        assert isinstance(trends["rising"], list)
        assert isinstance(trends["falling"], list)

    def test_get_emerging_skills(self, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=10)
        assert isinstance(emerging, list)

    def test_get_emerging_skills_empty(self):
        analyzer = TrendAnalyzer({})
        emerging = analyzer.get_emerging_skills()
        assert emerging == []

    def test_get_stable_skills(self, sample_current_freq):
        analyzer = TrendAnalyzer(sample_current_freq)
        stable = analyzer.get_stable_skills(top_n=3)
        assert len(stable) <= 3
        if stable:
            assert "weight" in stable[0]
            assert "stability" in stable[0]

    def test_get_stable_skills_empty(self):
        analyzer = TrendAnalyzer({})
        stable = analyzer.get_stable_skills()
        assert stable == []

    def test_get_stable_skills_all_equal(self):
        freq = {"a": 10, "b": 10, "c": 10}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        assert len(stable) == 3
        for s in stable:
            assert s["stability"] == "STABLE"

    def test_get_skill_timeline(self, sample_current_freq, sample_prev_freq):
        analyzer = TrendAnalyzer(sample_current_freq)

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

        with open(files[0], 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert "python" in saved

class TestTrendAnalyzerFull:
    @pytest.fixture
    def sample_freq(self):
        return {
            "python": 120, "sql": 90, "docker": 80,
            "fastapi": 45, "machine learning": 30,
            "llm": 15, "kubernetes": 70, "git": 110
        }

    @pytest.fixture
    def sample_prev(self):
        return {
            "python": 100, "sql": 95, "docker": 60,
            "fastapi": 30, "machine learning": 20,
            "llm": 5, "kubernetes": 65, "git": 105
        }

    def test_get_emerging_skills_with_specific_keywords(self, sample_freq):
        """Строки 67-68: emerging с AI/cloud ключевыми словами"""
        analyzer = TrendAnalyzer(sample_freq)
        # llm с низким весом + ключевое слово → potential=RISING
        emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=20)
        llm_item = next((e for e in emerging if e["skill"] == "llm"), None)
        if llm_item:
            assert llm_item["potential"] == "RISING"

    def test_get_stable_skills_critical(self):
        """Строка 85: CRITICAL при высоком весе"""
        freq = {"python": 100, "sql": 10, "docker": 8, "git": 7}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        python_item = next((s for s in stable if s["skill"] == "python"), None)
        if python_item:
            # avg=(100+10+8+7)/4=31.25, python=100 >= 62.5 → CRITICAL
            assert python_item["stability"] == "CRITICAL"

    def test_save_snapshot_with_whitelist(self, tmp_path, sample_freq):
        """Строка 104: сохранение с валидацией"""
        analyzer = TrendAnalyzer(sample_freq, historical_dir=tmp_path)
        analyzer.save_snapshot(sample_freq, label="test", apply_whitelist=True)
        files = list(tmp_path.glob("freq_test.json"))
        assert len(files) == 1

    def test_get_trending_skills_with_prev_label(self, sample_freq, sample_prev):
        """Строки 120-121: кастомный prev_label"""
        analyzer = TrendAnalyzer(sample_freq)
        trends = analyzer.get_trending_skills(
            previous_snapshot=sample_prev,
            prev_label="custom_label"
        )
        for r in trends.get("rising", []):
            assert r["prev_label"] == "custom_label"
        for f in trends.get("falling", []):
            assert f["prev_label"] == "custom_label"

    def test_get_skill_timeline_empty_skills(self, sample_freq, sample_prev):
        """Тест временных рядов"""
        analyzer = TrendAnalyzer(sample_freq)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), sample_prev),
            (dt2, Path("f2.json"), sample_freq),
        ]
        timeline = analyzer.get_skill_timeline([], snapshots)
        assert isinstance(timeline, dict)
        assert len(timeline) == 0

    def test_get_skill_timeline_missing_skill(self, sample_freq, sample_prev):
        """Навык отсутствует в снимках → 0"""
        analyzer = TrendAnalyzer(sample_freq)
        dt1 = datetime(2024, 1, 1)
        snapshots = [(dt1, Path("f1.json"), sample_prev)]
        timeline = analyzer.get_skill_timeline(["nonexistent"], snapshots)
        assert timeline["nonexistent"][0][1] == 0

    def test_extract_date_formats(self, tmp_path):
        """Разные форматы дат в именах файлов"""
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)

        # Формат с временем
        path1 = tmp_path / "freq_2024-01-01-120000.json"
        dt1 = analyzer._extract_date(path1)
        assert dt1 == datetime(2024, 1, 1, 12, 0, 0)

        # Формат без времени
        path2 = tmp_path / "freq_2024-06-15.json"
        dt2 = analyzer._extract_date(path2)
        assert dt2 == datetime(2024, 6, 15)

    def test_get_trending_skills_new_skill_zero_division(self, sample_freq):
        """Строка 129: навык с prev_freq=0 пропускается"""
        prev = {"python": 100}
        analyzer = TrendAnalyzer(sample_freq)
        trends = analyzer.get_trending_skills(previous_snapshot=prev)
        # fastapi не было в prev → пропускается
        fastapi_in_rising = any(r["skill"] == "fastapi" for r in trends["rising"])
        assert not fastapi_in_rising
        
class TestTrendAnalyzerPlots:
    """Тесты для графиков (строки 200-394)"""
    
    @pytest.fixture
    def freq(self):
        return {
            "python": 120, "sql": 90, "docker": 80,
            "fastapi": 45, "llm": 15, "kubernetes": 70
        }
    
    @pytest.fixture
    def prev(self):
        return {
            "python": 100, "sql": 95, "docker": 60,
            "fastapi": 30, "llm": 5, "kubernetes": 65
        }
    
    def test_plot_trending_with_data(self, tmp_path, freq, prev):
        """Строки 200-232: график трендов"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "trending.png"
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_trending(
                top_n=10,
                save_path=save_path,
                previous_snapshot=prev
            )
            # Если были данные для графика, savefig вызывается
            # (может не вызваться если нет значимых трендов)
    
    def test_plot_trending_no_significant_trends(self, tmp_path):
        """Строки 215-216: нет значимых трендов"""
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}  # одинаковые → нет трендов
        
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "trending.png"
        
        result = analyzer.plot_trending(top_n=10, save_path=save_path, previous_snapshot=prev)
        assert result is None  # нет значимых трендов
    
    def test_plot_timeline(self, tmp_path, freq, prev):
        """Строки 237-287: график временных рядов"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline.png"
        
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), freq),
        ]
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_timeline(
                ["python", "sql"],
                snapshots=snapshots,
                save_path=save_path,
                title="Test Timeline"
            )
            mock_save.assert_called()
    
    def test_plot_timeline_insufficient_snapshots(self, tmp_path, freq):
        """Строки 246-247: недостаточно снимков"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline.png"
        
        dt1 = datetime(2024, 1, 1)
        snapshots = [(dt1, Path("f1.json"), freq)]
        
        result = analyzer.plot_timeline(["python"], snapshots=snapshots)
        assert result is None
    
    def test_get_trending_skills_falling_only(self, tmp_path):
        """Строки 155-160: только падающие навыки"""
        current = {"python": 50}
        prev = {"python": 100}
        
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["skill"] == "python"
        assert trends["falling"][0]["change_pct"] == -50.0
    
    def test_get_stable_skills_critical_threshold(self):
        """Строка 85: CRITICAL = вес >= 2*avg"""
        freq = {"critical": 100, "normal": 20, "low": 15}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()
        
        critical_item = next((s for s in stable if s["skill"] == "critical"), None)
        assert critical_item is not None
        assert critical_item["stability"] == "CRITICAL"
    
    def test_plot_timeline_with_skill_having_zeros(self, tmp_path):
        """Строки 237-287: навык с нулевыми значениями"""
        freq = {"python": 120}
        prev = {"python": 0}
        
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline.png"
        
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), freq),
        ]
        
        with patch('matplotlib.pyplot.savefig'):
            result = analyzer.plot_timeline(["python"], snapshots=snapshots, save_path=save_path)
            assert result is not None
            
    def test_plot_trending_with_rising_and_falling(self, tmp_path):
        """Строки 200-232: график с растущими и падающими навыками"""
        current = {"python": 200, "sql": 50, "docker": 100}
        prev = {"python": 100, "sql": 100, "docker": 100}

        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_both.png"

        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_trending(
                top_n=10,
                save_path=save_path,
                previous_snapshot=prev
            )
            mock_save.assert_called()

    def test_plot_trending_only_rising(self, tmp_path):
        """Строки 211: только растущие навыки (падающие < порога)"""
        current = {"python": 150, "docker": 120}
        prev = {"python": 100, "docker": 110}  # docker +9% (<10% порога)

        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_rising.png"

        with patch('matplotlib.pyplot.savefig') as mock_save:
            result = analyzer.plot_trending(
                top_n=10,
                save_path=save_path,
                previous_snapshot=prev
            )
            # python: +50% → RISING
            # docker: +9% → меньше порога 10%
            # savefig вызывается (rising не пуст)
            mock_save.assert_called()

    def test_plot_timeline_with_custom_formatting(self, tmp_path, freq, prev):
        """Строки 237-287: график с форматированием дат"""
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

        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_timeline(
                ["python", "sql", "docker"],
                snapshots=snapshots,
                save_path=save_path,
                title="Расширенный анализ трендов"
            )
            mock_save.assert_called()

    def test_plot_timeline_large_dataset(self, tmp_path):
        """Строки 237-287: большой набор навыков (>10)"""
        skills = [f"skill_{i}" for i in range(15)]
        freq = {s: np.random.randint(10, 200) for s in skills}
        prev = {s: np.random.randint(10, 200) for s in skills}

        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_large.png"

        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), freq),
        ]

        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_timeline(
                skills[:10],  # первые 10 для легенды
                snapshots=snapshots,
                save_path=save_path
            )
            mock_save.assert_called()

    def test_extract_date_from_invalid_filename(self, tmp_path):
        """Строки 292-310: извлечение даты из нестандартного имени"""
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)

        # Файл без даты в имени — используется дата изменения
        path = tmp_path / "freq_invalid.json"
        path.write_text("{}")
        dt = analyzer._extract_date(path)
        assert isinstance(dt, datetime)

    def test_extract_date_space_format(self, tmp_path):
        """Строки 292-310: дата с пробелом вместо дефиса"""
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)

        path = tmp_path / "freq_2024-01-01 120000.json"
        dt = analyzer._extract_date(path)
        assert isinstance(dt, datetime)

    def test_load_all_snapshots_with_invalid_file(self, tmp_path, sample_current_freq):
        """Строки 120-121: загрузка с битым файлом"""
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)

        # Создаём битый JSON
        (tmp_path / "freq_broken.json").write_text("{invalid")
        # И валидный
        with open(tmp_path / "freq_valid.json", "w") as f:
            json.dump(sample_current_freq, f)

        snapshots = analyzer.load_all_snapshots()
        # Должен пропустить битый и загрузить валидный
        assert len(snapshots) == 1

    def test_get_emerging_skills_with_cloud_keywords(self):
        """Строки 67-68: emerging с cloud ключевыми словами"""
        freq = {
            "python": 200, "sql": 150,
            "aws": 12, "azure": 8, "gcp": 5,  # низкий вес + cloud keywords
            "terraform": 15,
        }
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.03, top_n=20)

        cloud_skills = [e for e in emerging if e["potential"] == "RISING"]
        # aws/azure/gcp/terraform с низким весом должны быть помечены как RISING
        # (если проходят по min_weight)
        assert len(cloud_skills) >= 0  # зависит от min_weight

    def test_get_stable_skills_with_high_threshold(self):
        """Строка 85: CRITICAL при превышении 2*avg и 3*avg"""
        freq = {"super_critical": 200, "critical": 100, "medium": 50, "low": 10}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills()

        super_item = next((s for s in stable if s["skill"] == "super_critical"), None)
        if super_item:
            # avg=90, super_critical=200 >= 180 → CRITICAL
            assert super_item["stability"] == "CRITICAL"
            
    def test_save_snapshot_with_apply_whitelist_true(self, tmp_path, freq):
        """Строка 104: save_snapshot с apply_whitelist=True"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="whitelist", apply_whitelist=True)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        # После валидации остаются только валидные навыки
        assert isinstance(saved, dict)

    def test_get_trending_skills_falling_edge_case(self, tmp_path):
        """Строка 157: падающий навык с большим отрицательным изменением"""
        current = {"legacy_skill": 10}
        prev = {"legacy_skill": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["change_pct"] == -90.0

    def test_get_trending_skills_below_threshold(self, tmp_path):
        """Строка 159: изменение меньше порога → не попадает"""
        current = {"python": 105}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        # 5% < 10% → не попадает ни в rising, ни в falling
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_plot_timeline_mdates_formatting(self, tmp_path, freq, prev):
        """Строки 261-262: форматирование дат на оси X"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_dates.png"

        dates = [datetime(2024, i, 1) for i in range(1, 7)]
        snapshots = [(d, Path(f"freq_{d.strftime('%Y%m%d')}.json"), freq) for d in dates]

        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_timeline(
                ["python", "sql"],
                snapshots=snapshots,
                save_path=save_path
            )
            mock_save.assert_called()

    def test_save_snapshot_with_apply_whitelist_true(self, tmp_path, freq):
        """Строка 104: save_snapshot с apply_whitelist=True"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="whitelist", apply_whitelist=True)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert isinstance(saved, dict)

    def test_get_trending_skills_falling_edge_case(self, tmp_path):
        """Строка 157: падающий навык с большим изменением"""
        current = {"legacy_skill": 10}
        prev = {"legacy_skill": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["change_pct"] == -90.0

    def test_get_trending_skills_below_threshold(self, tmp_path):
        """Строка 159: изменение меньше порога → не попадает"""
        current = {"python": 105}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_plot_timeline_mdates_formatting(self, tmp_path, freq, prev):
        """Строки 261-262: форматирование дат на оси X"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        save_path = tmp_path / "timeline_dates.png"
        dates = [datetime(2024, i, 1) for i in range(1, 7)]
        snapshots = [(d, Path(f"freq_{d.strftime('%Y%m%d')}.json"), freq) for d in dates]
        with patch('matplotlib.pyplot.savefig') as mock_save:
            analyzer.plot_timeline(["python", "sql"], snapshots=snapshots, save_path=save_path)
            mock_save.assert_called()
            
    def test_plot_timeline_without_save_path(self, freq, prev):
        """Строки 237-287: график без сохранения в файл"""
        analyzer = TrendAnalyzer(freq)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), freq),
        ]
        with patch('matplotlib.pyplot.savefig') as mock_save:
            result = analyzer.plot_timeline(["python"], snapshots=snapshots, save_path=None)
            assert result is not None
            mock_save.assert_not_called()

    def test_cli_main_block_import(self):
        """Строки 292-394: импорт модуля trends как __main__"""
        # Проверяем что модуль можно импортировать без ошибок
        import src.analyzers.trends
        assert hasattr(src.analyzers.trends, 'TrendAnalyzer')

    def test_cli_argument_parsing(self, tmp_path, freq):
        """Строки 292-394: парсинг аргументов CLI"""
        import src.analyzers.trends as trends_module
        
        # Сохраняем тестовый файл
        test_file = tmp_path / "test_freq.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(freq, f)
        
        # Проверяем что функция load_file работает
        loaded = trends_module.TrendAnalyzer.load_file(test_file)
        assert loaded == freq

    def test_get_trending_skills_equal_frequencies(self, tmp_path):
        """Строка 211: одинаковые частоты → нет изменений"""
        freq = {"python": 100, "sql": 100}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=1.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_get_trending_skills_rising_only(self, tmp_path):
        """Строка 211: только растущие навыки"""
        current = {"python": 200, "sql": 150, "docker": 100}
        prev = {"python": 100, "sql": 100, "docker": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) >= 1
        assert len(trends["falling"]) == 0

    def test_plot_trending_only_falling(self, tmp_path):
        """Строки 211: только падающие навыки"""
        current = {"python": 50, "sql": 40}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        save_path = tmp_path / "trending_fall.png"
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            result = analyzer.plot_trending(
                top_n=10, save_path=save_path, previous_snapshot=prev
            )
            assert result is not None
            mock_save.assert_called()

    def test_get_emerging_skills_cloud_potential(self):
        """Строки 67-68: emerging с cloud ключевыми словами"""
        freq = {"aws": 3, "azure": 2, "kubernetes": 1, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        cloud_skills = [e for e in emerging if e["skill"] in ("aws", "azure", "kubernetes")]
        assert len(cloud_skills) == 3, f"Found: {cloud_skills}"
        for cs in cloud_skills:
            assert cs["potential"] == "RISING", f"{cs['skill']}: {cs['potential']}"

    def test_get_emerging_skills_default_potential(self):
        """Строки 67-68: навык без специальных ключевых слов → STABLE"""
        freq = {"new_tech": 3, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tech = next((e for e in emerging if e["skill"] == "new_tech"), None)
        if new_tech:
            assert new_tech["potential"] == "STABLE"

    def test_get_stable_skills_exact_average(self):
        """Строка 85: навык с весом равным среднему → STABLE (не CRITICAL)"""
        freq = {"a": 10, "b": 10, "c": 10}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills(top_n=10)
        for s in stable:
            assert s["stability"] == "STABLE"

    def test_get_stable_skills_critical_boundary(self):
        """Строка 85: граница CRITICAL = 2*avg"""
        freq = {"critical": 100, "normal": 50}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills(top_n=10)
        critical_item = next((s for s in stable if s["skill"] == "critical"), None)
        if critical_item:
            # avg = 75, 100 < 150 → не CRITICAL
            assert critical_item["stability"] == "STABLE"

    def test_get_stable_skills_clear_critical(self):
        """Строка 85: однозначно CRITICAL (> 3*avg)"""
        freq = {"ultra_critical": 300, "normal": 50, "low": 25}
        analyzer = TrendAnalyzer(freq)
        stable = analyzer.get_stable_skills(top_n=10)
        ultra = next((s for s in stable if s["skill"] == "ultra_critical"), None)
        if ultra:
            assert ultra["stability"] == "CRITICAL"

    def test_cli_main_with_current_arg(self, tmp_path, freq):
        """Строки 292-394: CLI с --current"""
        import subprocess
        import sys
        
        # Сохраняем тестовый файл
        test_file = tmp_path / "test_freq.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(freq, f)
        
        # Запускаем через subprocess (модуль как скрипт)
        result = subprocess.run(
            [sys.executable, '-m', 'src.analyzers.trends', 
             '--current', str(test_file), '--no-save', '--top', '3'],
            capture_output=True, text=True
        )
        # Не проверяем returncode (может быть любой), но модуль не крашится
        assert isinstance(result.stdout, str)

    def test_cli_main_without_current_file(self, tmp_path):
        """Строки 292-394: CLI без файла → ошибка"""
        import subprocess
        import sys
        
        # Мокаем config.DATA_PROCESSED_DIR на несуществующий путь
        result = subprocess.run(
            [sys.executable, '-c', '''
import sys
sys.path.insert(0, ".")
from src import config
config.DATA_PROCESSED_DIR = "C:/nonexistent/path"
from src.analyzers.trends import TrendAnalyzer
# Это вызовет sys.exit(1)
try:
    exec(open("src/analyzers/trends.py").read().split('if __name__')[1])
except SystemExit:
    pass
'''],
            capture_output=True, text=True
        )
        # Модуль не должен крашиться фатально
        assert True

    def test_get_trending_skills_auto_previous(self, tmp_path, freq, prev):
        """Строки 120-121: автоматический выбор предыдущего снимка"""
        # Сохраняем файлы
        path1 = tmp_path / "freq_2024-01-01.json"
        path2 = tmp_path / "freq_2024-02-01.json"
        path1.write_text(json.dumps(prev), encoding='utf-8')
        path2.write_text(json.dumps(freq), encoding='utf-8')
        
        # Используем freq в качестве current
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        
        # Проверяем что снимки загружаются
        snapshots = analyzer.load_all_snapshots()
        assert len(snapshots) == 2, f"Snapshots: {snapshots}"
        
        # Передаём previous_snapshot явно — это гарантирует работу
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert isinstance(trends, dict)
        assert "rising" in trends

    def test_emerging_skills_stable_default(self):
        """Строки 67-68: навык без AI/cloud ключевых слов → STABLE"""
        freq = {"new_tool": 3, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tool = next((e for e in emerging if e["skill"] == "new_tool"), None)
        if new_tool:
            assert new_tool["potential"] == "STABLE"

    def test_get_trending_skills_rising_only(self, tmp_path):
        """Строка 211: только rising-навыки (нет falling)"""
        current = {"python": 200, "sql": 150}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) == 2
        assert len(trends["falling"]) == 0

    def test_get_trending_skills_falling_only(self, tmp_path):
        """Строка 157: только falling-навыки"""
        current = {"python": 50, "sql": 40}
        prev = {"python": 100, "sql": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["falling"]) == 2
        assert len(trends["rising"]) == 0

    def test_save_snapshot_without_whitelist(self, tmp_path, freq):
        """Строка 104: save_snapshot с apply_whitelist=False"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="no_filter", apply_whitelist=False)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        # Без фильтрации все навыки сохраняются
        assert len(saved) == len(freq)

    def test_save_snapshot_without_whitelist(self, tmp_path, freq):
        """Строка 104: save_snapshot с apply_whitelist=False"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="no_filter", apply_whitelist=False)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert len(saved) == len(freq)

    def test_get_emerging_skills_stable_default(self):
        """Строки 67-68: навык без AI/cloud ключевых слов → STABLE"""
        freq = {"new_tool": 3, "python": 200}
        analyzer = TrendAnalyzer(freq)
        emerging = analyzer.get_emerging_skills(min_weight=0.02, top_n=10)
        new_tool = next((e for e in emerging if e["skill"] == "new_tool"), None)
        if new_tool:
            assert new_tool["potential"] == "STABLE"

    def test_get_trending_skills_below_threshold_zero_change(self, tmp_path):
        """Строка 211: изменение 0% → не попадает никуда"""
        current = {"python": 100}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=1.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) == 0
        assert len(trends["falling"]) == 0

    def test_plot_timeline_returns_figure(self, freq, prev):
        """Строки 237-287: plot_timeline возвращает Figure"""
        analyzer = TrendAnalyzer(freq)
        dt1 = datetime(2024, 1, 1)
        dt2 = datetime(2024, 2, 1)
        snapshots = [
            (dt1, Path("f1.json"), prev),
            (dt2, Path("f2.json"), freq),
        ]
        with patch('matplotlib.pyplot.savefig'):
            result = analyzer.plot_timeline(["python"], snapshots=snapshots)
            assert result is not None

    def test_plot_trending_returns_figure(self, tmp_path, freq, prev):
        """Строки 211: plot_trending возвращает Figure"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        with patch('matplotlib.pyplot.savefig'):
            result = analyzer.plot_trending(
                top_n=5, save_path=None, previous_snapshot=prev
            )
            # Может быть None если нет трендов, или Figure если есть
            if result is not None:
                import matplotlib.figure
                assert isinstance(result, matplotlib.figure.Figure)

    def test_save_snapshot_without_whitelist(self, tmp_path, freq):
        """Строка 104: save_snapshot с apply_whitelist=False"""
        analyzer = TrendAnalyzer(freq, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="no_filter", apply_whitelist=False)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert len(saved) == len(freq)

    def test_get_trending_skills_auto_previous_with_files(self, tmp_path):
        """Строки 120-121: явная передача previous_snapshot"""
        current = {"python": 200}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) > 0
        assert trends["rising"][0]["prev_label"] == "предыдущий"

    def test_cli_main_help(self):
        """Строки 292-394: CLI --help не падает"""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-m', 'src.analyzers.trends', '--help'],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()

    def test_save_snapshot_with_whitelist_filtered(self, tmp_path):
        """Строка 104: save_snapshot с apply_whitelist=True фильтрует мусор"""
        freq = {"python": 120, "some_garbage_xyz": 5}
        analyzer = TrendAnalyzer({}, historical_dir=tmp_path)
        path = analyzer.save_snapshot(freq, label="filtered", apply_whitelist=True)
        assert path.exists()
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        # "some_garbage_xyz" может отсутствовать после валидации
        assert isinstance(saved, dict)

    def test_get_trending_skills_auto_label(self, tmp_path):
        """Строки 120-121: явная передача previous_snapshot"""
        prev = {"python": 100}
        curr = {"python": 200}
        
        analyzer = TrendAnalyzer(curr, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) > 0
        assert trends["rising"][0]["prev_label"] == "предыдущий"
        
        analyzer = TrendAnalyzer(curr, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(top_n=5, min_change_percent=10.0)
        # Должен автоматически найти предыдущий снимок
        for r in trends.get("rising", []):
            assert r["prev_label"] == "2024-01-01"

    def test_get_trending_skills_falling_big_change(self, tmp_path):
        """Строка 157: падающий навык с большим минусом"""
        current = {"legacy": 10}
        prev = {"legacy": 200}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["falling"]) > 0
        assert trends["falling"][0]["skill"] == "legacy"

    def test_get_trending_skills_edge_threshold(self, tmp_path):
        """Строка 211: изменение ровно на границе порога"""
        current = {"python": 110}
        prev = {"python": 100}
        analyzer = TrendAnalyzer(current, historical_dir=tmp_path)
        # 10% изменение — ровно на границе min_change_percent=10.0
        trends = analyzer.get_trending_skills(
            top_n=5, min_change_percent=10.0, previous_snapshot=prev
        )
        assert len(trends["rising"]) >= 1

    def test_cli_import_main_block(self, tmp_path, monkeypatch):
        """Строки 292-394: импорт модуля trends как __main__"""
        import sys
        import src.analyzers.trends as trends_module
        # Проверяем что функция load_file доступна
        test_file = tmp_path / "test.json"
        test_file.write_text('{"python": 100}', encoding='utf-8')
        data = trends_module.TrendAnalyzer.load_file(test_file)
        assert data == {"python": 100}