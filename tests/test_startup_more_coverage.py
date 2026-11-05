"""Coverage tests for src/api_pkg/startup.py.

Deterministic, no network/DB/model downloads. Heavy dependencies mocked.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open
from unittest.mock import patch

import pytest

import src.api_pkg.deps as deps
import src.api_pkg.startup as startup
from src import Ok


class _Ok:
    @staticmethod
    def is_err() -> bool:
        return False

    @staticmethod
    def err():
        return None


class _Err:
    @staticmethod
    def is_err() -> bool:
        return True

    @staticmethod
    def err():
        return "boom"


@pytest.fixture(autouse=True)
def _clean_deps():
    """Reset shared state before each test."""
    deps.evaluator = None
    deps.recommendation_engine = None
    deps.trend_analyzer = None
    deps.student_profiles.clear()
    deps.skill_weights.clear()
    deps.hybrid_weights.clear()
    deps.competency_mapping.clear()
    deps.skill_freq.clear()
    deps.taxonomy = None
    deps.current_skills_set.clear()
    deps.basic_vacancies.clear()
    deps.raw_file = None
    deps.vacancy_load_error = None
    deps.prophet_engine = None
    deps.skill_engine = None
    deps.is_ready = False
    yield


# --------------------------------------------------------------------------- helpers


def test_read_json_reads_dict(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    assert startup._read_json(path) == {"x": 1}


def test_available_ram_bytes_parses_linux_meminfo(monkeypatch):
    m = mock_open(read_data="MemAvailable: 1500 kB\n")
    monkeypatch.setattr("builtins.open", m)
    assert startup._available_ram_bytes() == 1500 * 1024


def test_available_ram_bytes_returns_none_on_error(monkeypatch):
    def _broken(*_a, **_kw):
        raise OSError("no proc")

    monkeypatch.setattr("builtins.open", _broken)
    assert startup._available_ram_bytes() is None


# --------------------------------------------------------------------------- custom profiles


def test_register_custom_student_profiles(tmp_path):
    sdir = tmp_path / "students"
    sdir.mkdir()
    good = {
        "компетенции": ["SS1.1"],
        "навыки": ["python"],
        "target_level": "middle",
    }
    (sdir / "good_competency.json").write_text(json.dumps(good), encoding="utf-8")
    already = {}
    restored = startup.register_custom_student_profiles(sdir, already, lambda c: [str(c).upper()])
    assert "good" in restored
    assert already["good"].profile_name == "good"


def test_register_custom_student_profiles_skips_invalid(tmp_path):
    sdir = tmp_path / "students"
    sdir.mkdir()
    invalid = {"not": "a profile"}
    (sdir / "empty_competency.json").write_text(json.dumps(invalid), encoding="utf-8")
    (sdir / "bad name_competency.json").write_text("{}", encoding="utf-8")
    assert startup.register_custom_student_profiles(sdir, {}, lambda c: []) == []


# --------------------------------------------------------------------------- warmup notifications


def test_notify_warmup_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.notifications.system.queue_system_error",
        lambda *a, **k: calls.append((a, k)),
    )
    startup._notify_warmup_failure("comp", "oops")
    assert calls
    assert "comp" in calls[0][0][0]


def test_resolve_warmup_failure(monkeypatch):
    resolved = []

    async def fake(component):
        resolved.append(component)

    monkeypatch.setattr("src.notifications.system.resolve_warmup_failure", fake)
    asyncio.run(startup._resolve_warmup_failure("x"))
    assert resolved == ["x"]


# --------------------------------------------------------------------------- pipeline error resolution


def test_resolve_pipeline_errors_if_ready(monkeypatch):
    resolved_stages = []
    resolved_failed = []

    async def fake_stage(stage):
        resolved_stages.append(stage)

    async def fake_failed():
        resolved_failed.append(True)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        "src.notifications.system.resolve_pipeline_stage_errors", fake_stage
    )
    monkeypatch.setattr("src.notifications.system.resolve_pipeline_failed", fake_failed)
    asyncio.run(startup._resolve_pipeline_errors_if_ready())
    assert "data_collection" in resolved_stages
    assert resolved_failed


# --------------------------------------------------------------------------- waiting mode / business gauges


def test_set_waiting_mode_sets_ready(monkeypatch):
    class FakeClusterer:
        def load_model(self, _level):
            return True

    monkeypatch.setattr(deps, "get_clusterer", lambda: FakeClusterer())
    asyncio.run(startup._set_waiting_mode())
    assert deps.is_ready is True
    assert deps.vacancy_load_error == "not_found"


def test_refresh_business_gauges(monkeypatch):
    calls = []

    class VacGauge:
        def set(self, v): calls.append(("vacancies", v))

    class ProfGauge:
        def set(self, v): calls.append(("profiles", v))

    class SkillGauge:
        def set(self, v): calls.append(("skills", v))

    deps.basic_vacancies = [1, 2]
    deps.student_profiles = {"a": None}
    deps.skill_weights = {"x": 1}
    monkeypatch.setattr("src.monitoring.metrics.vacancies_loaded", VacGauge())
    monkeypatch.setattr("src.monitoring.metrics.active_profiles", ProfGauge())
    monkeypatch.setattr("src.monitoring.metrics.skill_count", SkillGauge())
    startup._refresh_business_gauges()
    assert dict(calls) == {"vacancies": 2, "profiles": 1, "skills": 1}


# --------------------------------------------------------------------------- DB cleanup helpers


def test_finalize_stale_pipeline_runs(monkeypatch):
    sql_calls = []

    class FakePool:
        async def execute(self, sql):
            sql_calls.append(sql)
            return "UPDATE 3"

    monkeypatch.setattr("src.db.get_pool", lambda: FakePool())
    asyncio.run(startup._finalize_stale_pipeline_runs())
    assert any("pipeline_runs" in str(s) for s in sql_calls)


def test_purge_expired_sessions_no_pool(monkeypatch):
    monkeypatch.setattr("src.db.get_pool", lambda: None)
    asyncio.run(startup._purge_expired_sessions())


async def _noop_coro(*_a, **_kw):
    return None


def test_run_bootstrap_safely_catches_and_notifies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.notifications.system.queue_system_error",
        lambda *a, **k: calls.append((a, k)),
    )
    async def _broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(startup, "_bootstrap", _broken)
    asyncio.run(startup._run_bootstrap_safely())
    assert any("bootstrap" in str(c[0]) for c in calls)


def test_ensure_dirs_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(startup.config.settings, "DATA_RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(startup.config.settings, "STUDENTS_DIR", tmp_path / "students")
    startup.config.settings.ensure_dirs()
    assert startup.config.settings.DATA_RAW_DIR.exists()
    assert startup.config.settings.STUDENTS_DIR.exists()


# --------------------------------------------------------------------------- bootstrap branches


def test_bootstrap_no_vacancy_files(monkeypatch):
    from src.api_pkg.routers import pipeline as pl_mod
    monkeypatch.setattr(asyncio, "ensure_future", lambda coro, *a, **k: coro)
    monkeypatch.setattr(asyncio, "create_task", lambda coro, *a, **k: MagicMock())
    monkeypatch.setattr(pl_mod, "_load_tasks", lambda: None)
    monkeypatch.setattr(startup, "_finalize_stale_pipeline_runs", _noop_coro)
    monkeypatch.setattr(startup, "_purge_expired_sessions", _noop_coro)
    monkeypatch.setattr(startup, "_resolve_pipeline_errors_if_ready", _noop_coro)
    monkeypatch.setattr(startup, "_periodic_housekeeping", _noop_coro)
    monkeypatch.setattr(startup, "_warmup_background", _noop_coro)
    monkeypatch.setattr(startup, "_set_waiting_mode", _noop_coro)
    monkeypatch.setattr(startup.config, "DATA_PROCESSED_DIR", Path("/nonexistent/processed"))
    monkeypatch.setattr(startup.config, "DATA_RAW_DIR", Path("/nonexistent/raw"))
    asyncio.run(startup._bootstrap())


# --------------------------------------------------------------------------- warmup background


@pytest.fixture
def warmup_mocks(monkeypatch, tmp_path):
    """Heavy dependencies replaced by fast mocks; only happy path left."""
    deps.competency_mapping = {"SS1.1": ["python"]}
    deps.skill_freq = {}
    deps.skill_weights = {}
    deps.hybrid_weights = {}
    deps.student_profiles.clear()
    deps.current_skills_set = {"python"}

    fake_parser = MagicMock()
    fake_parser.extract_skills_from_vacancies.return_value = Ok(
        {"frequencies": {"python": 10}, "hybrid_weights": {"python": 1.0}}
    )
    fake_parser.extract_skills_from_description.return_value = Ok(["python"])
    monkeypatch.setattr(startup, "VacancyParser", lambda: fake_parser)

    fake_cache = MagicMock()
    fake_cache.load.return_value = Ok("bogus")
    fake_cache.save.return_value = None
    monkeypatch.setattr(startup, "CacheManager", lambda p: fake_cache)

    fake_filter = MagicMock()
    fake_filter.get_clean_weights.return_value = Ok({"python": 1.0})
    monkeypatch.setattr(startup, "SkillFilter", lambda: fake_filter)

    monkeypatch.setattr(startup, "load_it_skills", lambda: {"python"})
    monkeypatch.setattr(startup, "load_competency_mapping", lambda: {"SS1.1": ["python"]})
    monkeypatch.setattr(startup, "filter_skills_by_whitelist", lambda d, s: d)
    monkeypatch.setattr(startup, "merge_skills_hierarchically", lambda *a, **k: ["python"])

    def _skill_taxonomy(*a, **k):
        raise RuntimeError("taxonomy unavailable")

    monkeypatch.setattr(startup, "SkillTaxonomy", _skill_taxonomy)

    fake_trend = MagicMock()
    monkeypatch.setattr(startup, "TrendAnalyzer", lambda *a, **k: fake_trend)

    fake_level = MagicMock()
    fake_level.get_weights_for_level.return_value = Ok({"python": 0.5})
    monkeypatch.setattr(startup, "SkillLevelAnalyzer", lambda: fake_level)

    fake_eval = MagicMock()
    fake_rec = MagicMock()
    fake_rec.fit.return_value = Ok(None)
    monkeypatch.setattr(startup, "ProfileEvaluator", lambda **k: fake_eval)
    monkeypatch.setattr(startup, "RecommendationEngine", lambda **k: fake_rec)

    class FakeNormalizer:
        @staticmethod
        def normalize(s):
            return Ok(s.lower())
    monkeypatch.setattr(startup, "SkillNormalizer", FakeNormalizer())

    class FakeClusterer:
        def load_model(self, _level):
            return True
    monkeypatch.setattr(deps, "get_clusterer", lambda: FakeClusterer())

    from src.pipeline.skill_extractor import _parser_code_version as original_version
    monkeypatch.setattr(
        "src.pipeline.skill_extractor._parser_code_version",
        lambda: original_version() + "_test",
    )

    # Patch out metrics-gauge calls to avoid importing prometheus-state specifics.
    monkeypatch.setattr(startup, "_refresh_business_gauges", lambda: None)

    return {"tmp_path": tmp_path, "parser": fake_parser}


def _write_student_files(students_dir: Path):
    students_dir.mkdir(parents=True, exist_ok=True)
    for name in ("base", "dc", "top_dc"):
        path = students_dir / f"{name}_competency.json"
        path.write_text(json.dumps({"компетенции": ["SS1.1"]}), encoding="utf-8")


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "hh.json"
    path.write_text("[]", encoding="utf-8")
    return path


def test_warmup_background_profiles_and_prophet_success(
    warmup_mocks, raw_file, tmp_path, monkeypatch
):
    _write_student_files(tmp_path / "data" / "students")

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *_exc): return False

    import src.database as db_mod
    monkeypatch.setattr(db_mod, "async_session_factory", FakeSession)

    async def fake_load_time_series(_session):
        return Ok([{"date": "2026-01-01", "skill": "python", "count": 10}])

    import src.predictors.prophet_forecast as pf_mod
    monkeypatch.setattr(pf_mod, "load_time_series", fake_load_time_series)

    class FakeEngine:
        def __init__(self):
            self._models = {"python": object()}

        async def fit_async(self, *_a, **_kw):
            return Ok(self)

    monkeypatch.setattr(pf_mod, "ProphetForecastEngine", FakeEngine)
    monkeypatch.setattr(startup, "_available_ram_bytes", lambda: 64 * 1024**3)

    deps.prophet_engine = None
    vac = {
        "name": "Python dev",
        "description": "python sql",
        "snippet": {"requirement": "py", "responsibility": "sql"},
        "experience": {"id": "between1and3"},
    }
    asyncio.run(startup._warmup_background([vac], raw_file))

    assert "dc" in deps.student_profiles
    assert "top_dc" in deps.student_profiles
    assert deps.prophet_engine is not None


def test_warmup_background_prophet_skipped_low_memory(warmup_mocks, raw_file, tmp_path, monkeypatch):
    _write_student_files(tmp_path / "data" / "students")
    monkeypatch.setattr(startup, "_available_ram_bytes", lambda: 0)
    vac = {"name": "python", "description": "", "experience": "middle"}
    deps.prophet_engine = None
    asyncio.run(startup._warmup_background([vac], raw_file))
    assert deps.prophet_engine is None


def test_warmup_background_prophet_fit_fails(warmup_mocks, raw_file, tmp_path, monkeypatch):
    _write_student_files(tmp_path / "data" / "students")
    monkeypatch.setattr(startup, "_available_ram_bytes", lambda: 64 * 1024**3)
    async def fake_load_time_series(_s):
        return Ok([1])

    import src.predictors.prophet_forecast as pf_mod
    monkeypatch.setattr(pf_mod, "load_time_series", fake_load_time_series)

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *_exc): return False

    import src.database as db_mod
    monkeypatch.setattr(db_mod, "async_session_factory", FakeSession)

    from src.result import Err

    class BrokenEngine:
        def __init__(self):
            self._models = {}

        async def fit_async(self, *_a, **_kw):
            # Настоящий Err: startup сопоставляет результат через `case Err(e)`,
            # и поддельный объект не попадёт ни в одну ветку.
            return Err(RuntimeError("fit failed"))

    monkeypatch.setattr(pf_mod, "ProphetForecastEngine", BrokenEngine)
    deps.prophet_engine = "unset"
    asyncio.run(startup._warmup_background([{"name": "x"}], raw_file))
    assert deps.prophet_engine is None


def test_warmup_background_student_loader_isolation(warmup_mocks, raw_file, tmp_path, monkeypatch):
    """One corrupted profile read must not break loading of the others."""
    students_dir = tmp_path / "data" / "students"
    students_dir.mkdir(parents=True, exist_ok=True)
    base = students_dir / "base_competency.json"
    base.write_text("not json", encoding="utf-8")
    for name in ("dc", "top_dc"):
        (students_dir / f"{name}_competency.json").write_text(
            json.dumps({"компетенции": ["SS1.1"]}), encoding="utf-8"
        )

    original_read = startup._read_json

    def _patched_read(path):
        if str(path).endswith("base_competency.json"):
            raise ValueError("corrupt base")
        return original_read(path)

    monkeypatch.setattr(startup, "_read_json", _patched_read)
    asyncio.run(startup._warmup_background([{"name": "x"}], raw_file))
    assert "dc" in deps.student_profiles
    assert "top_dc" in deps.student_profiles
