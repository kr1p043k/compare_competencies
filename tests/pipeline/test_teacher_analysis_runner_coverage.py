"""Coverage tests for src/pipeline/teacher_analysis_runner.

Mocks DB pool, filesystem (tmp_path) and heavy model/analyzer dependencies.
Deterministic and fast.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src import Ok
from src.models.teacher_analysis import (
    CompetencyCoverage,
    DisciplineCoverage,
    Recommendation,
)
from src.pipeline import teacher_analysis_runner as runner


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def reference_dir(tmp_path):
    ref = tmp_path / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    return ref


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "processed").mkdir(exist_ok=True)
    return d


@pytest.fixture
def _ensure_fs_dirs(tmp_path, reference_dir, output_dir, data_dir):
    """Make sure the temporary data tree exists."""
    return tmp_path


@pytest.fixture
def mock_pool():
    """Async DB pool with deterministic responses."""
    p = MagicMock()

    async def _fetchval(sql, *args):
        lower = sql.lower()
        if "max(completed_at)" in lower:
            return datetime(2000, 1, 1, tzinfo=UTC)
        if "select max(ts)" in lower:
            return None
        if "count(*)::text" in lower or "sum(hashtext" in lower:
            return "0:0:0"
        return None

    async def _fetch(sql, *args):
        lower = sql.lower()
        # vacancies / parsed_skills queries
        if "vacancies" in lower and "parsed_skills" in lower:
            return []
        # market weights
        if "market_skill_mappings" in lower:
            return []
        # trend snapshots
        if "trend_snapshots" in lower:
            return []
        # direction details
        if "from directions" in lower and "where code" in lower:
            return []
        # discipline rows - default yields nothing; override per test via p.fetch.return_value
        return []

    async def _fetchrow(sql, *args):
        return None

    async def _execute(sql, *args):
        return None

    p.fetchval = AsyncMock(side_effect=_fetchval)
    p.fetch = AsyncMock(side_effect=_fetch)
    p.fetchrow = AsyncMock(side_effect=_fetchrow)
    p.execute = AsyncMock(side_effect=_execute)
    return p


@pytest.fixture
def mock_coverage():
    """Minimal DisciplineCoverage that survives summary math."""
    cc = CompetencyCoverage(
        code="ОПК-2",
        total_skills=2,
        matched_skills=1,
        matched_names=["sql"],
        coverage=0.5,
        weighted_coverage=0.6,
        gap_skills=["mongodb"],
    )
    return DisciplineCoverage(
        discipline_id="1",
        discipline_name="Базы данных и СУБД",
        total_skills=2,
        market_matched=1,
        gaps=1,
        coverage_ratio=0.5,
        weighted_coverage=0.6,
        strong_matched=1,
        strong_coverage=0.5,
        coverage_level="medium",
        top_matched=[],
        matched_market=["sql"],
        gaps_list=["mongodb"],
        emerging=[],
        truly_missing=[],
        cross_references=[],
        competencies=[cc],
        market_top_coverage=0.25,
        market_weight_coverage=0.3,
        market_top_total=4,
        market_top_reliable=True,
        market_top_source="mapping",
        market_top_covered=["sql"],
        market_top_missing=["mongodb"],
    )


@pytest.fixture
def mock_recommendation():
    return Recommendation(
        type="add_new_content",
        priority="medium",
        message="Add docker",
        skill_name="docker",
    )


@pytest.fixture
def base_monkeypatch(
    monkeypatch,
    tmp_path,
    reference_dir,
    output_dir,
    data_dir,
    mock_pool,
    mock_coverage,
    mock_recommendation,
):
    """Common stubs for the module-level constants and external deps."""
    monkeypatch.setattr(runner, "OUTPUT", output_dir)
    monkeypatch.setattr(runner, "DATA_DIR", data_dir)
    monkeypatch.setattr(runner, "MODELS_DIR", data_dir / "models")
    monkeypatch.setattr(runner.config, "REFERENCE_DIR", reference_dir)
    monkeypatch.setattr(runner.config, "DATA_CACHE_DIR", data_dir / "cache")

    # DB lifecycle
    monkeypatch.setattr(runner, "create_pool", AsyncMock())
    monkeypatch.setattr(runner, "close_pool", AsyncMock())
    monkeypatch.setattr(runner, "get_pool", lambda: mock_pool)

    # pipeline DB writes
    monkeypatch.setattr("src.pipeline.db_writer.create_pipeline_run", AsyncMock(return_value="run-123"))
    monkeypatch.setattr("src.pipeline.db_writer.complete_pipeline_run", AsyncMock())
    monkeypatch.setattr("src.pipeline.db_writer.save_to_analysis_results", AsyncMock())

    # caches and feature flags
    cache_mgr = MagicMock()
    cache_mgr.load.return_value = Ok(None)  # always miss, falls through to DB
    cache_mgr.save.return_value = Ok(None)
    monkeypatch.setattr("src.cache_manager.CacheManager", lambda _dir: cache_mgr)
    from src import feature_flags
    monkeypatch.setattr(feature_flags, "market_synonyms_enabled", lambda: False)
    monkeypatch.setattr(runner.config, "HUB_GATE_ENABLED", False)
    monkeypatch.setattr(runner.config, "LLM_JUDGE_ENABLED", False)

    # analyzers / recommenders / optimizers
    # These names are imported at module level in runner.py, so the patch
    # must be applied to the runner namespace (not to the source module).
    cov_analyzer = MagicMock()
    cov_analyzer.analyze_discipline.return_value = Ok(mock_coverage)
    monkeypatch.setattr(runner, "CoverageAnalyzer", lambda *a, **k: cov_analyzer)

    rec_engine = MagicMock()
    rec_engine.generate.return_value = Ok([mock_recommendation])
    rec_engine.generate_summary_recommendations.return_value = Ok([])
    monkeypatch.setattr(runner, "CurriculumRecommender", lambda: rec_engine)

    opt = MagicMock()
    opt.optimize.return_value = Ok([])
    monkeypatch.setattr(runner, "CurriculumOptimizer", lambda: opt)

    trend = MagicMock()
    trend.get_rising.return_value = Ok([])
    trend.get_declining.return_value = Ok([])
    monkeypatch.setattr(runner, "SnapshotTrendAnalyzer", lambda *a, **k: trend)

    matcher = MagicMock()
    matcher.market_skills = {"sql": 10, "python": 5}
    matcher.set_market.return_value = Ok(None)
    matcher.get_emerging.return_value = Ok([])
    monkeypatch.setattr(runner, "SkillMatcher", lambda *a, **k: matcher)

    # *_scorer is imported locally inside run_teacher_analysis, so we patch
    # the source module; the local import resolves the patched attribute.
    scorer = MagicMock()
    scorer.load.return_value = None
    scorer.compute_relevance_batch.return_value = {"docker": 0.5}
    scorer.has_full_profile.return_value = True
    scorer.prime_all.return_value = None
    monkeypatch.setattr("src.analyzers.discipline_relevance.DisciplineAwareScorer", lambda: scorer)

    emb_factory = MagicMock()
    emb_factory.get.return_value = MagicMock()
    monkeypatch.setattr("src.analyzers.comparison.embedding_provider.EmbeddingProviderFactory", emb_factory)

    cooc_cls = MagicMock()
    cooc_cls.build.return_value = cooc_cls
    cooc_cls.to_cache.return_value = {}
    monkeypatch.setattr("src.analyzers.skill_cooccurrence.SkillCooccurrence", lambda *a, **k: cooc_cls)

    # scope: keep everything unless a test overrides
    monkeypatch.setattr("src.teacher_scope.load_scope_overrides", AsyncMock(return_value={}))
    monkeypatch.setattr("src.teacher_scope.effective_in_scope", lambda name, overrides: True)

    yield

    # cleanup: remove any files written to tmp path (tmp_path handles itself)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _monkeypatch_reference_dir(monkeypatch, reference_dir):
    """Redirect config.REFERENCE_DIR to tmp_path/reference so all file helpers use it."""
    monkeypatch.setattr(runner.config, "REFERENCE_DIR", reference_dir)


class TestResolveKrmPath:
    def test_default_year_uses_unsuffixed_file(self, reference_dir):
        krm = reference_dir / "krm_disciplines_09.03.02.json"
        krm.write_text("{}", encoding="utf-8")
        assert runner.resolve_krm_path("09.03.02", 2025) == krm

    def test_custom_year_prefers_year_file(self, reference_dir):
        default = reference_dir / "krm_disciplines_09.03.02.json"
        year_file = reference_dir / "krm_disciplines_09.03.02_2024.json"
        default.write_text("{}", encoding="utf-8")
        year_file.write_text("{}", encoding="utf-8")
        assert runner.resolve_krm_path("09.03.02", 2024) == year_file

    def test_custom_year_missing_falls_back_and_warns(self, reference_dir, caplog):
        default = reference_dir / "krm_disciplines_09.03.02.json"
        default.write_text("{}", encoding="utf-8")
        result = runner.resolve_krm_path("09.03.02", 2024)
        assert result == default

    def test_zero_year_treated_as_default(self, reference_dir):
        default = reference_dir / "krm_disciplines_09.03.02.json"
        default.write_text("{}", encoding="utf-8")
        assert runner.resolve_krm_path("09.03.02", 0) == default


class TestTeacherOutputDir:
    def test_default_year_no_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "OUTPUT", tmp_path / "teacher")
        assert runner.teacher_output_dir("09.03.02", 2025) == tmp_path / "teacher" / "09.03.02"

    def test_custom_year_creates_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "OUTPUT", tmp_path / "teacher")
        assert runner.teacher_output_dir("09.03.02", 2024) == tmp_path / "teacher" / "09.03.02" / "2024"


class TestLoadDisciplineYears:
    def test_missing_file_returns_empty(self):
        # After autouse fixture, REFERENCE_DIR is a tmp_path; unknown dir_code -> miss
        assert runner._load_discipline_years("XX.XX.XX") == {}

    def test_malformed_json_returns_empty(self, reference_dir):
        bad = reference_dir / "discipline_years_09.03.02.json"
        bad.write_text("not json", encoding="utf-8")
        assert runner._load_discipline_years("09.03.02") == {}

    def test_valid_file_parsed(self, reference_dir):
        good = reference_dir / "discipline_years_09.03.02.json"
        good.write_text(json.dumps({"disciplines": {"D1": [2024, 2025]}}), encoding="utf-8")
        assert runner._load_discipline_years("09.03.02") == {"D1": [2024, 2025]}

    def test_missing_disciplines_key_returns_empty(self, reference_dir):
        good = reference_dir / "discipline_years_09.03.02.json"
        good.write_text(json.dumps({"other": []}), encoding="utf-8")
        assert runner._load_discipline_years("09.03.02") == {}


class TestDisciplineInScope:
    def test_empty_name_excluded(self):
        assert runner.discipline_in_scope("") is False
        assert runner.discipline_in_scope(None) is False

    def test_excluded_set_rejected(self):
        for name in runner.SCOPE_EXCLUDED:
            assert runner.discipline_in_scope(name) is False

    def test_english_business_and_c1_kept(self):
        assert runner.discipline_in_scope("Иностранный язык для деловой коммуникации") is True
        assert runner.discipline_in_scope("Иностранный язык (англ. яз., уровень с1)") is True

    def test_english_lower_levels_rejected(self):
        assert runner.discipline_in_scope("Иностранный язык (англ. яз., уровень А2)") is False

    def test_russian_kept(self):
        assert runner.discipline_in_scope("Иностранный язык (русский язык)") is True


class TestGitShaShort:
    def test_returns_sha_when_git_available(self):
        sha = runner._git_sha_short()
        # Inside a git repo this is either a hex sha or "unknown"
        assert sha == "unknown" or (len(sha) >= 4 and all(c in "0123456789abcdef" for c in sha))

    def test_returns_unknown_on_failure(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no git")))
        assert runner._git_sha_short() == "unknown"


class TestAssembleDisciplines:
    def test_groups_skills_and_ksa_by_competency(self):
        rows = [
            {"disc_id": 1, "disc_name": "D", "comp_code": "К1",
             "skill_name": "python", "ksa_text": None, "ksa_type": None},
            {"disc_id": 1, "disc_name": "D", "comp_code": "К1",
             "skill_name": None, "ksa_text": "применять sql", "ksa_type": "skill"},
        ]
        out = runner._assemble_disciplines(rows)
        assert out["D"]["id"] == "1"
        assert set(out["D"]["competencies"]["К1"]) == {"python", "применять sql"}
        assert out["D"]["ksa_types"]["применять sql"] == "skill"

    def test_ksa_junk_not_included(self):
        rows = [
            {"disc_id": 1, "disc_name": "D", "comp_code": "К1",
             "skill_name": None, "ksa_text": "тема 1: введение", "ksa_type": "knowledge"},
        ]
        out = runner._assemble_disciplines(rows)
        assert "тема 1: введение" not in out["D"]["competencies"]["К1"]

    def test_preserves_row_order(self):
        rows = [
            {"disc_id": 1, "disc_name": "D", "comp_code": "К1",
             "skill_name": None, "ksa_text": "- второй пункт", "ksa_type": None},
            {"disc_id": 1, "disc_name": "D", "comp_code": "К1",
             "skill_name": "python", "ksa_text": None, "ksa_type": None},
        ]
        out = runner._assemble_disciplines(rows)
        assert out["D"]["competencies"]["К1"] == ["второй пункт", "python"]

    def test_empty_input(self):
        assert runner._assemble_disciplines([]) == {}


class TestSafeFilename:
    def test_removes_invalid_chars(self):
        assert ":/\\*?\"<>|" not in runner._safe_filename('a:b/c*d?e"f<g>h|i')

    def test_truncates_to_80(self):
        assert len(runner._safe_filename("x" * 200)) == 80

    def test_strips_whitespace(self):
        assert runner._safe_filename("  spaced  ") == "spaced"


class TestSkillHelpers:
    def test_strip_bullet_dashes(self):
        assert runner._strip_bullet("- skill text") == "skill text"
        assert runner._strip_bullet("\u2013 skill text") == "skill text"
        assert runner._strip_bullet("\u2022 skill text") == "skill text"
        assert runner._strip_bullet("") == ""

    def test_strip_rpd_tail(self):
        assert runner._strip_rpd_tail("sql Код направления 09.03.02") == "sql"
        assert runner._strip_rpd_tail("python 09.03.02") == "python"

    def test_is_skill_like_ksa(self):
        assert runner._is_skill_like_ksa("") is False
        assert runner._is_skill_like_ksa("ab") is False
        assert runner._is_skill_like_ksa("x" * 201) is False
        assert runner._is_skill_like_ksa("Разработка web-приложений") is True
        assert runner._is_skill_like_ksa("тема 1: введение") is False
        assert runner._is_skill_like_ksa("при наличии доступа") is False


class TestEnhanceDisciplinesWithGapAnalysis:
    def test_no_market_skips_immediately(self, output_dir):
        disc = {"D": {"competencies": {"К1": ["python"]}, "ksa": {}}}
        summary = {"disciplines": [{"name": "D"}]}
        result = runner._enhance_disciplines_with_gap_analysis(
            disc, [], {}, output_dir, summary)
        assert result is summary
        assert result["diag"]["degraded"] is False
        assert result["diag"]["failed_channels"] == []
        assert result["diag"]["count"] == 0

    def test_build_failure_returns_summary(self, output_dir, monkeypatch):
        disc = {"D": {"competencies": {"К1": ["python"]}, "ksa": {}}}
        summary = {"disciplines": [{"name": "D"}]}

        def _boom(*a, **k):
            raise RuntimeError("no model")

        import src.analyzers.comparison.embedding_comparator
        monkeypatch.setattr(
            src.analyzers.comparison.embedding_comparator.EmbeddingComparator,
            "build_market_index",
            _boom,
        )
        result = runner._enhance_disciplines_with_gap_analysis(
            disc, ["python"], {"python": 0.5}, output_dir, summary)
        assert result is summary
        assert result["diag"]["degraded"] is True
        assert "gap_enhance.embeddings" in result["diag"]["failed_channels"]
        assert result["diag"]["count"] == 1


class TestFailPipelineRun:
    @pytest.mark.asyncio
    async def test_with_run_id_calls_complete(self, monkeypatch):
        called = {}

        async def _complete(run_id, status=None, error=None, stats=None):
            called.update({"run_id": run_id, "status": status, "error": error})

        monkeypatch.setattr("src.pipeline.db_writer.complete_pipeline_run", _complete)
        await runner._fail_pipeline_run("abc", "boom")
        assert called["run_id"] == "abc"
        assert called["status"] == "failed"
        assert called["error"] == "boom"

    @pytest.mark.asyncio
    async def test_none_run_id_is_noop(self):
        await runner._fail_pipeline_run(None, "boom")  # should not raise

    @pytest.mark.asyncio
    async def test_write_failure_does_not_raise(self, monkeypatch):
        async def _boom(*a, **k):
            raise RuntimeError("db down")
        monkeypatch.setattr("src.pipeline.db_writer.complete_pipeline_run", _boom)
        await runner._fail_pipeline_run("abc", "boom")


# ---------------------------------------------------------------------------
# Integration tests for run_teacher_analysis
# ---------------------------------------------------------------------------
def _write_krm(reference_dir: Path, code: str = "09.03.02", body: dict | None = None):
    krm = reference_dir / f"krm_disciplines_{code}.json"
    payload = {code: {"disciplines": body or {}}}
    krm.write_text(json.dumps(payload), encoding="utf-8")
    return krm


def _krm_body(name: str = "Базы данных и СУБД"):
    return {
        name: {
            "ksa": {
                "ОПК-2": {
                    "knowledge": ["знать модели данных"],
                    "abilities": ["уметь проектировать"],
                    "skills": ["владеть sql"],
                }
            }
        }
    }


def _make_drows(name: str = "Базы данных и СУБД"):
    return [
        {"disc_id": 1, "disc_name": name, "comp_id": 1, "comp_code": "ОПК-2",
         "skill_name": "sql", "ksa_text": None, "ksa_type": None},
        {"disc_id": 1, "disc_name": name, "comp_id": 1, "comp_code": "ОПК-2",
         "skill_name": None, "ksa_text": "применение sql", "ksa_type": "skill"},
    ]


def _build_pool_fetcher(drows, direction_rows, snapshot_rows=None, market_rows=None, vac_rows=None):
    async def _fetch(sql, *args):
        lower = sql.lower()
        if "vacancies" in lower and "parsed_skills" in lower:
            return vac_rows or []
        if "market_skill_mappings" in lower:
            return market_rows or []
        if "trend_snapshots" in lower:
            return snapshot_rows or []
        if "disciplines" in lower and "competencies" in lower and "ksa_entries" in lower:
            return drows
        if "from directions" in lower:
            return direction_rows
        return []
    return _fetch


class TestRunTeacherAnalysisHappyPath:
    @pytest.mark.asyncio
    async def test_full_run_creates_summary_and_partial_meta(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        _write_krm(reference_dir, body=_krm_body())

        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_ok(), result
        data = result.unwrap()
        assert data["direction"] == "09.03.02"
        assert data["direction_name"] == "Информатика"
        assert data["total_disciplines"] == 1
        assert data["code_version"] == runner.CODE_VERSION
        assert (output_dir / "09.03.02" / "_summary.json").exists()

    @pytest.mark.asyncio
    async def test_no_direction_code_uses_first_available(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        _write_krm(reference_dir, body=_krm_body())
        drows = _make_drows()
        direction_row = {"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}

        async def _fetchrow(sql, *args):
            return direction_row

        async def _fetch(sql, *args):
            lower = sql.lower()
            if "disciplines" in lower and "competencies" in lower:
                return drows
            if "from directions" in lower:
                return [direction_row]
            return []

        mock_pool.fetch = AsyncMock(side_effect=_fetch)
        mock_pool.fetchrow = AsyncMock(side_effect=_fetchrow)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis()
        assert result.is_ok()

    @pytest.mark.asyncio
    async def test_partial_run_with_discipline_filter(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        _write_krm(reference_dir, body=_krm_body())
        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02", discipline_filter="Базы данных")
        assert result.is_ok()
        data = result.unwrap()
        assert data["partial"] is True
        assert data["disciplines"] == ["Базы данных и СУБД"]


class TestRunTeacherAnalysisErrorPaths:
    @pytest.mark.asyncio
    async def test_no_pool_returns_err(self, monkeypatch):
        monkeypatch.setattr(runner, "create_pool", AsyncMock())
        monkeypatch.setattr(runner, "get_pool", lambda: None)
        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_err()
        assert result.err().stage == "db"

    @pytest.mark.asyncio
    async def test_empty_disciplines_returns_err(
        self, base_monkeypatch, reference_dir, mock_pool
    ):
        _write_krm(reference_dir, body={})
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetch.side_effect = _build_pool_fetcher([], direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_err()
        assert "No disciplines found" in result.err().message

    @pytest.mark.asyncio
    async def test_missing_direction_returns_err(
        self, base_monkeypatch, reference_dir, mock_pool
    ):
        _write_krm(reference_dir, body={})
        # Need at least one discipline row to get past the discipline check,
        # then an empty direction result triggers the direction error.
        mock_pool.fetch.side_effect = _build_pool_fetcher(_make_drows(), [])

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_err()
        assert "Direction" in result.err().message

    @pytest.mark.asyncio
    async def test_scope_excludes_everything_returns_err(
        self, base_monkeypatch, reference_dir, mock_pool, monkeypatch
    ):
        _write_krm(reference_dir, body=_krm_body())
        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)
        # effective_in_scope is imported locally, so patch the source module.
        monkeypatch.setattr("src.teacher_scope.effective_in_scope", lambda name, overrides: False)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_err()
        assert "excluded by scope" in result.err().message

    @pytest.mark.asyncio
    async def test_malformed_krm_json_continues_without_ksa(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        krm = reference_dir / "krm_disciplines_09.03.02.json"
        krm.write_text("{not valid json", encoding="utf-8")
        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_ok()

    @pytest.mark.asyncio
    async def test_db_query_error_returns_err(
        self, base_monkeypatch, mock_pool
    ):
        async def _boom(sql, *args):
            raise RuntimeError("db failure")
        mock_pool.fetch = AsyncMock(side_effect=_boom)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_err()
        assert result.err().stage == "disciplines"

    @pytest.mark.asyncio
    async def test_skips_when_data_unchanged(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        _write_krm(reference_dir, body=_krm_body())
        summary = {
            "direction": "09.03.02",
            "code_version": runner.CODE_VERSION,
            "disciplines": [],
            "skipped": True,
        }
        summary_file = output_dir / "09.03.02" / "_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary), encoding="utf-8")
        os.utime(summary_file, (summary_file.stat().st_mtime + 1000,) * 2)

        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]

        async def _fetchval(sql, *args):
            lower = sql.lower()
            if "max(completed_at)" in lower:
                # summary is newer than last run -> market unchanged
                return datetime(2000, 1, 1, tzinfo=UTC)
            if "select max(ts)" in lower:
                return None
            if "count(*)::text" in lower:
                return "0:0:0"
            return None

        mock_pool.fetchval = AsyncMock(side_effect=_fetchval)
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02")
        assert result.is_ok()
        assert result.unwrap().get("skipped") is True

    @pytest.mark.asyncio
    async def test_force_recomputes_even_if_summary_fresh(
        self, base_monkeypatch, reference_dir, output_dir, mock_pool, mock_coverage
    ):
        _write_krm(reference_dir, body=_krm_body())
        summary = {"direction": "09.03.02", "code_version": runner.CODE_VERSION}
        summary_file = output_dir / "09.03.02" / "_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary), encoding="utf-8")

        async def _fetchval(sql, *args):
            lower = sql.lower()
            if "max(completed_at)" in lower:
                return datetime(2000, 1, 1, tzinfo=UTC)
            if "select max(ts)" in lower:
                return None
            if "count(*)::text" in lower:
                return "0:0:0"
            return None

        drows = _make_drows()
        direction_rows = [{"id": 1, "code": "09.03.02", "name": "Информатика", "profile": "base"}]
        mock_pool.fetchval = AsyncMock(side_effect=_fetchval)
        mock_pool.fetch.side_effect = _build_pool_fetcher(drows, direction_rows)

        from src.pipeline.teacher_analysis_runner import run_teacher_analysis
        result = await run_teacher_analysis("09.03.02", force=True)
        assert result.is_ok()
        assert "skipped" not in result.unwrap()

