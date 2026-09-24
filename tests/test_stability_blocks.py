"""Точечные тесты стабилизационных фиксов (блоки 1-7, без LLM/n8n).

Быстрые pure-unit тесты нового кода: идемпотентность, auth-приоритет,
гео-ошибки, _run_async_safely, mape=None, кэп трендов, капы корреляций,
Err пустой карты доменов, PIPELINE_RETRIES.
"""

import py_compile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import Request
from fastapi.exceptions import HTTPException


def _req(headers=None, query=b"", cookies=""):
    scope = {
        "type": "http",
        "path": "/",
        "headers": [(k.lower(), v) for k, v in (headers or [])],
        "query_string": query,
    }
    if cookies:
        scope["headers"] = [(b"cookie", cookies.encode())] + scope["headers"]
    return Request(scope)


class TestIdempotency:
    def test_new_task_id_format_unique(self):
        from src.api_pkg.routers.pipeline import _new_task_id

        a = _new_task_id("full-cycle")
        b = _new_task_id("full-cycle")
        assert a.startswith("full-cycle_")
        assert b.startswith("full-cycle_")
        assert a != b

    def test_key_from_header_case_insensitive(self):
        from src.api_pkg.routers.pipeline import _idempotency_key_from_request

        assert _idempotency_key_from_request(_req([(b"Idempotency-Key", b"k1")])) == "k1"
        assert _idempotency_key_from_request(_req([(b"idempotency-key", b"k2")])) == "k2"
        assert _idempotency_key_from_request(_req()) == ""

    async def test_register_duplicate_returns_existing(self):
        import src.api_pkg.routers.pipeline as pl

        pl._idempotency_map["dup-key"] = "existing_1"
        pl.pipeline_tasks["existing_1"] = object()
        try:
            tid, dup = await pl._register_task("full-cycle", "dup-key")
            assert (tid, dup) == ("existing_1", True)
        finally:
            pl._idempotency_map.pop("dup-key", None)
            pl.pipeline_tasks.pop("existing_1", None)

    async def test_register_new_persists(self):
        import src.api_pkg.routers.pipeline as pl

        with patch.object(pl, "_persist_idempotency_sync", return_value=None):
            tid, dup = await pl._register_task("rebuild", "fresh-key-xyz")
            assert dup is False
            assert tid.startswith("rebuild_")
            assert pl._idempotency_map.get("fresh-key-xyz") == tid
            pl._idempotency_map.pop("fresh-key-xyz", None)


class TestAuthTokenPriority:
    def test_bearer_wins(self):
        from src.api_pkg.routers.auth import _request_token

        r = _req([(b"authorization", b"Bearer BEAR")], query=b"token=Q", cookies="token=C")
        assert _request_token(r) == "BEAR"

    def test_cookie_before_query(self):
        from src.api_pkg.routers.auth import _request_token

        r = _req(query=b"token=Q", cookies="token=C")
        assert _request_token(r) == "C"

    def test_query_still_works_soft(self):
        from src.api_pkg.routers.auth import _request_token

        assert _request_token(_req(query=b"token=Q")) == "Q"
        assert _request_token(_req()) == ""


class TestGeo:
    async def test_digits_passthrough(self):
        from src.api_pkg.routers.pipeline import _resolve_area_ids

        assert await _resolve_area_ids("113, 2") == "113,2"

    async def test_unknown_city_400(self):
        import src.api_pkg.routers.pipeline as pl

        with patch.object(pl, "_resolve_city_name_sync", return_value=None):
            with pytest.raises(HTTPException) as e:
                await pl._resolve_area_ids("НесуществующийГород")
            assert e.value.status_code == 400

    async def test_empty_400(self):
        from src.api_pkg.routers.pipeline import _resolve_area_ids

        with pytest.raises(HTTPException) as e:
            await _resolve_area_ids("   ")
        assert e.value.status_code == 400

    async def test_resolver_retry_then_503(self):
        import src.api_pkg.routers.pipeline as pl

        pl._areas_cache = None
        import requests

        with patch.object(requests, "get", side_effect=Exception("down")):
            with pytest.raises(HTTPException) as e:
                await pl._resolve_area_ids("Москва")
            assert e.value.status_code == 503
        pl._areas_cache = None


class TestRunAsyncSafely:
    def test_returns_value(self):
        from src.pipeline.runner import _run_async_safely

        async def coro():
            return 42

        assert _run_async_safely(coro, context="t") == 42

    def test_exception_skipped(self):
        from src.pipeline.runner import _run_async_safely

        async def boom():
            raise RuntimeError("x")

        assert _run_async_safely(boom, context="t") is None

    def test_pipeline_retries_from_config(self):
        from src import config

        assert config.PIPELINE_RETRIES == 2


class TestSkillForecastHonesty:
    def _engine(self):
        from src.predictors.skill_forecast import SkillForecastEngine

        e = SkillForecastEngine()
        e._models = {
            "flat": {"slope": 0.0, "intercept": 80, "n": 1, "rmse": None, "mape": None, "last_freq": 80},
            "grow": {"slope": 1.0, "intercept": 10, "n": 10, "rmse": 1.0, "mape": 0.05, "last_freq": 100},
        }
        e._is_fitted = True
        return e

    def test_flat_mape_none_not_zero(self):
        fr = self._engine().predict("flat").ok()
        assert fr.mape is None
        assert fr.forecast_months == 0
        assert fr.engine_used == "trend_flat"

    def test_tops_valid_first(self):
        top = self._engine().top_growing(2).ok()
        assert [r.skill for r in top] == ["grow", "flat"]


class TestTrendCap:
    def test_raw_and_flag(self):
        from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer

        a = SnapshotTrendAnalyzer([
            {"skill_freq": {"python": 10}},
            {"skill_freq": {"python": 100}},
        ])
        row = a.get_rising(10).ok()[0]
        assert row["change_pct"] == 200
        assert row["change_pct_raw"] == 900.0
        assert row["capped"] is True

    def test_small_change_not_capped(self):
        from src.analyzers.trend_analyzer import SnapshotTrendAnalyzer

        a = SnapshotTrendAnalyzer([
            {"skill_freq": {"sql": 100}},
            {"skill_freq": {"sql": 110}},
        ])
        row = a.get_rising(10).ok()[0]
        assert row["capped"] is False
        assert row["change_pct"] == row["change_pct_raw"]


class TestCorrelationCaps:
    def test_matrix_too_big_err(self):
        from src.analyzers.skills.skill_correlation import SkillCorrelationAnalyzer

        r = SkillCorrelationAnalyzer().get_correlation_matrix([f"s{i}" for i in range(201)])
        assert r.is_err()

    def test_fit_truncates_huge_vacancy(self):
        from src.analyzers.skills.skill_correlation import SkillCorrelationAnalyzer

        a = SkillCorrelationAnalyzer()
        with patch(
            "src.analyzers.skills.skill_correlation.SkillNormalizer.normalize",
            side_effect=lambda s: __import__("src").Ok(s),
        ):
            a.fit([["skill-%d" % i for i in range(300)]])
        # Частоты полные (линейно), а квадратичная часть (пары) срезана до C(80,2)
        assert len(a._cooccurrence) <= 80 * 79 // 2


class TestDomainEmpty:
    def test_empty_map_err(self, tmp_path):
        from src.analyzers.comparison.domain_analyzer import DomainAnalyzer

        da = DomainAnalyzer(domain_map_path=tmp_path / "nope.json")
        assert da.compute_domain_coverage(["python"]).is_err()


class TestHybridPrefilter:
    def test_fuzzy_still_finds_close(self):
        from src.analyzers.hybrid_matcher import HybridMatcher

        m = HybridMatcher(market_skills={"python": {}, "javascript": {}, "kubernetes": {}})
        name, method, _ = m.match("pythn").ok()
        assert name == "python"
        assert method == "hybrid_fuzzy"

    def test_far_query_no_match_fast(self):
        from src.analyzers.hybrid_matcher import HybridMatcher

        m = HybridMatcher(market_skills={"python": {}})
        assert m.match("zzzzzzzzzzqqqqqq").ok()[0] is None


def test_migration_compiles():
    root = Path(__file__).parent.parent
    py_compile.compile(
        str(root / "alembic" / "versions" / "pipeline_idempotency_key.py"),
        doraise=True,
    )


class TestTopUsableFilter:
    def _r(self, months, freq):
        from src.predictors.skill_forecast import ForecastResult
        return ForecastResult(
            skill="s", current_frequency=freq, predicted_growth=0.1,
            confidence=0.5, next_year_frequency=freq + 1,
            engine_used="trend", data_points=4, mape=0.1, forecast_months=months,
        )

    def test_no_horizon_excluded(self):
        from src.api_pkg.routers.forecast import _is_usable_forecast
        assert _is_usable_forecast(self._r(0, 999)) is False
        assert _is_usable_forecast(self._r(3, 999)) is True

    def test_min_freq_gate(self):
        from src.api_pkg.routers.forecast import _is_usable_forecast
        assert _is_usable_forecast(self._r(3, 49), min_freq=50) is False
        assert _is_usable_forecast(self._r(3, 50), min_freq=50) is True
        assert _is_usable_forecast(self._r(3, 5), min_freq=0) is True

    def test_top_defaults_keep_junk_out(self):
        import inspect
        from src.api_pkg.routers.forecast import get_top_forecasts
        params = inspect.signature(get_top_forecasts).parameters

        def _default(name):
            d = params[name].default
            return getattr(d, "default", d)

        assert _default("sort") == "growth"
        assert _default("min_freq") == 50


class TestObservedDirection:
    def _items(self):
        return [
            {"skill": "a", "observed_change_pct": -5.0},
            {"skill": "b", "observed_change_pct": 0.0},
            {"skill": "c", "observed_change_pct": 7.0},
        ]

    def test_declining_only_drops(self):
        from src.api_pkg.routers.forecast import _select_observed_direction
        got = _select_observed_direction(self._items(), "declining")
        assert [r["skill"] for r in got] == ["a"]

    def test_growing_only_gains(self):
        from src.api_pkg.routers.forecast import _select_observed_direction
        got = _select_observed_direction(self._items(), "growing")
        assert [r["skill"] for r in got] == ["c"]

    def test_all_keeps_everything(self):
        from src.api_pkg.routers.forecast import _select_observed_direction
        assert len(_select_observed_direction(self._items(), "all")) == 3


class TestEngineInfo:
    def test_prophet_engine_reported(self):
        from src.api_pkg.routers.forecast import _engine_info
        from src.predictors.prophet_forecast import ProphetForecastEngine

        eng = ProphetForecastEngine()
        eng._models = {"a": object(), "b": object()}
        info = _engine_info(eng)
        assert info["engine"] == "prophet"
        assert info["prophet_models"] == 2

    def test_genetic_engine_reported(self):
        from src.api_pkg.routers.forecast import _engine_info
        from src.predictors.skill_forecast import SkillForecastEngine

        info = _engine_info(SkillForecastEngine())
        assert info["engine"] == "genetic"
        assert info["prophet_models"] == 0
        assert info["prophet_fitted"] is False


class TestForecastOffLoop:
    def _r(self, skill, growth, months, freq):
        from src.predictors.skill_forecast import ForecastResult
        return ForecastResult(
            skill=skill, current_frequency=freq, predicted_growth=growth,
            confidence=0.5, next_year_frequency=freq + 1,
            engine_used="trend", data_points=4, mape=0.1, forecast_months=months,
        )

    def test_compute_top_branches(self):
        from src.api_pkg.routers.forecast import _compute_top
        from src.predictors.skill_forecast import SkillForecastEngine

        gen = SkillForecastEngine()
        gen._models = {
            "x": {"slope": 0.1, "intercept": 10, "n": 5, "last_freq": 100.0, "mape": 0.1},
            "y": {"slope": -0.1, "intercept": 200, "n": 5, "last_freq": 200.0, "mape": 0.1},
        }
        gen._is_fitted = True
        res, method = _compute_top(gen, "growing", 3, 10, "growth")
        assert method == "genetic" and res.is_ok()
        res, _ = _compute_top(gen, "declining", 3, 10, "growth")
        grows = [r.predicted_growth for r in res.ok()]
        assert grows == sorted(grows)

    def test_top_cache_hit_and_stale(self):
        import time
        from src.api_pkg.routers import forecast as fc

        key = ("test", 1)
        assert fc._top_cache_get(key) is None
        fc._top_cache_set(key, ["cached"])
        assert fc._top_cache_get(key) == ["cached"]
        # протухшая запись — miss
        fc._TOP_CACHE[key] = (time.monotonic() - fc._TOP_CACHE_TTL - 1, ["old"])
        assert fc._top_cache_get(key) is None
        fc._TOP_CACHE.pop(key, None)


class TestNoLeakDetails:
    """Gate: пользовательские ответы не тащат внутренности (detail=str(e))."""

    def test_no_str_e_details_in_routers(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).parent.parent / "src" / "api_pkg" / "routers"
        bad = []
        for p in sorted(root.glob("*.py")):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"detail\s*=\s*str\(e", line):
                    bad.append(f"{p.name}:{i}: {line.strip()[:100]}")
                m = re.search(r'detail\s*=\s*f["\'](.*)["\']', line)
                if m and re.search(r"\{e(?:xc|rr)?[^}]*\}", m.group(1)):
                    bad.append(f"{p.name}:{i}: {line.strip()[:100]}")
        assert not bad, "leaked internals in user responses:\n" + "\n".join(bad)


class TestUserErrorDetail:
    def test_admin_sees_technical(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.api_pkg.routers.auth import user_error_detail

        req = object()
        with patch(
            "src.api_pkg.routers.auth.get_current_user",
            new=AsyncMock(return_value={"u": "a@x", "r": "admin"}),
        ):
            assert asyncio.run(user_error_detail(req, "TECHNICAL", "safe")) == "TECHNICAL"

    def test_non_admin_gets_safe(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.api_pkg.routers.auth import user_error_detail

        req = object()
        for role in ({"u": "t@x", "r": "teacher"}, {"u": "s@x", "r": "student"}, None):
            with patch(
                "src.api_pkg.routers.auth.get_current_user",
                new=AsyncMock(return_value=role),
            ):
                assert asyncio.run(user_error_detail(req, "TECHNICAL", "safe")) == "safe"

    def test_no_request_or_broken_auth_is_safe(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.api_pkg.routers.auth import user_error_detail

        assert asyncio.run(user_error_detail(None, "TECHNICAL", "safe")) == "safe"
        with patch(
            "src.api_pkg.routers.auth.get_current_user",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            assert asyncio.run(user_error_detail(object(), "TECHNICAL", "safe")) == "safe"

    def test_admin_empty_technical_falls_back(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.api_pkg.routers.auth import user_error_detail

        with patch(
            "src.api_pkg.routers.auth.get_current_user",
            new=AsyncMock(return_value={"u": "a@x", "r": "admin"}),
        ):
            assert asyncio.run(user_error_detail(object(), "", "safe")) == "safe"


class TestPipelineDataFixes:
    """Регрессия: gap-analysis падал с пустыми hybrid/level на кэш-файлах без description."""

    def test_vacancy_has_extracted_skills_field(self):
        from src.models.vacancy import Vacancy

        assert "extracted_skills" in Vacancy.__dataclass_fields__
        assert Vacancy.__dataclass_fields__["extracted_skills"].default_factory() == []

    def test_bm25_text_from_snippet_on_object(self):
        from src.models.vacancy import Vacancy
        from src.parsing.skills.bm25_ranker import BM25Ranker

        vac = Vacancy.from_api({
            "id": "1",
            "name": "t",
            "area": {"id": "1", "name": "a"},
            "employer": {"id": "1", "name": "e"},
            "snippet": {"requirement": "python sql", "responsibility": "data"},
        })
        assert vac.description in (None, "")
        text = BM25Ranker.__new__(BM25Ranker)._extract_vacancy_text(vac)
        assert "python" in text and "data" in text

    def test_trend_cap_label(self):
        # Кап 0.3 должен подаваться как «+30% и более», а не ровные +30%.
        import re

        src = open("src/predictors/recommendation_engine.py", encoding="utf-8").read()
        assert "+30% и более" in src
        assert re.search(r"Топ роста \(\+[^)]*%\)", src) is None or "+30% и более" in src

    def test_automation_vocab_end_to_end(self):
        # Вакансия «Финансовый Навигатор»: automation-лексика обязана извлекаться.
        from src.models.vacancy import Vacancy
        from src.parsing.skills.vacancy_parser import VacancyParser

        text = (
            "опыт работы с AI-инструментами; умение создавать и настраивать чат-ботов; "
            "понимание CRM; работа с Make, n8n; понимание API и Webhooks; "
            "создание дашбордов"
        )
        vac = Vacancy.from_api({
            "id": "t-nav", "name": "IT-специалист",
            "area": {"id": "1", "name": "Астана"},
            "employer": {"id": "1", "name": "Финансовый Навигатор"},
            "description": text,
        })
        res = VacancyParser().extract_skills_from_vacancies([vac])
        freqs = res.ok().get("frequencies", {}) if res.is_ok() else {}
        for must in ("chatbot", "crm", "n8n", "api", "webhook", "dashboard"):
            assert must in freqs, f"lost skill: {must} in {sorted(freqs)}"

    def test_latinized_cyrillic_restores(self):        # Латинизация из parse_vacancy-дедупа не должна убивать кириллические навыки.
        from src.parsing.skills.skill_normalizer import SkillNormalizer

        assert SkillNormalizer.normalize("чaт-бoтoв").ok() == "chatbot"
        assert SkillNormalizer.normalize("дaшбopдoв").ok() == "dashboard"
        # Честная латиница не страдает.
        assert SkillNormalizer.normalize("api").ok() == "api"
        assert SkillNormalizer.normalize("crm").ok() == "crm"

    def test_no_double_api_prefix_in_routers(self):
        # Роутеры монтируются с prefix="/api" — пути вида "/api/..." дают /api/api/*.
        # Исключение: health_router монтируется без префикса.
        import pathlib
        import re

        root = pathlib.Path(__file__).parent.parent / "src" / "api_pkg" / "routers"
        bad = []
        for p in sorted(root.glob("*.py")):
            if p.name == "health.py":
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r'@router\.(get|post|put|delete|patch)\(\s*["\']\/api\/', line):
                    bad.append(f"{p.name}:{i}: {line.strip()[:100]}")
                if re.search(r'^router\s*=\s*APIRouter\(.*prefix\s*=\s*["\']\/api', line):
                    bad.append(f"{p.name}:{i}: router-level /api prefix: {line.strip()[:100]}")
        assert not bad, "double /api prefix (unreachable routes):\n" + "\n".join(bad)


class TestEmergingStrippedCore:
    """Регрессия: c++/c#/.net в emerging, хотя РПД их преподаёт
    (NORMALIZE_RE съедает ++/#/. при нормализации норм РПД)."""

    def test_cpp_not_emerging_when_rpd_teaches_it(self):
        from src.analyzers.skill_matcher import SkillMatcher, normalize

        m = SkillMatcher.__new__(SkillMatcher)
        m.market_skills = {"c++": 1039, "python": 2000, "cobol": 50}
        rpd = {
            normalize(s)
            for s in [
                "Develop efficient multithreaded solutions in C++.",
                "разрабатывать программы на C++, использующие GPU",
                "основ синтаксиса языка C/C++",
            ]
        }
        res = m.get_emerging(rpd, top_n=10)
        assert res.is_ok()
        skills = [s for s, _, _ in res.ok()]
        assert "c++" not in skills, f"c++ leaked to emerging: {skills}"
        assert "cobol" in skills

    def test_dotnet_core_fallback(self):
        from src.analyzers.skill_matcher import SkillMatcher, normalize

        m = SkillMatcher.__new__(SkillMatcher)
        m.market_skills = {".net": 500, "cobol": 50}
        rpd = {normalize("разработка приложений на платформе .NET")}
        res = m.get_emerging(rpd, top_n=10)
        assert res.is_ok()
        assert ".net" not in [s for s, _, _ in res.ok()]
