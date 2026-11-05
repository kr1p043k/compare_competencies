"""Coverage tests for src/api_pkg/routers/forecast.py.

No subprocess / DB / model downloads; monkeypatched forecast engine and data loader.
"""

from fastapi.testclient import TestClient

import src.api_pkg.routers.auth as auth_mod
import src.api_pkg.routers.forecast as forecast_mod
from src import Ok
from src.api_pkg import create_app


class _FakeEngine:
    name = "trend"
    is_fitted = True
    _models = {}

    @staticmethod
    def forecast_all(months):
        return Ok([forecast_mod.ForecastResult(
            skill="python",
            current_frequency=100.0,
            predicted_growth=0.25,
            confidence=0.7,
            next_year_frequency=125.0,
            engine_used="trend",
            data_points=5,
            mape=0.1,
            forecast_months=months,
        )])

    @staticmethod
    def forecast(skill, months=12):
        return Ok(forecast_mod.ForecastResult(
            skill=skill,
            current_frequency=100.0,
            predicted_growth=0.25,
            confidence=0.7,
            next_year_frequency=125.0,
            engine_used="trend",
            data_points=5,
            mape=0.1,
            forecast_months=months,
        ))

    def top_growing(self, n=10, months=12):
        return self.forecast_all(months)


class _FakeEngineEmpty:
    name = "trend"
    is_fitted = True

    @staticmethod
    def forecast_all(months):
        return Ok([])

    @staticmethod
    def forecast(skill, months=12):
        return Ok(forecast_mod.ForecastResult(
            skill=skill,
            current_frequency=0.0,
            predicted_growth=0.0,
            confidence=0.0,
            next_year_frequency=0.0,
            engine_used="trend",
            data_points=0,
            mape=None,
            forecast_months=months,
        ))

    def top_growing(self, n=10, months=12):
        return Ok([])


def _make_client(monkeypatch, *, engine=None, freq_data=None):
    app = create_app()

    async def _fake_user(_req):
        return {"u": "t@t.com", "r": "teacher", "uid": "u-1"}

    monkeypatch.setattr(auth_mod, "get_current_user", _fake_user)
    if engine is not None:
        async def _fake_engine():
            return Ok(engine)
        monkeypatch.setattr(forecast_mod, "_get_forecast_engine", _fake_engine)
    if freq_data is not None:
        monkeypatch.setattr(forecast_mod, "_get_forecast_data", lambda: freq_data)
    # keep tests deterministic: clear module caches for each scenario
    forecast_mod._clear_forecast_caches()
    return TestClient(app)


# ----------------------------------------------------------------------------- /forecast/all


def test_get_all_forecasts_success(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/all")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    assert body["forecasts"][0]["skill"] == "python"
    assert body["engine"] == "trend"


def test_get_all_forecasts_cache_hit(monkeypatch):
    engine = _FakeEngine()
    client = _make_client(monkeypatch, engine=engine)
    r1 = client.get("/api/forecast/all")
    assert r1.status_code == 200
    # cache is populated; break the compute path so a recompute would fail.
    monkeypatch.setattr(
        engine, "forecast_all",
        lambda months: (_ for _ in ()).throw(RuntimeError("should not recompute")),
    )
    r2 = client.get("/api/forecast/all")
    assert r2.status_code == 200
    assert r2.json()["forecasts"][0]["skill"] == r1.json()["forecasts"][0]["skill"]


def test_get_all_forecasts_engine_unavailable_returns_503(monkeypatch):
    async def _fail_engine():
        return forecast_mod.Err(forecast_mod.DomainError("no data"))

    monkeypatch.setattr(forecast_mod, "_get_forecast_engine", _fail_engine)
    client = _make_client(monkeypatch)
    r = client.get("/api/forecast/all")
    assert r.status_code == 503, r.text
    assert "Недоступен" in r.json()["detail"] or "недоступен" in r.json()["detail"]


def test_get_all_forecasts_validation_months_out_of_range(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/all?months=0")
    assert r.status_code == 422, r.text


# ----------------------------------------------------------------------------- /forecast/top


def test_get_top_forecasts_success(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/top?n=5&direction=growing&sort=popular")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"] == "growing"
    assert body["n"] == 5
    assert body["forecasts"]


def test_get_top_forecasts_cache_miss_then_hit(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r1 = client.get("/api/forecast/top")
    assert r1.status_code == 200
    monkeypatch.setattr(forecast_mod, "_compute_top", lambda *_: (_ for _ in ()).throw(RuntimeError("no recompute")))
    r2 = client.get("/api/forecast/top")
    assert r2.status_code == 200
    assert r2.json()["forecasts"][0]["skill"] == r1.json()["forecasts"][0]["skill"]


def test_get_top_forecasts_invalid_direction(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/top?direction=sideways")
    assert r.status_code == 422, r.text


# ----------------------------------------------------------------------------- /forecast/engine


def test_get_forecast_engine_status(monkeypatch):
    monkeypatch.setattr(forecast_mod.deps, "prophet_engine", None)
    monkeypatch.setattr(forecast_mod.deps, "skill_engine", _FakeEngine())
    client = _make_client(monkeypatch)
    r = client.get("/api/forecast/engine")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "trend"


# ----------------------------------------------------------------------------- /forecast/observed


def test_get_observed_drops_wrong_months_returns_400(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/observed?months=2")
    assert r.status_code == 400, r.text


# ----------------------------------------------------------------------------- /forecast/{skill}


def test_get_skill_forecast_success(monkeypatch):
    client = _make_client(monkeypatch, engine=_FakeEngine())
    r = client.get("/api/forecast/python?months=6")
    assert r.status_code == 200, r.text
    assert r.json()["skill"] == "python"


def test_get_skill_forecast_not_found(monkeypatch):
    class _MissingEngine(_FakeEngine):
        @staticmethod
        def forecast(skill, months):
            return forecast_mod.Err(forecast_mod.DomainError("missing"))
    client = _make_client(monkeypatch, engine=_MissingEngine())
    r = client.get("/api/forecast/missing-skill")
    assert r.status_code == 404, r.text


# ----------------------------------------------------------------------------- empty-data -> 503 on lazy fit


def test_get_popular_forecasts_empty_data_returns_503(monkeypatch):
    # If _get_forecast_data returns {} and no engine is cached, _get_forecast_engine errors.
    monkeypatch.setattr(forecast_mod, "_get_forecast_data", lambda: {})
    monkeypatch.setattr(forecast_mod.deps, "prophet_engine", None)
    monkeypatch.setattr(forecast_mod.deps, "skill_engine", None)
    client = _make_client(monkeypatch)
    r = client.get("/api/forecast/popular")
    assert r.status_code == 503, r.text
