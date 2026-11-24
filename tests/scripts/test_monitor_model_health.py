"""Unit tests for scripts/monitor_model_health.py (BACKLOG 6.6.2).

Pure unit tests on synthetic data: no joblib artifacts, no DB, no live scoring.
The verdict contract (status/reasons/metrics), the PSI/KS threshold behaviour
and the NDCG-drop comparison are exercised directly through evaluate().
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import monitor_model_health as mmh
from scripts.check_feature_drift import summarize


def _frame_with_baseline(*, shift_feature: str | None = None, shift_amount: float = 5.0):
    """Synthetic serving frame + matching drift baseline (same summarize())."""
    rng = np.random.default_rng(42)
    columns = {
        "level_encoded": rng.normal(2.0, 0.6, 200),
        "cosine_sim": rng.normal(0.5, 0.1, 200),
    }
    baseline_features = {name: summarize(values) for name, values in columns.items()}
    if shift_feature is not None:
        columns[shift_feature] = columns[shift_feature] + shift_amount
    x_test = pd.DataFrame(columns)
    return x_test, {"features": baseline_features}


def test_verdict_ok_when_no_drift_and_ndcg_at_baseline():
    x_test, baseline = _frame_with_baseline()
    verdict = mmh.evaluate(x_test, baseline, baseline_ndcg=0.7455, current_ndcg_value=0.7455)
    assert verdict["status"] == "ok"
    assert verdict["reasons"] == []
    assert verdict["metrics"]["ndcg"]["drop"] == 0.0
    assert verdict["metrics"]["feature_drift"]["max_psi"] == 0.0
    assert verdict["thresholds"] == {
        "ndcg_drop": mmh.DEFAULT_NDCG_DROP_TOLERANCE,
        "ks": mmh.DEFAULT_KS,
        "psi": mmh.DEFAULT_PSI,
        "mean_shift_z": mmh.DEFAULT_Z,
    }


def test_verdict_alert_when_psi_above_threshold():
    x_test, baseline = _frame_with_baseline(shift_feature="cosine_sim")
    verdict = mmh.evaluate(x_test, baseline, baseline_ndcg=0.7455, current_ndcg_value=0.7455)
    assert verdict["status"] == "alert"
    assert any("cosine_sim" in r and "psi" in r for r in verdict["reasons"])
    assert verdict["metrics"]["feature_drift"]["max_psi"] > mmh.DEFAULT_PSI
    assert verdict["metrics"]["ndcg"]["drop"] == 0.0


def test_verdict_alert_when_ndcg_below_baseline():
    x_test, baseline = _frame_with_baseline()
    verdict = mmh.evaluate(x_test, baseline, baseline_ndcg=0.7455, current_ndcg_value=0.60)
    assert verdict["status"] == "alert"
    assert any(r.startswith("ndcg") for r in verdict["reasons"])
    assert verdict["metrics"]["ndcg"]["drop"] == pytest.approx(0.1455)
    assert verdict["metrics"]["feature_drift"]["max_psi"] == 0.0


def test_ndcg_drop_at_tolerance_edge_is_ok():
    reasons, metrics = mmh.check_ndcg_drop(baseline=0.7455, current=0.7455 - 0.05)
    assert reasons == []
    assert metrics["drop"] == pytest.approx(0.05)
    reasons, _ = mmh.check_ndcg_drop(baseline=0.7455, current=0.7455 - 0.051)
    assert reasons and "ndcg" in reasons[0]


def test_ndcg_missing_baseline_or_current_alerts():
    reasons, _ = mmh.check_ndcg_drop(baseline=None, current=0.7)
    assert any("baseline missing" in r for r in reasons)
    reasons, _ = mmh.check_ndcg_drop(baseline=0.7, current=None)
    assert any("unavailable" in r for r in reasons)


def test_missing_feature_row_in_baseline_is_a_reason():
    x_test, baseline = _frame_with_baseline()
    baseline["features"].pop("cosine_sim")
    reasons, _ = mmh.check_feature_drift(x_test, baseline, 0.25, 0.25, 1.0)
    assert reasons == ["cosine_sim: missing baseline"]


def test_resolve_baseline_ndcg_prefers_explicit_file_then_sidecar_then_report(tmp_path):
    artifact = "ltr_ranker_demo.joblib"
    report = {"results": {artifact: {"ndcg@10": 0.50}}}
    # nothing stored anywhere -> None
    value, source = mmh.resolve_baseline_ndcg(artifact, tmp_path / "none.json", tmp_path, report)
    assert value == 0.50 and source == "frozen report"
    # sidecar wins over the report
    (tmp_path / f"{artifact}.metrics.json").write_text(json.dumps({"ndcg@10": 0.60}), encoding="utf-8")
    value, source = mmh.resolve_baseline_ndcg(artifact, tmp_path / "none.json", tmp_path, report)
    assert value == 0.60 and "sidecar" in source
    # explicit baseline file wins over the sidecar
    explicit = tmp_path / "ndcg_baseline.json"
    explicit.write_text(json.dumps({"human_ndcg_at_10": 0.70}), encoding="utf-8")
    value, source = mmh.resolve_baseline_ndcg(artifact, explicit, tmp_path, report)
    assert value == 0.70 and source == str(explicit)


def test_current_ndcg_from_frozen_eval_style_report(tmp_path):
    path = tmp_path / "frozen_eval.json"
    path.write_text(
        json.dumps({"results": {"m.joblib": {"ndcg@10": 0.71}}}),
        encoding="utf-8",
    )
    assert mmh.current_ndcg_from_report(path, "m.joblib") == 0.71
    assert mmh.current_ndcg_from_report(path, "absent.joblib") is None
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"human_ndcg_at_10": 0.42}), encoding="utf-8")
    assert mmh.current_ndcg_from_report(flat, "whatever") == 0.42


def test_post_verdict_payload_and_errors(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["ctype"] = request.headers["Content-type"]
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(mmh.urllib.request, "urlopen", fake_urlopen)
    verdict = {"status": "alert", "reasons": ["ndcg drop"]}
    assert mmh.post_verdict(verdict, "https://n8n.invalid/webhook/model-health") == "http 200"
    assert captured["body"] == verdict
    assert captured["ctype"] == "application/json"

    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(mmh.urllib.request, "urlopen", boom)
    with pytest.raises(OSError, match="connection refused"):
        mmh.post_verdict(verdict, "https://n8n.invalid/webhook/model-health")


def test_emit_prints_and_writes_file(tmp_path, capsys):
    out = tmp_path / "reports" / "verdict.json"
    verdict = {"status": "ok", "reasons": []}
    mmh._emit(verdict, str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == verdict
    assert json.loads(capsys.readouterr().out) == verdict
