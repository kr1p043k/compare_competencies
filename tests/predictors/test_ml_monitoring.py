"""Tests for the ML monitoring scripts (manifests, drift).

Pure unit tests: no data/models, no DB, no embedding model. The manifest
validator and the drift statistics are exercised on synthetic artifacts/arrays
so a regression in the checkers is caught in CI on a fresh clone.
"""

from __future__ import annotations

import joblib
import numpy as np

from scripts import check_feature_drift
from scripts import check_model_manifests as cmm

FEATURES = ["a", "b", "c"]


def _artifact(models_dir, name="ltr_ranker_demo.joblib", features=FEATURES):
    joblib.dump({"feature_names": features}, models_dir / name)
    return name


def _report(name, ndcg=0.7):
    return {"results": {name: {"ndcg@10": ndcg}}}


def test_pick_ndcg_precedence():
    assert cmm.pick_ndcg({"human_ndcg_at_10": 0.8, "ndcg@10": 0.7}) == 0.8
    assert cmm.pick_ndcg({"ndcg@10": 0.7, "ndcg_at_10": 0.6}) == 0.7
    assert cmm.pick_ndcg({"ndcg_at_10": 0.6}) == 0.6
    assert cmm.pick_ndcg({"other": 1.0}) is None
    assert cmm.pick_ndcg(None) is None


def test_validate_passes_with_manifest(tmp_path):
    name = _artifact(tmp_path)
    (tmp_path / "ltr_ranker_demo.manifest.json").write_text(
        '{"feature_names": ["a", "b", "c"], "metrics": {"human_ndcg_at_10": 0.7}}',
        encoding="utf-8",
    )
    assert cmm.validate(tmp_path, _report(name)) == []


def test_validate_flags_missing_sidecar(tmp_path):
    name = _artifact(tmp_path)
    problems = cmm.validate(tmp_path, _report(name))
    assert any("no .manifest.json" in p for p in problems)


def test_validate_flags_feature_mismatch(tmp_path):
    name = _artifact(tmp_path)
    (tmp_path / "ltr_ranker_demo.manifest.json").write_text(
        '{"feature_names": ["a", "b"], "metrics": {"ndcg@10": 0.7}}',
        encoding="utf-8",
    )
    problems = cmm.validate(tmp_path, _report(name))
    assert any("feature_names" in p for p in problems)


def test_validate_flags_missing_ndcg(tmp_path):
    name = _artifact(tmp_path)
    (tmp_path / "ltr_ranker_demo.joblib.metrics.json").write_text(
        '{"feature_names": ["a", "b", "c"]}',
        encoding="utf-8",
    )
    problems = cmm.validate(tmp_path, _report(name))
    assert any("no NDCG@10" in p for p in problems)


def test_write_sidecars_creates_both_files(tmp_path):
    name = _artifact(tmp_path)
    written = cmm.write_sidecars(tmp_path, _report(name, ndcg=0.71))
    assert written == 1
    assert (tmp_path / "ltr_ranker_demo.manifest.json").exists()
    assert (tmp_path / "ltr_ranker_demo.joblib.metrics.json").exists()
    assert cmm.validate(tmp_path, _report(name, ndcg=0.71)) == []


def test_psi_zero_for_identical_distributions():
    values = np.linspace(0, 1, 50)
    assert check_feature_drift.psi(values, values) == 0.0


def test_psi_large_for_shifted_distributions():
    reference = np.linspace(0, 1, 50)
    current = reference + 5.0
    assert check_feature_drift.psi(reference, current) > 0.25


def test_ks_stat_detects_full_separation():
    assert check_feature_drift.ks_stat(np.zeros(50), np.ones(50)) == 1.0


def test_summarize_reports_summary_and_values():
    summary = check_feature_drift.summarize(np.array([1.0, 2.0, 3.0]))
    assert summary["mean"] == 2.0
    assert summary["values"] == [1.0, 2.0, 3.0]
