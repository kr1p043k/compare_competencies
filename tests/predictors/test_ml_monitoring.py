"""Tests for the ML monitoring scripts (manifests, drift).

Pure unit tests: no data/models, no DB, no embedding model. The manifest
validator and the drift statistics are exercised on synthetic artifacts/arrays
so a regression in the checkers is caught in CI on a fresh clone.

Контракт scripts.check_model_manifests (после редизайна 7b5512f2):
    validate(name, feature_names, manifest, runtime_features, incompatible)
    _manifest_ndcg(manifest)  - приоритет human_ndcg_at_10 > ndcg@10 > ndcg_at_10
    _artifact_feature_names(path)
Раньше тест ссылался на удалённый API (pick_ndcg/write_sidecars/validate(dir, об.
report)) - таких функций в модуле нет, тест приведён к реальному контракту.
"""

from __future__ import annotations

import joblib
import numpy as np

from scripts import check_feature_drift
from scripts import check_model_manifests as cmm

FEATURES = ["a", "b", "c"]
RUNTIME = ["a", "b", "c"]
MANIFEST_OK = {"metrics": {"human_ndcg_at_10": 0.7}}


def _artifact(models_dir, name="ltr_ranker_demo.joblib", features=FEATURES):
    joblib.dump({"feature_names": features}, models_dir / name)
    return name


# ---------- _manifest_ndcg: приоритет ключей ----------


def test_manifest_ndcg_precedence():
    assert cmm._manifest_ndcg({"metrics": {"human_ndcg_at_10": 0.8, "ndcg@10": 0.7}}) == 0.8
    assert cmm._manifest_ndcg({"metrics": {"ndcg@10": 0.7, "ndcg_at_10": 0.6}}) == 0.7
    assert cmm._manifest_ndcg({"metrics": {"ndcg_at_10": 0.6}}) == 0.6


def test_manifest_ndcg_none_cases():
    assert cmm._manifest_ndcg({"metrics": {"other": 1.0}}) is None
    assert cmm._manifest_ndcg({"metrics": {"ndcg@10": "bad"}}) is None
    assert cmm._manifest_ndcg(None) is None
    assert cmm._manifest_ndcg({"metrics": None}) is None


# ---------- validate: реальный контракт ----------


def test_validate_passes_when_consistent():
    assert cmm.validate("m", ["a", "b", "c"], MANIFEST_OK, RUNTIME, {}) == []


def test_validate_flags_incompatible_model():
    problems = cmm.validate("m", ["a", "b", "c"], MANIFEST_OK, RUNTIME, {"m": "старая"})
    assert problems and "несовместим" in problems[0]


def test_validate_flags_missing_feature_names():
    problems = cmm.validate("m", None, MANIFEST_OK, RUNTIME, {})
    assert any("feature_names" in p for p in problems)


def test_validate_flags_feature_mismatch():
    problems = cmm.validate("m", ["a", "b"], MANIFEST_OK, RUNTIME, {})
    assert any("feature_names" in p for p in problems)


def test_validate_flags_missing_ndcg():
    problems = cmm.validate("m", ["a", "b", "c"], {"metrics": {}}, RUNTIME, {})
    assert any("NDCG" in p for p in problems)


# ---------- _artifact_feature_names ----------


def test_artifact_feature_names_reads_joblib(tmp_path):
    name = _artifact(tmp_path)
    assert cmm._artifact_feature_names(tmp_path / name) == tuple(FEATURES)


def test_artifact_feature_names_missing_file(tmp_path):
    assert cmm._artifact_feature_names(tmp_path / "nope.joblib") is None


# ---------- feature drift (не менялось) ----------


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
