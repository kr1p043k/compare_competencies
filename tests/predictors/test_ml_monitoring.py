"""Tests for the ML monitoring scripts (manifests, drift).

Pure unit tests: no data/models, no DB, no embedding model. The manifest
validator and the drift statistics are exercised on synthetic artifacts/arrays
so a regression in the checkers is caught in CI on a fresh clone.

Контракт scripts.check_model_manifests (HEAD, API из f6fdbebf/11e01c5b - его же
зовут Makefile `ml-manifests` и retrain_ranker.py `--write`):
    pick_ndcg(mapping)            - приоритет human_ndcg_at_10 > ndcg@10 > ndcg_at_10
    collect_sidecar(path)         - (sidecar, feature_names, ndcg) из .manifest/.metrics
    artifact_feature_names(path)  - список имён внутри .joblib
    frozen_ndcg(report, name)     - NDCG артефакта в frozen-отчёте
    validate(models_dir, report)  - список проблем (пустой = всё ок)
    write_sidecars(models_dir, report)
Ранее тест ссылался на приватный API редизайна 7b5512f2 (_manifest_ndcg,
5-аргументный validate), удалённый мержем 11e01c5b, - приведён к реальному.
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from scripts import check_feature_drift
from scripts import check_model_manifests as cmm

FEATURES = ["a", "b", "c"]
ARTIFACT = "ltr_ranker_demo.joblib"


def _artifact(models_dir, name=ARTIFACT, features=FEATURES):
    joblib.dump({"feature_names": features}, models_dir / name)
    return name


# ---------- pick_ndcg: приоритет ключей ----------


def test_pick_ndcg_precedence():
    assert cmm.pick_ndcg({"human_ndcg_at_10": 0.8, "ndcg@10": 0.7}) == 0.8
    assert cmm.pick_ndcg({"ndcg@10": 0.7, "ndcg_at_10": 0.6}) == 0.7


def test_pick_ndcg_none_cases():
    assert cmm.pick_ndcg({"other": 1.0}) is None
    assert cmm.pick_ndcg({"ndcg@10": "bad"}) is None
    assert cmm.pick_ndcg({"ndcg@10": True}) is None
    assert cmm.pick_ndcg(None) is None


# ---------- artifact_feature_names ----------


def test_artifact_feature_names_reads_joblib(tmp_path):
    name = _artifact(tmp_path)
    assert cmm.artifact_feature_names(tmp_path / name) == FEATURES


def test_artifact_feature_names_missing_file(tmp_path):
    assert cmm.artifact_feature_names(tmp_path / "nope.joblib") is None


# ---------- collect_sidecar / frozen_ndcg ----------


def test_collect_sidecar_reads_manifest_with_nested_metrics(tmp_path):
    path = tmp_path / _artifact(tmp_path)
    cmm.manifest_path(path).write_text(
        json.dumps({"feature_names": FEATURES, "metrics": {"human_ndcg_at_10": 0.7}}),
        encoding="utf-8",
    )
    _data, names, ndcg = cmm.collect_sidecar(path)
    assert names == FEATURES and ndcg == 0.7


def test_collect_sidecar_falls_back_to_metrics_json(tmp_path):
    path = tmp_path / _artifact(tmp_path)
    cmm.metrics_path(path).write_text(
        json.dumps({"feature_names": FEATURES, "ndcg@10": 0.6}),
        encoding="utf-8",
    )
    _data, names, ndcg = cmm.collect_sidecar(path)
    assert names == FEATURES and ndcg == 0.6


def test_frozen_ndcg_reads_results_by_artifact_name():
    report = {"results": {ARTIFACT: {"ndcg@10": 0.7195}}}
    assert cmm.frozen_ndcg(report, ARTIFACT) == 0.7195
    assert cmm.frozen_ndcg(report, "absent.joblib") is None


# ---------- validate / write_sidecars ----------


def test_validate_ok_for_sidecarred_artifact(tmp_path):
    path = tmp_path / _artifact(tmp_path)
    cmm.metrics_path(path).write_text(
        json.dumps({"feature_names": FEATURES, "human_ndcg_at_10": 0.7}),
        encoding="utf-8",
    )
    report = {"results": {ARTIFACT: {"ndcg@10": 0.7}}}
    assert cmm.validate(tmp_path, report) == []


def test_validate_flags_missing_sidecar(tmp_path):
    _artifact(tmp_path)
    problems = cmm.validate(tmp_path, {"results": {}})
    assert any("sidecar" in p for p in problems)


def test_validate_empty_models_dir(tmp_path):
    assert cmm.validate(tmp_path / "missing", None) == []


def test_write_sidecars_creates_manifest_and_metrics(tmp_path):
    name = _artifact(tmp_path)
    assert cmm.write_sidecars(tmp_path, {"results": {name: {"ndcg@10": 0.7455}}}) == 1
    manifest = json.loads(cmm.manifest_path(tmp_path / name).read_text(encoding="utf-8"))
    assert manifest["metrics"]["ndcg@10"] == 0.7455
    metrics = json.loads(cmm.metrics_path(tmp_path / name).read_text(encoding="utf-8"))
    assert metrics["human_ndcg_at_10"] == 0.7455


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
