"""Синхронизация SLO: workflow ↔ k6 ↔ документация (BACKLOG 6.7.4).

Гейт в CI и thresholds в k6-скрипте задают одни и те же пороги в двух местах.
Расхождение (напр. p95 400 в k6 против 800 в workflow) означает, что CI проверяет
не то, что заявлено, поэтому фиксируем инвариант тестом.
"""

from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "load-test.yml"
K6 = ROOT / "tests" / "load" / "k6_script.js"
BASELINE = ROOT / "docs" / "LOAD-BASELINE.md"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8", errors="ignore"))


def _gate_step() -> dict:
    for step in _workflow()["jobs"]["load-test"]["steps"]:
        if step.get("name") == "Check thresholds":
            return step
    raise AssertionError("шаг 'Check thresholds' не найден в load-test.yml")


def test_workflow_declares_slo_constants():
    env = _workflow().get("env") or {}
    assert str(env.get("SLO_P95_MS")) == "800"
    assert str(env.get("SLO_ERROR_RATE")) == "0.01"
    assert str(env.get("SLO_CHECKS_RATE")) == "0.99"


def test_gate_checks_all_three_slo_and_fails_closed():
    run = _gate_step().get("run", "")
    for metric in ("http_req_failed", "http_req_duration", 'p(95)', "checks"):
        assert metric in run, f"гейт не проверяет {metric}"
    assert "exit 1" in run, "гейт должен падать (exit 1), а не только печатать"
    assert "SLO_P95_MS" in run and "SLO_ERROR_RATE" in run and "SLO_CHECKS_RATE" in run


def test_k6_thresholds_match_workflow():
    script = K6.read_text(encoding="utf-8", errors="ignore")
    env = _workflow()["env"]

    err = re.search(r"http_req_failed:\s*\[\s*'rate<([0-9.]+)'", script)
    p95 = re.search(r"http_req_duration:\s*\[\s*'p\(95\)<([0-9.]+)'", script)
    assert err and p95, "в k6-скрипте не найдены thresholds error-rate/p95"

    assert float(err.group(1)) == float(env["SLO_ERROR_RATE"])
    assert float(p95.group(1)) == float(env["SLO_P95_MS"])


def test_baseline_documents_slo():
    text = BASELINE.read_text(encoding="utf-8", errors="ignore")
    assert "SLO" in text, "docs/LOAD-BASELINE.md должен описывать SLO-пороги"
