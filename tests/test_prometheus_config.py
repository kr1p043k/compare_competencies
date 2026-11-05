"""Контракт scrape-конфигурации: заголовок, а не Bearer.

Контрактный тест бэкенда не увидит регресс в самом prometheus.yml, поэтому
конфиг проверяется отдельно: если backend-job снова начнёт слать
`Authorization: Bearer`, который бэкенд не читает, тест падает.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "prometheus" / "prometheus.yml"


def _jobs() -> list[dict]:
    with CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("scrape_configs", [])


def _backend_job() -> dict:
    for job in _jobs():
        if job.get("job_name") == "backend":
            return job
    raise AssertionError("в prometheus.yml нет job_name=backend")


def test_backend_scrape_uses_x_metrics_token_header():
    headers = _backend_job().get("http_headers") or {}
    assert "X-Metrics-Token" in headers, f"backend-job шлёт {list(headers)}"
    files = headers["X-Metrics-Token"].get("files") or []
    assert files and files[0].endswith("metrics_token"), f"нет файла с токеном: {files}"


def test_backend_scrape_does_not_use_bearer_authorization():
    # Ровно та регрессия, из-за которой Prometheus получал 401 и up==0.
    assert "authorization" not in _backend_job()


def test_other_jobs_untouched():
    names = {j.get("job_name") for j in _jobs()}
    for expected in ("backend", "postgres", "node-exporter", "prometheus"):
        assert expected in names, f"пропал job {expected}: {sorted(names)}"
