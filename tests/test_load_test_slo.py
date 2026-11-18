# tests/test_load_test_slo.py
"""SLO-гейты load-теста: синхронизация CI-workflow и k6-скрипта.

Статическая валидация без запуска k6 и сети:
- .github/workflows/load-test.yml объявляет env SLO_* и шаг
  "Check thresholds", который проверяет все три метрики через эти пороги;
- thresholds в tests/load/k6_script.js совпадают с SLO-константами workflow,
  иначе первичный гейт (k6, exit 99) и вторичный (bash-шаг) рассинхронизуются.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "load-test.yml"
K6_SCRIPT_PATH = REPO_ROOT / "tests" / "load" / "k6_script.js"

# thresholds: { http_req_failed: ['rate<0.05'], http_req_duration: ['p(95)<500'] }
K6_ERROR_RATE_RE = re.compile(
    r"http_req_failed:\s*\[\s*['\"]rate\s*<\s*([0-9]+(?:\.[0-9]+)?)['\"]"
)
K6_P95_RE = re.compile(
    r"http_req_duration:\s*\[\s*['\"]p\(95\)\s*<\s*([0-9]+(?:\.[0-9]+)?)['\"]"
)

SLO_KEYS = ("SLO_ERROR_RATE", "SLO_P95_MS", "SLO_CHECKS_RATE")


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW_PATH.is_file(), f"workflow file not found: {WORKFLOW_PATH}"
    data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "load-test.yml: пустой или невалидный YAML"
    return data


@pytest.fixture(scope="module")
def slo_env(workflow: dict) -> dict[str, float]:
    env = workflow.get("env")
    assert isinstance(env, dict), "load-test.yml: нет workflow-level env-блока с SLO_*"
    for key in SLO_KEYS:
        assert key in env, f"load-test.yml: нет env.{key}"
        assert isinstance(env[key], str), f"load-test.yml: env.{key} должен быть строкой, got {env[key]!r}"
    return {key: float(env[key]) for key in SLO_KEYS}


@pytest.fixture(scope="module")
def check_step_run(workflow: dict) -> str:
    steps = workflow["jobs"]["load-test"]["steps"]
    run: str | None = None
    for step in steps:
        if step.get("name") == "Check thresholds":
            run = step.get("run", "")
            break
    assert run, 'load-test.yml: нет шага "Check thresholds" с run-блоком'
    return run


@pytest.fixture(scope="module")
def k6_script_text() -> str:
    assert K6_SCRIPT_PATH.is_file(), f"k6 script not found: {K6_SCRIPT_PATH}"
    return K6_SCRIPT_PATH.read_text(encoding="utf-8")


def _k6_thresholds(k6_text: str) -> tuple[float, float]:
    m_err = K6_ERROR_RATE_RE.search(k6_text)
    assert m_err, "k6_script.js: не найден threshold http_req_failed вида ['rate<...']"
    m_p95 = K6_P95_RE.search(k6_text)
    assert m_p95, "k6_script.js: не найден threshold http_req_duration вида ['p(95)<...']"
    return float(m_err.group(1)), float(m_p95.group(1))


def test_workflow_declares_slo_env_constants(slo_env: dict[str, float]) -> None:
    assert set(slo_env) == set(SLO_KEYS)
    assert 0.0 < slo_env["SLO_ERROR_RATE"] < 1.0, "SLO_ERROR_RATE должен быть долей (0..1)"
    assert slo_env["SLO_P95_MS"] > 0, "SLO_P95_MS должен быть положительным числом мс"
    assert 0.0 < slo_env["SLO_CHECKS_RATE"] <= 1.0, "SLO_CHECKS_RATE должен быть долей (0..1]"


def test_check_thresholds_step_covers_all_slo_metrics(check_step_run: str) -> None:
    for token in ("http_req_failed", "http_req_duration", "p(95)", "exit 1"):
        assert token in check_step_run, (
            f'шаг "Check thresholds" не упоминает {token!r} (fail-closed гейт неполный)'
        )


def test_check_thresholds_step_uses_slo_env(check_step_run: str) -> None:
    assert re.search(r">\s*\$SLO_ERROR_RATE", check_step_run), (
        "сравнение error rate должно использовать $SLO_ERROR_RATE, а не захардкоженное число"
    )
    assert re.search(r">\s*\$SLO_P95_MS", check_step_run), (
        "сравнение p95 должно использовать $SLO_P95_MS, а не захардкоженное число"
    )
    assert re.search(r"<\s*\$SLO_CHECKS_RATE", check_step_run), (
        "сравнение checks должно использовать $SLO_CHECKS_RATE, а не захардкоженное число"
    )


def test_k6_script_declares_thresholds(k6_script_text: str) -> None:
    err, p95 = _k6_thresholds(k6_script_text)
    assert 0.0 < err < 1.0, f"неверный k6-порог http_req_failed rate<{err}"
    assert p95 > 0, f"неверный k6-порог http_req_duration p(95)<{p95}"


def test_k6_thresholds_match_workflow_slo(slo_env: dict[str, float], k6_script_text: str) -> None:
    err, p95 = _k6_thresholds(k6_script_text)
    assert err == slo_env["SLO_ERROR_RATE"], (
        f"k6 http_req_failed rate<{err}, а workflow SLO_ERROR_RATE={slo_env['SLO_ERROR_RATE']} - "
        "первичный и вторичный гейты рассинхронизированы"
    )
    assert p95 == slo_env["SLO_P95_MS"], (
        f"k6 http_req_duration p(95)<{p95}, а workflow SLO_P95_MS={slo_env['SLO_P95_MS']} - "
        "первичный и вторичный гейты рассинхронизированы"
    )
