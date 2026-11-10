"""Гейт по CI: обязательные проверки на месте и advisory помечены как таковые.

Цель - не дать тихо удалить или ослабить обязательный шаг (ruff correctness,
secret scan, typecheck/tests/build, alembic, openapi-drift, ML-регресс) и не
дать advisory-шагам стать блокирующими по ошибке.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CI = REPO / ".github" / "workflows" / "ci.yml"

# job -> обязательные (fail-closed) имена шагов
REQUIRED = {
    "lint": ["Ruff (correctness)", "Secret scan (fail-closed)"],
    "frontend": ["Typecheck", "Unit tests", "Build"],
    "migrations": ["Single alembic head", "Upgrade from scratch"],
    "test": ["OpenAPI drift check", "Tests"],
    "ml-regression": ["Frozen ML regression gate"],
}

# advisory-шаги, у которых ОБЯЗАН стоять continue-on-error: true
ADVISORY = [
    ("lint", "Ruff (full, advisory)"),
    ("lint", "Dependency audit (pip-audit, advisory)"),
    ("frontend", "Dependency audit (npm audit, advisory)"),
    ("test", "Dependency drift (advisory)"),
]


def _jobs() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]


def _step(job: dict, name: str) -> dict | None:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    return None


def test_required_ci_gates_exist() -> None:
    jobs = _jobs()
    missing: list[str] = []
    for job_name, steps in REQUIRED.items():
        job = jobs.get(job_name)
        if job is None:
            missing.append(f"job {job_name} исчез")
            continue
        for step_name in steps:
            if _step(job, step_name) is None:
                missing.append(f"{job_name}: шаг '{step_name}' исчез")
    assert not missing, "Обязательные гейты CI пропали:\n" + "\n".join(missing)


def test_required_ci_gates_are_fail_closed() -> None:
    jobs = _jobs()
    softened: list[str] = []
    for job_name, steps in REQUIRED.items():
        job = jobs.get(job_name, {})
        for step_name in steps:
            step = _step(job, step_name)
            if step is not None and step.get("continue-on-error"):
                softened.append(f"{job_name}: '{step_name}' помечен continue-on-error")
    assert not softened, "Обязательные гейты ослаблены:\n" + "\n".join(softened)


def test_advisory_steps_stay_advisory() -> None:
    jobs = _jobs()
    problems: list[str] = []
    for job_name, step_name in ADVISORY:
        step = _step(jobs.get(job_name, {}), step_name)
        if step is None:
            problems.append(f"{job_name}: advisory-шаг '{step_name}' исчез")
        elif not step.get("continue-on-error"):
            problems.append(f"{job_name}: '{step_name}' больше не advisory")
    assert not problems, "Нарушения по advisory-шагам:\n" + "\n".join(problems)


def test_secret_scan_script_is_present() -> None:
    assert (REPO / "scripts" / "check_secrets.py").is_file()
