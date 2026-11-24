"""Гвардии для GitHub Actions workflow (инфраструктура, зона D).

История: однажды в `ci.yml` условие `if: ${{ secrets.X != '' }}` попало в
позицию, где контекст `secrets` недоступен. GitHub не смог разобрать workflow
и **все** прогоны падали с нулём джобов, без внятной ошибки в логах шага.
YAML при этом оставался валидным, поэтому обычный линтер молчал.

Эти тесты ловят такой класс поломок локально, до пуша:

- каждый файл парсится как YAML и имеет `on:` и непустой `jobs`;
- у каждой джобы есть `runs-on` и хотя бы один шаг;
- `secrets.*` не встречается в `if:`-выражениях (известный убийца парсера);
- скрипты, вызываемые из шагов (`python scripts/x.py`), существуют в репозитории;
- деплой не срабатывает сам по push (только ручной запуск).

Стиль: как в tests/test_dockerfile_hardening.py - чтение файлов, без сети.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: YAML 1.1 читает ключ `on:` как boolean True - нормализуем имя триггера.
TRIGGER_KEYS = {"on", True}


def workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml"))


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def jobs_of(doc: dict) -> dict:
    return doc.get("jobs") or {}


def steps_of(job: dict) -> list[dict]:
    return [s for s in (job.get("steps") or []) if isinstance(s, dict)]


def run_commands(doc: dict) -> list[str]:
    return [str(s.get("run") or "") for job in jobs_of(doc).values() for s in steps_of(job)]


def test_workflows_directory_is_not_empty() -> None:
    assert workflow_files(), f"нет workflow-файлов в {WORKFLOWS}"


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_workflow_parses_with_trigger_and_jobs(path: Path) -> None:
    doc = load(path)
    assert isinstance(doc, dict), f"{path.name}: верхний уровень не объект"
    assert TRIGGER_KEYS & set(doc), f"{path.name}: нет секции on: (триггеры)"
    jobs = jobs_of(doc)
    assert jobs, f"{path.name}: нет джоб"


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_every_job_has_runner_and_steps(path: Path) -> None:
    for name, job in jobs_of(load(path)).items():
        assert isinstance(job, dict), f"{path.name}/{name}: джоба не объект"
        assert job.get("runs-on"), f"{path.name}/{name}: нет runs-on"
        assert steps_of(job), f"{path.name}/{name}: нет шагов"


def secrets_in_conditions(doc: dict) -> list[str]:
    """Все `if:`-выражения, где упомянут `secrets.*` (job и step уровни)."""
    offenders: list[str] = []
    for name, job in jobs_of(doc).items():
        if not isinstance(job, dict):
            continue
        if isinstance(job.get("if"), str) and "secrets." in job["if"]:
            offenders.append(f"{name}: {job['if']}")
        for step in steps_of(job):
            condition = step.get("if")
            if isinstance(condition, str) and "secrets." in condition:
                offenders.append(f"{name}/{step.get('name', '?')}: {condition}")
    return offenders


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_secrets_not_used_in_if_expressions(path: Path) -> None:
    """`secrets.*` внутри `if:` ломает разбор workflow у GitHub.

    Секреты нельзя читать в условиях на уровне job/step: контекст недоступен,
    и workflow считается невалидным целиком. Правильный путь - положить секрет
    в `env:` шага и проверять переменную окружения в shell.
    """
    offenders = secrets_in_conditions(load(path))
    assert not offenders, (
        f"{path.name}: secrets.* в if: - GitHub не разберёт workflow: {offenders}"
    )


def test_detector_catches_secrets_in_job_condition() -> None:
    """Негативный контроль: детектор обязан ругаться на реальную поломку."""
    bad = {
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "if": "${{ secrets.DATABASE_URL != '' }}",
                "steps": [{"run": "echo hi"}],
            }
        }
    }
    assert secrets_in_conditions(bad)


def test_detector_catches_secrets_in_step_condition() -> None:
    bad = {
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "guarded",
                        "if": "${{ secrets.TOKEN }}",
                        "run": "echo hi",
                    }
                ],
            }
        }
    }
    assert secrets_in_conditions(bad) == ["test/guarded: ${{ secrets.TOKEN }}"]


def test_detector_allows_secrets_in_env() -> None:
    """Секрет в `env:` шага - законный способ, детектор молчит."""
    good = {
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {
                        "name": "db",
                        "env": {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"},
                        "run": "if [ -z \"$DATABASE_URL\" ]; then exit 0; fi",
                    }
                ],
            }
        }
    }
    assert secrets_in_conditions(good) == []


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_referenced_python_scripts_exist(path: Path) -> None:
    """Шаг вызывает `python scripts/foo.py` - файл обязан быть в репозитории."""
    missing: list[str] = []
    for command in run_commands(load(path)):
        for match in re.finditer(r"python3?\s+([\w./-]+\.py)", command):
            script = match.group(1)
            if not (ROOT / script).is_file():
                missing.append(script)
    assert not missing, f"{path.name}: нет файлов скриптов: {sorted(set(missing))}"


def test_deploy_is_manual_only() -> None:
    """Деплой не должен запускаться сам по push.

    Прод за NAT (Tailscale), автодеплой идёт таймером на сервере; workflow -
    только ручной запуск. Автотриггер по push означал бы, что любой мерж в
    main пытается ходить по SSH с раннера, где этого доступа нет.
    """
    doc = load(WORKFLOWS / "deploy.yml")
    triggers = doc.get("on") if "on" in doc else doc.get(True)
    assert triggers == "workflow_dispatch" or (
        isinstance(triggers, dict) and set(triggers) == {"workflow_dispatch"}
    ), f"deploy.yml должен запускаться только вручную, а не: {triggers}"
