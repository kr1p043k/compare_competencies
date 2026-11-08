"""Путь к .env у раннера teacher-анализа.

Регресс: `load_dotenv(Path(__file__).resolve().parent.parent / ".env")` вёл в
`src/.env`, которого нет, поэтому load_dotenv молча ничего не делал и раннер
не получал переменные окружения из корневого .env.
"""
from __future__ import annotations


def test_env_path_points_to_repo_root() -> None:
    from src.pipeline.teacher_analysis_runner import ENV_PATH

    assert ENV_PATH.name == ".env"
    # Корень репозитория узнаём по alembic.ini, а не по наличию .env:
    # в чистом worktree .env может быть не создан, а alembic.ini есть всегда.
    assert (ENV_PATH.parent / "alembic.ini").exists(), ENV_PATH
    assert ENV_PATH.parent.name != "src", "путь не должен вести в src/"


def test_load_dotenv_is_called_with_env_path() -> None:
    """load_dotenv получает именно ENV_PATH, а не другой путь."""
    from src.pipeline import teacher_analysis_runner as runner

    src = __import__("pathlib").Path(runner.__file__).read_text(encoding="utf-8")
    assert "load_dotenv(ENV_PATH)" in src
    assert 'parent / ".env"' not in src
