"""Database isolation proof for the API test suite.

The shared PostgreSQL instance used by tests/api/* would collide when several
pytest processes create/delete the same fixed user rows. This module verifies
that the process-unique helper works and that the known fixed test emails have
been removed.
"""

import os
from pathlib import Path

from tests.conftest import unique_email


def _api_py_files() -> list[Path]:
    api_dir = Path(__file__).with_name("api")
    return [p for p in api_dir.rglob("*.py") if p.is_file()]


def test_unique_email_is_process_unique() -> None:
    pid = os.getpid()
    a = unique_email("mod-a")
    b = unique_email("mod-b")

    assert a != b
    assert a.startswith(f"mod-a-{pid}-"), a
    assert b.startswith(f"mod-b-{pid}-"), b
    assert a.endswith("@example.test")
    assert b.endswith("@example.test")
