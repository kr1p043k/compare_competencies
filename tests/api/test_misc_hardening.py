"""Unit: clusters summary route + log caps."""

from src.api_pkg import create_app


def test_clusters_summary_registered():
    paths = sorted({r.path for r in create_app().routes if hasattr(r, 'path')})
    assert "/api/clusters/summary" in paths


def test_log_entry_caps():
    import pytest
    from pydantic import ValidationError

    from src.api_pkg.routers.health import LogEntry
    LogEntry(message="x" * 2000)
    with pytest.raises(ValidationError):
        LogEntry(message="x" * 2001)
    with pytest.raises(ValidationError):
        LogEntry(message="ok", data={f"k{i}": "v" for i in range(25)})
