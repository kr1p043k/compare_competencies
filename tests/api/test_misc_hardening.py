"""Unit: clusters summary route + log caps."""

from src.api_pkg import create_app
from tests.conftest import all_route_paths


def test_clusters_summary_registered():
    paths = sorted(all_route_paths(create_app()))
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
