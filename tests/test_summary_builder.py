"""Tests for src.api_pkg.summary_builder."""

import json

import pytest

from src import config
from src.api_pkg import summary_builder as sb


@pytest.fixture
def sample_recs():
    return {
        "profile_a": {
            "summary": {"score": 12},
            "target_profession": "Data Scientist",
            "dominant_domain_name": "AI",
            "closest_roles": ["ML Engineer"],
            "gaps": ["python"],
            "domain_coverage": 0.7,
            "recommendations": ["learn SQL"],
        },
        "profile_b": {
            "target_profession": "DevOps",
            "gaps": ["docker"],
        },
    }


@pytest.fixture
def populated_result_dir(tmp_path, monkeypatch):
    result_dir = tmp_path / "data" / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATA_RESULT_DIR", result_dir)
    return result_dir


def test_load_recommendations_from_disk_empty(populated_result_dir):
    assert sb.load_recommendations_from_disk() == {}


def test_load_recommendations_from_disk_missing_dir(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_dir"
    monkeypatch.setattr(config, "DATA_RESULT_DIR", missing)
    assert sb.load_recommendations_from_disk() == {}


def test_load_recommendations_from_disk_loads_files(populated_result_dir):
    (populated_result_dir / "p1").mkdir()
    (populated_result_dir / "p1" / "full_recommendations_001.json").write_text(
        json.dumps({"summary": {"score": 5}}), encoding="utf-8"
    )
    (populated_result_dir / "p2").mkdir()
    (populated_result_dir / "p2" / "full_recommendations_002.json").write_text(
        json.dumps({"summary": {"score": 7}}), encoding="utf-8"
    )
    recs = sb.load_recommendations_from_disk()
    assert set(recs.keys()) == {"p1", "p2"}
    assert recs["p1"]["summary"]["score"] == 5


def test_load_recommendations_from_disk_bad_json_is_logged(populated_result_dir, caplog):
    (populated_result_dir / "bad").mkdir()
    (populated_result_dir / "bad" / "full_recommendations_x.json").write_text(
        "not json", encoding="utf-8"
    )
    recs = sb.load_recommendations_from_disk()
    assert recs == {}


def test_build_summary_payload_empty():
    assert sb.build_summary_payload({}) == {}


def test_build_summary_payload_adds_summary_keys(sample_recs):
    payload = sb.build_summary_payload(sample_recs)
    assert payload["profile_a"]["score"] == 12
    assert payload["profile_a"]["target_profession"] == "Data Scientist"


def test_build_summary_payload_handles_missing_summary():
    recs = {"p": {"gaps": ["x"]}}
    payload = sb.build_summary_payload(recs)
    assert payload["p"] == {"gaps": ["x"]}


def test_build_summary_payload_does_not_overwrite_with_none():
    recs = {"p": {"summary": {"score": 1}, "target_profession": None, "gaps": None}}
    payload = sb.build_summary_payload(recs)
    assert payload["p"]["score"] == 1
    assert "target_profession" not in payload["p"]
    assert "gaps" not in payload["p"]
