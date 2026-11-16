"""Тесты валидатора n8n-воркфлоу (BACKLOG 6.7.5)."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import check_n8n_workflows as chk  # noqa: E402


def _write(tmp_path: pathlib.Path, name: str, payload: dict | str, *, bom: bool = False) -> pathlib.Path:
    path = tmp_path / name
    text = payload if isinstance(payload, str) else json.dumps(payload)
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    return path


def _good() -> dict:
    return {
        "nodes": [
            {"name": "Start", "type": "n8n-nodes-base.start", "parameters": {}},
            {"name": "HTTP", "type": "n8n-nodes-base.httpRequest", "parameters": {}},
        ],
        "connections": {"Start": {"main": [[{"node": "HTTP", "type": "main", "index": 0}]]}},
    }


def test_repo_workflows_are_valid():
    assert chk.check_all() == 0


def test_good_workflow_has_no_problems(tmp_path):
    problems, warnings = chk.validate_file(_write(tmp_path, "ok.json", _good()))
    assert problems == []
    assert warnings == []


def test_missing_nodes_reported(tmp_path):
    problems, _ = chk.validate_file(_write(tmp_path, "bad.json", {"connections": {}}))
    assert any("nodes" in p for p in problems)


def test_duplicate_node_names_reported(tmp_path):
    payload = _good()
    payload["nodes"].append({"name": "Start", "type": "n8n-nodes-base.noOp", "parameters": {}})
    problems, _ = chk.validate_file(_write(tmp_path, "dup.json", payload))
    assert any("дубли" in p for p in problems)


def test_bom_is_warning_not_problem(tmp_path):
    problems, warnings = chk.validate_file(_write(tmp_path, "bom.json", _good(), bom=True))
    assert problems == []
    assert any("BOM" in w for w in warnings)


def test_invalid_json_reported(tmp_path):
    problems, _ = chk.validate_file(_write(tmp_path, "broken.json", "{not json"))
    assert any("некорректный JSON" in p for p in problems)


def test_secret_literal_reported(tmp_path):
    payload = _good()
    payload["nodes"][0]["parameters"] = {"apiKey": "sk-abcdefghijklmnop12345"}
    problems, _ = chk.validate_file(_write(tmp_path, "secret.json", payload))
    assert any("секрет" in p for p in problems)


@pytest.mark.parametrize("name", ["nightly_pipeline.json", "trend_alert.json"])
def test_known_workflows_exist(name):
    assert (chk.WORKFLOWS_DIR / name).is_file()
