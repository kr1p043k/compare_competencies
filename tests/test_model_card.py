"""Генератор Model Card: сборка markdown из индекса реестра моделей."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_model_card import build_model_card, main


def test_empty_registry_mentions_no_models():
    card = build_model_card({}, generated_at="2026-01-01T00:00:00+00:00")
    assert "# Model Registry" in card
    assert "не зарегистрированы" in card
    assert "2026-01-01T00:00:00+00:00" in card


def test_card_lists_models_metrics_and_size():
    index = {
        "ltr": [
            {
                "version": "2",
                "path": "C:/x/model.joblib",
                "size_bytes": 3 * 1024 * 1024,
                "metrics": {"ndcg@10": 0.62},
                "tags": {"role": "prod"},
                "registered_at": "2026-01-02T00:00:00+00:00",
            }
        ]
    }
    card = build_model_card(index, generated_at="2026-01-02T00:00:00+00:00")
    assert "## ltr" in card
    assert "| 2 |" in card
    assert "3.0" in card  # 3 MiB
    assert "ndcg@10=0.62" in card
    assert "role=prod" in card


def test_main_writes_from_registry_dir(tmp_path: Path):
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "registry_index.json").write_text(
        json.dumps({"clusters": [{"version": "1", "size_bytes": 10, "metrics": {}, "tags": {}}]}),
        encoding="utf-8",
    )
    out = tmp_path / "MODEL_CARD.md"
    written = main(out=out, registry_dir=reg)
    assert written == out
    text = out.read_text(encoding="utf-8")
    assert "## clusters" in text


def test_main_without_registry_writes_placeholder(tmp_path: Path):
    out = tmp_path / "MODEL_CARD.md"
    main(out=out, registry_dir=tmp_path / "missing")
    assert "не зарегистрированы" in out.read_text(encoding="utf-8")
