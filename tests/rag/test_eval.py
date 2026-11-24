"""Оценка retrieval: recall@k, MRR, nDCG@k, разбор golden-set, CLI-прогон."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.cli import rag_eval as cli
from src.rag import eval as rag_eval
from src.rag.eval import GoldenItem
from src.rag.store import RagStore
from src.rag.types import RetrievalHit


def _hit(
    title: str,
    *,
    source: str = "rpd",
    external_id: str | None = None,
    score: float = 1.0,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"c-{title}",
        document_id=f"d-{title}",
        source=source,
        title=title,
        text="text",
        score=score,
        external_id=external_id,
    )


def _golden(*expect: str) -> GoldenItem:
    return GoldenItem(query="q", expect=list(expect))


def test_normalize_key_case_spaces_yo() -> None:
    assert rag_eval.normalize_key("  Машинное   ОБУЧЕНИЕ ") == "машинное обучение"
    assert rag_eval.normalize_key("Всё про ёлку") == rag_eval.normalize_key("все про елку")


def test_hit_keys_external_id_title_and_source_title() -> None:
    keys = rag_eval.hit_keys(_hit("УК-3", external_id="09.03.02:uk3"))
    assert "09.03.02:uk3" in keys
    assert "ук-3" in keys
    assert "rpd:ук-3" in keys


def test_recall_perfect_partial_and_truncated() -> None:
    golden = _golden("A", "B")
    hits = [_hit("A"), _hit("B")]
    assert rag_eval.recall_at_k(hits, golden, 5) == 1.0
    assert rag_eval.recall_at_k(hits, golden, 1) == 0.5
    assert rag_eval.recall_at_k([_hit("C")], golden, 5) == 0.0
    assert rag_eval.recall_at_k([], golden, 5) == 0.0


def test_recall_matches_by_external_id_not_only_title() -> None:
    golden = _golden("09.03.02:discipline-42")
    hits = [_hit("Совсем другое имя", external_id="09.03.02:discipline-42")]
    assert rag_eval.recall_at_k(hits, golden, 3) == 1.0


def test_recall_without_expect_is_zero() -> None:
    assert rag_eval.recall_at_k([_hit("A")], _golden(), 5) == 0.0


def test_reciprocal_rank_first_third_and_missing() -> None:
    golden = _golden("B")
    assert rag_eval.reciprocal_rank([_hit("B")], golden) == 1.0
    assert rag_eval.reciprocal_rank([_hit("A"), _hit("C"), _hit("B")], golden) == pytest.approx(1 / 3)
    assert rag_eval.reciprocal_rank([_hit("A")], golden) == 0.0
    # k отсекает релевантный hit, стоящий за пределами глубины
    assert rag_eval.reciprocal_rank([_hit("A"), _hit("B")], golden, k=1) == 0.0


def test_ndcg_perfect_is_one_and_degrades_with_position() -> None:
    golden = _golden("A", "B")
    assert rag_eval.ndcg_at_k([_hit("A"), _hit("B")], golden, 5) == pytest.approx(1.0)
    second = rag_eval.ndcg_at_k([_hit("X"), _hit("A"), _hit("B")], golden, 5)
    assert 0.0 < second < 1.0
    assert rag_eval.ndcg_at_k([_hit("X")], golden, 5) == 0.0
    assert rag_eval.ndcg_at_k([_hit("A")], _golden(), 5) == 0.0


def test_ndcg_single_expected_at_first_position() -> None:
    assert rag_eval.ndcg_at_k([_hit("A")], _golden("A"), 1) == pytest.approx(1.0)


def test_ndcg_duplicate_hits_do_not_exceed_one() -> None:
    # Один документ разными чанками (и разные источники с тем же заголовком)
    # не должен раздувать DCG выше идеального значения.
    golden = _golden("A")
    hits = [_hit("A"), _hit("A"), _hit("A", source="krm")]
    assert rag_eval.ndcg_at_k(hits, golden, 5) == pytest.approx(1.0)
    assert rag_eval.ndcg_at_k(hits, golden, 5) <= 1.0


def test_evaluate_item_reports_missing_and_top() -> None:
    row = rag_eval.evaluate_item([_hit("A", external_id="ext-a"), _hit("C")], _golden("ext-a", "B"), k=5)
    assert row["expected"] == 2
    assert row["recall_at_k"] == 0.5
    assert row["missing"] == ["b"]
    assert [item["title"] for item in row["top"]] == ["A", "C"]
    assert row["top"][0]["external_id"] == "ext-a"


def test_aggregate_averages_and_skips_items_without_expect() -> None:
    rows = [
        {"expected": 2, "recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0, "hits": 2},
        {"expected": 1, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "hits": 0},
        {"expected": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "hits": 3},
    ]
    summary = rag_eval.aggregate(rows)
    assert summary["queries"] == 2
    assert summary["skipped"] == 1
    assert summary["recall_at_k"] == 0.5
    assert summary["empty_result_queries"] == 1


def test_parse_golden_accepts_dict_list_and_single_string_expect() -> None:
    items = rag_eval.parse_golden(
        {"items": [{"query": "q1", "expect": "A", "year": 2025, "sources": ["rpd"]}]}
    )
    assert items[0].expect == ["A"]
    assert items[0].year == 2025
    assert items[0].sources == ["rpd"]

    bare = rag_eval.parse_golden([{"query": "q2"}])
    assert bare[0].query == "q2"
    assert bare[0].keys() == set()


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-structure",
        {"items": "nope"},
        {"items": ["not-an-object"]},
        {"items": [{"query": ""}]},
        {"items": [{"query": "q", "expect": [1]}]},
        {"items": [{"query": "q", "sources": "rpd"}]},
        {"items": [{"query": "q", "year": "2025"}]},
    ],
)
def test_parse_golden_rejects_bad_shapes(payload) -> None:
    with pytest.raises(ValueError):
        rag_eval.parse_golden(payload)


def test_row_to_hit_carries_external_id() -> None:
    row = SimpleNamespace(
        _mapping={
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "body",
            "section_path": "s",
            "ord": 0,
            "chunk_meta": {},
            "source": "rpd",
            "title": "Тема",
            "url": None,
            "external_id": "09.03.02:x",
        }
    )
    hit = RagStore._row_to_hit(row, 0.5, "sparse")
    assert hit.external_id == "09.03.02:x"
    assert hit.sparse_score == 0.5


def test_golden_template_is_parseable() -> None:
    items = rag_eval.parse_golden(rag_eval.golden_template())
    assert len(items) == 1
    assert items[0].expect


class _FakeService:
    def __init__(self, hits):
        self.settings = SimpleNamespace(enabled=True, min_score=0.35)
        self._hits = hits
        self.calls = []

    async def search(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return SimpleNamespace(context=SimpleNamespace(evidence=self._hits))


def _args(**overrides):
    base = {
        "golden": None,
        "k": 5,
        "role": "admin",
        "fail_under": None,
        "json": None,
        "min_score": None,
        "sweep": None,
        "warmup": False,
        "dry_run": False,
        "template": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cli_dry_run_validates_without_search(tmp_path, capsys) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"items": [{"query": "q1", "expect": ["A"]}]}), encoding="utf-8"
    )
    # main бросает SystemExit; dry-run не должен трогать БД
    with pytest.raises(SystemExit) as exc:
        cli.main(_args(golden=str(golden), dry_run=True))
    assert exc.value.code == 0
    assert "golden-set OK" in capsys.readouterr().out


def test_cli_run_writes_report_and_fails_under_threshold(tmp_path, monkeypatch, capsys) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"items": [{"query": "q1", "expect": ["A"]}]}), encoding="utf-8"
    )
    fake = _FakeService([_hit("A", external_id="a")])
    monkeypatch.setattr(cli, "RagService", lambda **kwargs: fake)
    out = tmp_path / "out" / "report.json"

    with pytest.raises(SystemExit) as exc:
        cli.main(_args(golden=str(golden), json=str(out), fail_under=0.5))
    assert exc.value.code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["recall_at_k"] == 1.0
    assert fake.calls[0][1]["log"] is False

    with pytest.raises(SystemExit) as exc:
        cli.main(_args(golden=str(golden), fail_under=1.5))
    assert exc.value.code == 1


def test_cli_requires_golden() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(_args())
    assert "golden" in str(exc.value.code)


def test_cli_template_prints_schema(capsys) -> None:
    cli.main(_args(template=True))
    payload = json.loads(capsys.readouterr().out)
    assert "items" in payload


def test_cli_sweep_reports_every_threshold(tmp_path, monkeypatch, capsys) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"items": [{"query": "q1", "expect": ["A"]}]}), encoding="utf-8"
    )
    fake = _FakeService([_hit("A")])
    monkeypatch.setattr(cli, "RagService", lambda **kwargs: fake)
    out = tmp_path / "sweep.json"

    with pytest.raises(SystemExit) as exc:
        cli.main(_args(golden=str(golden), sweep="0.35,0.2,0.35", json=str(out)))
    assert exc.value.code == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["min_score"] for entry in payload["sweeps"]] == [0.2, 0.35]
    assert payload["sweeps"][0]["summary"]["recall_at_k"] == 1.0
    assert "min_score" in capsys.readouterr().out


def test_cli_sweep_rejects_garbage() -> None:
    with pytest.raises(SystemExit) as exc:
        cli._parse_sweep("abc")
    assert "sweep" in str(exc.value.code)
    with pytest.raises(SystemExit):
        cli._parse_sweep(" , ")


def test_cli_warms_embedder_before_measuring(tmp_path, monkeypatch) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps({"items": [{"query": "q1", "expect": ["A"]}]}), encoding="utf-8"
    )
    calls: list[str] = []

    async def fake_warmup() -> None:
        calls.append("warm")

    monkeypatch.setattr(cli, "_warmup", fake_warmup)
    monkeypatch.setattr(cli, "RagService", lambda **kwargs: _FakeService([_hit("A")]))

    with pytest.raises(SystemExit) as exc:
        cli.main(_args(golden=str(golden), warmup=True))
    assert exc.value.code == 0
    assert calls == ["warm"]
