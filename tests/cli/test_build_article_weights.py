"""Тесты сборщика научных весов: разбор источников, извлечение, нормировка."""

from __future__ import annotations

from src.cli import build_article_weights as bw
from src.parsing.skills.skill_parser import SkillParser


def test_parse_openalex_reconstructs_abstract():
    payload = {
        "results": [
            {
                "title": "Deep learning for code",
                "publication_year": 2024,
                "abstract_inverted_index": {"We": [0], "use": [1], "PyTorch": [2]},
                "keywords": [{"display_name": "Machine Learning"}],
            }
        ]
    }
    docs = bw.parse_openalex_works(payload)
    assert len(docs) == 1
    doc = docs[0]
    assert doc["abstract"] == "We use PyTorch"
    assert doc["keywords"] == ["Machine Learning"]
    assert doc["year"] == 2024
    assert doc["source"] == "openalex"


def test_parse_openalex_tolerates_garbage():
    assert bw.parse_openalex_works({}) == []
    assert bw.parse_openalex_works({"results": "nope"}) == []
    docs = bw.parse_openalex_works({"results": [None, {"title": None}]})
    assert len(docs) == 1
    assert docs[0]["title"] == ""
    assert docs[0]["abstract"] == ""
    assert docs[0]["year"] is None


def test_parse_arxiv_feed_extracts_fields():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Scaling laws for retrieval</title>
        <summary>We study embeddings and rerankers.</summary>
        <published>2025-02-03T10:00:00Z</published>
        <category term="cs.IR"/>
        <category term="cs.LG"/>
      </entry>
    </feed>"""
    docs = bw.parse_arxiv_feed(xml)
    assert len(docs) == 1
    assert docs[0]["title"] == "Scaling laws for retrieval"
    assert docs[0]["keywords"] == ["cs.IR", "cs.LG"]
    assert docs[0]["year"] == 2025
    assert docs[0]["source"] == "arxiv"


def test_parse_arxiv_feed_broken_xml_is_empty():
    assert bw.parse_arxiv_feed("<feed><entry>") == []
    assert bw.parse_arxiv_feed("") == []


def test_extract_document_skills_counts_documents_not_mentions():
    parser = SkillParser()
    docs = [
        {"title": "Python and python again", "abstract": "SQL too", "keywords": []},
        {"title": "Only python", "abstract": "", "keywords": ["SQL"]},
        {"title": "", "abstract": "", "keywords": []},
    ]
    counts = bw.extract_document_skills(docs, parser, {"python", "sql"})
    assert counts == {"python": 2, "sql": 2}


def test_extract_document_skills_respects_vocab():
    parser = SkillParser()
    docs = [{"title": "python and kubernetes", "abstract": "", "keywords": []}]
    counts = bw.extract_document_skills(docs, parser, {"python"})
    assert counts == {"python": 1}


def test_extract_document_skills_drops_prose_stopwords():
    """«can» в it_skills - это CAN bus, но в абстракте это модальный глагол."""
    parser = SkillParser()
    docs = [
        {"title": "Can python can scale? This can work.", "abstract": "We can use python.", "keywords": []},
    ]
    counts = bw.extract_document_skills(docs, parser, {"python", "can"})
    assert counts == {"python": 1}


def test_extract_document_skills_reads_long_abstracts():
    parser = SkillParser()
    filler = "neutral sentence about nothing. " * 40
    docs = [{"title": "", "abstract": filler + "pytorch", "keywords": []}]
    counts = bw.extract_document_skills(docs, parser, {"pytorch"})
    assert counts == {"pytorch": 1}


def test_normalize_weights_drops_rare_and_scales_to_one():
    out = bw.normalize_weights({"python": 10, "sql": 5, "cobol": 1}, min_freq=2)
    assert out == {"python": 1.0, "sql": 0.5}
    assert bw.normalize_weights({"x": 1}, min_freq=2) == {}
    assert bw.normalize_weights({}) == {}


def test_build_payload_schema_and_global_from_sums():
    payload = bw.build_payload(
        {
            "Data Scientist": {"python": 10, "pytorch": 5},
            "Backend Developer": {"python": 5, "sql": 4},
            "Empty": {"rare": 1},
        },
        {"sources": ["openalex"], "per_query": 5, "since": "2024-01-01"},
        min_freq=2,
    )
    assert set(payload) == {"_meta", "roles", "global"}
    assert payload["_meta"]["normalization"] == "per_role_max"
    assert payload["_meta"]["professions"] == 2
    assert payload["_meta"]["sources"] == ["openalex"]
    assert payload["_meta"]["generated_at"]
    assert payload["roles"]["Data Scientist"] == {"python": 1.0, "pytorch": 0.5}
    assert payload["roles"]["Backend Developer"] == {"python": 1.0, "sql": 0.8}
    assert "Empty" not in payload["roles"]
    # global считается по сумме сырых частот: python 15, pytorch 5, sql 4
    assert payload["global"]["python"] == 1.0
    assert payload["global"]["pytorch"] == round(5 / 15, 4)
    assert payload["global"]["sql"] == round(4 / 15, 4)


def test_payload_feeds_science_weights(tmp_path):
    """Артефакт сборщика читается потребителем без правок схемы."""
    from src.analyzers.skills.science_weights import RoleSourceMix

    payload = bw.build_payload({"Data Scientist": {"python": 4, "pytorch": 2}}, {}, min_freq=2)
    article = tmp_path / "article.json"
    article.write_text(__import__("json").dumps(payload), encoding="utf-8")
    mix = tmp_path / "mix.json"
    mix.write_text(__import__("json").dumps({"default": 0.5}), encoding="utf-8")
    m = RoleSourceMix.from_files(article, mix)
    assert m.active
    assert m.lambda_for("anything") == 0.5
    out = m.blend({"python": 1.0, "sql": 0.4}, "anything")
    assert out["python"] == 1.0
    assert out["sql"] == 0.4

def test_previous_role_weights_and_carry_forward_counts(tmp_path):
    target = tmp_path / "article.json"
    target.write_text(
        __import__("json").dumps({"roles": {"QA Engineer": {"qa": 1.0, "sql": 0.4}, "Bad": {}}}),
        encoding="utf-8",
    )
    prev = bw._previous_role_weights(target)
    assert prev == {"QA Engineer": {"qa": 1.0, "sql": 0.4}}

    counts = bw._weights_to_counts(prev["QA Engineer"], min_freq=2)
    assert counts == {"qa": 100, "sql": 40}
    # нормировка после обратного пересчёта совпадает с исходной
    assert bw.normalize_weights(counts, min_freq=2) == {"qa": 1.0, "sql": 0.4}


def test_previous_role_weights_tolerates_missing_or_broken(tmp_path):
    assert bw._previous_role_weights(tmp_path / "nope.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert bw._previous_role_weights(broken) == {}
    no_roles = tmp_path / "noroles.json"
    no_roles.write_text('{"roles": []}', encoding="utf-8")
    assert bw._previous_role_weights(no_roles) == {}

def test_subset_run_preserves_other_roles(tmp_path, monkeypatch):
    """--professions X не должен затирать остальные роли в артефакте."""
    target = tmp_path / "article.json"
    target.write_text(
        __import__("json").dumps({"roles": {"QA Engineer": {"qa": 1.0}, "Data Scientist": {"sql": 1.0}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bw, "_collect_docs", lambda q, n, s, since, pause=None: [{"title": "python", "abstract": ""}]
    )
    bw.main(professions=["Data Scientist"], per_query=1, out=str(target), min_freq=1)

    payload = __import__("json").loads(target.read_text(encoding="utf-8"))
    assert set(payload["roles"]) == {"Data Scientist", "QA Engineer"}
    assert payload["roles"]["QA Engineer"] == {"qa": 1.0}
    assert payload["roles"]["Data Scientist"] == {"python": 1.0}
    assert payload["_meta"]["carried_over"] == ["QA Engineer"]

def test_get_json_retries_on_429(monkeypatch):
    """429 - это «подожди», а не «данных нет»: роль не должна терять документы."""
    calls = []

    class _Resp:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return {"results": [{"title": "ok"}]}

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _Resp(429) if len(calls) < 3 else _Resp(200)

    monkeypatch.setattr(bw.httpx, "get", fake_get)
    monkeypatch.setattr(bw, "RETRY_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(bw.time, "sleep", lambda s: None)

    payload = bw._get_json("https://api.openalex.org/works", {"search": "x"}, 1.0)
    assert payload == {"results": [{"title": "ok"}]}
    assert len(calls) == 3


def test_get_json_gives_up_after_backoff(monkeypatch):
    class _Resp:
        status_code = 429

        def raise_for_status(self):
            raise RuntimeError("HTTP 429")

        def json(self):
            return {}

    monkeypatch.setattr(bw.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(bw, "RETRY_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(bw.time, "sleep", lambda s: None)
    assert bw._get_json("https://api.openalex.org/works", {}, 1.0) is None


def test_openalex_mailto_from_env(monkeypatch):
    monkeypatch.delenv("OPENALEX_MAILTO", raising=False)
    assert bw._openalex_mailto() is None
    monkeypatch.setenv("OPENALEX_MAILTO", "  dev@example.org ")
    assert bw._openalex_mailto() == "dev@example.org"
