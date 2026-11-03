"""Тесты нормализатора ответов Course Trends API.

Схем ответов сервис не публикует, поэтому адаптер обязан переживать разные
имена полей. Здесь проверяется именно это: одна и та же статья в трёх
«диалектах» даёт один канонический вид.
"""

from __future__ import annotations

from src.infrastructure.course_trends_normalize import (
    extract_articles,
    looks_like_article,
    normalize_article,
)


def test_openalex_style_payload():
    payload = {
        "results": [
            {
                "title": "Graph Neural Networks for Skill Extraction",
                "abstract": "We study...",
                "publication_year": 2024,
                "doi": "10.1234/abc.1",
                "cited_by_count": 42,
                "authorships": [{"author": {"display_name": "Ivanov I."}}],
                "primary_location": {"landing_page_url": "https://example.org/p1"},
            }
        ]
    }
    articles = extract_articles(payload)
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Graph Neural Networks for Skill Extraction"
    assert a["year"] == 2024
    assert a["citations"] == 42
    assert a["url"] == "https://example.org/p1"
    assert a["doi"] == "10.1234/abc.1"


def test_minimal_style_payload():
    payload = {"items": [{"name": "Метод анализа РПД", "date": "2023-05-01"}]}
    articles = extract_articles(payload)
    assert len(articles) == 1
    assert articles[0]["title"] == "Метод анализа РПД"
    assert articles[0]["year"] == 2023


def test_russian_style_payload():
    payload = {
        "статьи": [
            {"название": "Компетентностный подход", "год": "2022",
             "аннотация": "Обзор", "авторы": ["Петров П.П.", "Сидоров С.С."]}
        ]
    }
    articles = extract_articles(payload)
    assert len(articles) == 1
    assert articles[0]["year"] == 2022
    assert articles[0]["authors"] == ["Петров П.П.", "Сидоров С.С."]


def test_deduplicates_same_doi_repeated_nested():
    payload = {
        "a": {"title": "Same paper", "doi": "10.1000/x", "year": 2020},
        "b": {"title": "Same paper (duplicate)", "doi": "10.1000/x", "year": 2020},
    }
    assert len(extract_articles(payload)) == 1


def test_deduplicates_by_title_when_no_doi():
    payload = {
        "a": {"title": "Один и тот же  заголовок", "year": 2020},
        "b": {"title": "один и тот же заголовок", "year": 2020},
    }
    assert len(extract_articles(payload)) == 1


def test_course_is_propagated_from_parent():
    payload = {
        "courses": [
            {"course": "Базы данных", "articles": [{"title": "SQL оптимизация", "year": 2021}]}
        ]
    }
    articles = extract_articles(payload)
    assert articles[0]["course"] == "Базы данных"


def test_rejects_non_articles():
    assert not looks_like_article({"pid": "09.03.02_2025"})
    assert not looks_like_article({"title": "   "})
    assert not looks_like_article({"name": "Python", "n_courses": 5})


def test_doi_used_as_url_fallback():
    article = normalize_article({"title": "Paper", "doi": "10.5555/xyz"})
    assert article["url"] == "https://doi.org/10.5555/xyz"


def test_invalid_doi_dropped():
    article = normalize_article({"title": "Paper", "doi": "not-a-doi"})
    assert article["doi"] is None


def test_deep_payload_does_not_recurse_forever():
    node: dict = {"title": "deep", "year": 2020}
    for _ in range(40):
        node = {"nested": node}
    assert extract_articles(node) == []  # глубже MAX_DEPTH - не считаем статьёй


def test_empty_and_garbage_payloads():
    for payload in (None, [], {}, "string", 42, [1, 2, 3]):
        assert extract_articles(payload) == []
