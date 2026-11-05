"""Тесты: тихие отказы теперь попадают в лог.

Каждый тест падает, если соответствующий обработчик снова начнёт глотать
исключение без записи. Логика не меняется — проверяется только видимость.
"""

from __future__ import annotations

from structlog.testing import capture_logs

from src.analyzers import discipline_scope, skill_matcher


def test_lemma_matching_disabled_is_logged_once(monkeypatch):
    """Без ru_morph лемматическое сопоставление выключено — это видно в логе."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.text.ru_morph":
            raise ImportError("no morphology")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(skill_matcher, "_LEMMA_WARNED", False)

    matcher = skill_matcher.SkillMatcher.__new__(skill_matcher.SkillMatcher)
    matcher._market_lemma_pats = object()  # чтобы дойти до импорта
    matcher._market_lemma = None
    matcher._market_names = []
    matcher._market_norm = {}

    with capture_logs() as logs:
        matcher._lemma_fuzzy_hit_name("базы данных")
        matcher._lemma_fuzzy_hit_name("машинное обучение")

    warns = [e for e in logs if e["event"] == "ru_morph_unavailable_lemma_matching_disabled"]
    assert len(warns) == 1, "предупреждение должно быть ровно одно, а не на каждый навык"


def test_scope_embedding_failure_is_logged(monkeypatch):
    """Упавший провайдер эмбеддингов больше не оставляет профиль пустым молча."""

    class BrokenProvider:
        def encode(self, *a, **k):
            raise RuntimeError("model crashed")

    class FakeMatcher:
        _market_embeddings = [[0.1, 0.2]]
        _market_names = ["sql"]
        _embedding_provider = BrokenProvider()

    monkeypatch.setattr(discipline_scope, "_EMBED_WARNED", False)

    with capture_logs() as logs:
        out = discipline_scope.neighbours(FakeMatcher(), "базы данных")

    assert out == []
    events = [e for e in logs if e["event"] == "scope_embedding_encode_failed"]
    assert len(events) == 1
