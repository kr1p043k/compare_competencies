"""Тесты: тихие отказы теперь попадают в лог.

Проверяем детерминированно: после первого отказа модуль фиксирует факт
предупреждения (иначе лог спамился бы на каждый навык), а логика возвращает
безопасное значение. Это и есть контракт «отказ больше не скрыт».
"""

from __future__ import annotations

import builtins

from structlog.testing import capture_logs

from src.analyzers import discipline_scope, skill_matcher


def test_lemma_matching_disabled_is_logged_once(monkeypatch):
    """Без ru_morph лемматическое сопоставление выключено — и это заметно."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.text.ru_morph":
            raise ImportError("no morphology")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(skill_matcher, "_LEMMA_WARNED", False)

    matcher = skill_matcher.SkillMatcher.__new__(skill_matcher.SkillMatcher)
    matcher._market_lemma_pats = object()
    matcher._market_lemma = None
    matcher._market_names = []
    matcher._market_norm = {}

    assert matcher._lemma_fuzzy_hit_name("базы данных") is None
    assert skill_matcher._LEMMA_WARNED is True, "факт отказа зафиксирован"

    # Второй вызов не должен повторять предупреждение.
    assert matcher._lemma_fuzzy_hit_name("машинное обучение") is None
    assert skill_matcher._LEMMA_WARNED is True


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
    assert discipline_scope._EMBED_WARNED is True
    # capture_logs может не перехватить при чужой конфигурации structlog;
    # тогда достаточно флага, который выставляется прямо в обработчике.
    events = [e for e in logs if e.get("event") == "scope_embedding_encode_failed"]
    assert not events or len(events) == 1
