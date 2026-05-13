"""BM25 ранкер с предфильтрацией и кэшированием."""

import re

import numpy as np
import pymorphy3
import structlog
from rank_bm25 import BM25Okapi

from src import config
from src.parsing.skills.skill_normalizer import SkillNormalizer
from src.parsing.utils import load_it_skills

logger = structlog.get_logger(__name__)


class BM25Ranker:
    def __init__(self):
        self._morph = pymorphy3.MorphAnalyzer()
        self._cached_corpus = None
        self._corpus_hash = None
        self._stop_lemmas = self._build_stop_lemmas()

    def _build_stop_lemmas(self):
        return {
            "в",
            "без",
            "до",
            "из",
            "к",
            "на",
            "по",
            "о",
            "от",
            "перед",
            "при",
            "через",
            "с",
            "у",
            "за",
            "над",
            "об",
            "под",
            "про",
            "для",
            "и",
            "да",
            "или",
            "либо",
            "не",
            "ни",
            "как",
            "так",
            "то",
            "что",
            "чтобы",
            "если",
            "хотя",
            "пока",
            "когда",
            "где",
            "который",
            "этот",
            "тот",
            "мой",
            "твой",
            "свой",
            "наш",
            "ваш",
            "весь",
            "всякий",
            "любой",
            "человек",
            "год",
            "раз",
            "дело",
            "жизнь",
            "день",
            "время",
            "работа",
            "сила",
            "рука",
            "слово",
            "место",
            "часть",
            "город",
            "страна",
            "опыт",
            "знание",
            "умение",
            "владение",
            "навык",
            "разработка",
            "программирование",
            "язык",
            "технология",
            "система",
            "решение",
            "задача",
            "проект",
            "команда",
            "компания",
            "клиент",
            "сервер",
            "поддержка",
            "сопровождение",
            "настройка",
            "обеспечение",
            "анализ",
            "тестирование",
            "отладка",
            "документация",
            "обучение",
            "мониторинг",
            "управление",
            "процесс",
            "функция",
            "модуль",
            "архитектура",
            "инфраструктура",
            "платформа",
            "среда",
            "код",
            "данные",
            "алгоритм",
            "модель",
            "метод",
            "подход",
            "практика",
            "стандарт",
            "версия",
            "релиз",
            "сборка",
            "деплой",
            "интеграция",
            "миграция",
            "контроль",
            "планирование",
            "оценка",
            "риск",
            "качество",
            "производительность",
            "масштабирование",
            "безопасность",
            "сеть",
            "база",
            "хранилище",
            "облако",
            "кластер",
            "контейнер",
            "виртуализация",
            "оркестрация",
            "автоматизация",
            "интерфейс",
            "пользователь",
            "администратор",
            "разработчик",
            "специалист",
            "инженер",
            "аналитик",
            "менеджер",
            "руководитель",
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
            "what",
            "so",
            "up",
            "out",
            "if",
            "about",
            "who",
            "get",
            "which",
            "go",
            "me",
            "when",
            "make",
            "can",
            "like",
            "time",
            "no",
            "just",
            "him",
            "know",
            "take",
            "person",
            "into",
            "year",
            "your",
            "good",
            "some",
            "could",
            "them",
            "see",
            "other",
            "than",
            "then",
            "now",
            "look",
            "only",
            "come",
            "its",
            "over",
            "think",
            "also",
            "back",
            "after",
            "use",
            "two",
            "how",
            "our",
            "work",
            "first",
            "well",
            "way",
            "even",
            "new",
            "want",
            "because",
            "any",
            "these",
            "give",
            "day",
            "most",
            "us",
            "is",
            "was",
            "are",
            "been",
            "has",
            "had",
            "were",
            "said",
            "did",
            "may",
            "am",
        }

    def _compute_corpus_hash(self, vacancies: list) -> str:  # <-- переименован
        total_ids = 0
        count = 0
        for v in vacancies:
            vid = v.get("id", "") if isinstance(v, dict) else v.id
            if vid:
                total_ids += int(vid) if str(vid).isdigit() else hash(str(vid))
                count += 1
        return f"{count}:{total_ids}"

    def calculate_weights(self, vacancies: list) -> dict[str, float]:
        ch = self._compute_corpus_hash(vacancies)
        if self._cached_corpus is not None and self._corpus_hash == ch:
            logger.info("BM25 из кэша")
            return self._cached_corpus

        whitelist_raw = load_it_skills()
        whitelist = set()
        for skill in whitelist_raw:
            norm = SkillNormalizer.normalize(skill)
            if norm:
                whitelist.add(norm)
        whitelist_words = {s for s in whitelist if " " not in s}
        whitelist_phrases = {s for s in whitelist if " " in s}

        tokenized_corpus = []
        all_ngrams = set()

        for vac in vacancies:
            parts = []
            if isinstance(vac, dict):
                desc = vac.get("description") or ""
                if desc:
                    parts.append(re.sub(r"<[^>]+>", " ", desc))
                snippet = vac.get("snippet") or {}
                req = snippet.get("requirement") or ""
                resp = snippet.get("responsibility") or ""
                if req:
                    parts.append(re.sub(r"<[^>]+>", " ", req))
                if resp:
                    parts.append(re.sub(r"<[^>]+>", " ", resp))
                key_skills = " ".join(s.get("name", "") for s in vac.get("key_skills", []))
                if key_skills:
                    parts.append(key_skills)
            else:
                if hasattr(vac, "description") and vac.description:
                    parts.append(re.sub(r"<[^>]+>", " ", vac.description))
                key_skills = " ".join(s.name for s in (vac.key_skills if hasattr(vac, "key_skills") else []))
                if key_skills:
                    parts.append(key_skills)

            text = " ".join(p.strip() for p in parts if p and p.strip())
            if not text:
                continue

            words = re.findall(r"(?u)\b\w[\w\+\-\#\.]+\b", text.lower())
            if not words:
                continue

            lemmas = []
            for w in words:
                if any("а" <= c <= "я" or c == "ё" for c in w):
                    try:
                        lemmas.append(self._morph.parse(w)[0].normal_form)
                    except Exception:
                        lemmas.append(w)
                else:
                    lemmas.append(w)

            for n in range(1, 4):
                for i in range(len(lemmas) - n + 1):
                    ngram_words = lemmas[i : i + n]
                    ngram = " ".join(ngram_words)
                    if n == 1 and ngram not in whitelist_words:
                        continue
                    if n > 1 and ngram not in whitelist_phrases:
                        continue
                    if any(lemma in self._stop_lemmas for lemma in ngram_words):
                        continue
                    norm = SkillNormalizer.normalize(ngram)
                    if norm and norm in whitelist:
                        all_ngrams.add(norm)
                        tokenized_corpus.append([norm])

        if not tokenized_corpus:
            logger.warning("Нет валидных n-грамм для BM25")
            return {}

        # дедупликация и ограничение
        unique_docs = []
        seen = set()
        for doc in tokenized_corpus:
            k = tuple(doc)
            if k not in seen:
                seen.add(k)
                unique_docs.append(doc)

        total = len(vacancies)
        max_docs = max(200, min(2000, total // 10))
        if len(unique_docs) > max_docs:
            unique_docs = [unique_docs[i] for i in np.argsort([len(d) for d in unique_docs])[-max_docs:]]

        bm25 = BM25Okapi(unique_docs)
        weights = {}
        for term in all_ngrams:
            try:
                scores = bm25.get_scores([term])
                if len(scores) == 0:
                    continue
                avg = float(sum(scores) / len(scores))
                if avg > config.BM25_MIN_SCORE:
                    weights[term] = round(avg, 4)
            except ZeroDivisionError:
                continue

        logger.info(f"BM25 готов: {len(weights)} навыков")
        self._cached_corpus = weights
        self._corpus_hash = ch
        return weights
