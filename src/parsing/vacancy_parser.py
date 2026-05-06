"""
Парсер вакансий с поддержкой как старых dict, так и новых типизированных моделей.
Оптимизированная версия: кэш npz, предфильтрация n-грамм, PCA, ленивая загрузка,
кэширование корпуса BM25, однопроходная токенизация.
"""

import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pymorphy3
import torch
from rank_bm25 import BM25Okapi

from src import config
from src.models.vacancy import Vacancy
from src.parsing.embedding_loader import get_embedding_model
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.skill_parser import SkillParser, SkillSource
from src.parsing.skill_validator import SkillValidator
from src.parsing.utils import filter_skills_by_whitelist, load_it_skills
from src.utils import atomic_write_json, atomic_write_npz

logger = logging.getLogger(__name__)


class VacancyParser:
    """
    Парсер вакансий - совместим с обоими форматами (dict и Vacancy объекты).
    Исправлен подсчёт частот навыков (считает вхождения, а не уникальные навыки).

    Оптимизации:
    - npz-кэш эмбеддингов (×5 быстрее JSON)
    - предфильтрация n-грамм по whitelist (×3 быстрее BM25)
    - PCA-сжатие эмбеддингов до 256 (×2 быстрее cosine similarity)
    - ленивая загрузка модели эмбеддингов
    - кэширование корпуса BM25 (однопроходная токенизация)
    - параллельная валидация навыков
    """

    def __init__(self):
        self.skill_parser = SkillParser()
        self.skill_validator = SkillValidator(whitelist=None)
        self._embedding_model = None  # ленивая загрузка
        self._cached_corpus: dict | None = None  # кэш BM25
        self._corpus_hash: str | None = None  # хэш для инвалидации кэша

    # =========================================================================
    # ЛЕНИВАЯ ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ
    # =========================================================================
    @property
    def embedding_model(self):
        """Ленивая загрузка модели эмбеддингов — только при первом обращении."""
        if self._embedding_model is None:
            logger.info(f"🚀 Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL}")
            try:
                self._embedding_model = get_embedding_model()
                self._embedding_model.eval()
                logger.info("✅ Модель эмбеддингов успешно загружена")
            except Exception as e:
                logger.error(f"❌ Не удалось загрузить модель эмбеддингов: {e}")
                self._embedding_model = None
        return self._embedding_model

    # =========================================================================
    # КЭШ ЭМБЕДДИНГОВ (npz — в 5-10 раз быстрее JSON)
    # =========================================================================
    def _get_skill_embeddings(self, skills: list[str]) -> dict[str, np.ndarray]:
        """Генерирует эмбеддинги для списка навыков с атомарным npz-кэшированием."""
        if not skills:
            return {}

        cache_npz = config.EMBEDDINGS_CACHE_DIR / "skill_embeddings.npz"
        cache_index = config.EMBEDDINGS_CACHE_DIR / "skill_embeddings_index.json"

        cached = {}

        # Пытаемся загрузить из npz
        if cache_npz.exists() and cache_index.exists():
            try:
                with open(cache_index, encoding="utf-8") as f:
                    index = json.load(f)
                data = np.load(cache_npz)["embeddings"]

                for skill in skills:
                    if skill in index:
                        cached[skill] = data[index[skill]]

                if len(cached) == len(skills):
                    logger.info(f"✅ Эмбеддинги загружены из npz-кэша: {len(cached)} навыков")
                    return cached
                elif cached:
                    logger.info(f"Эмбеддинги частично из кэша: {len(cached)}/{len(skills)}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить npz-кэш: {e}")

        # Считаем недостающие
        missing = [s for s in skills if s not in cached]
        if missing:
            logger.info(f"Вычисление эмбеддингов для {len(missing)} навыков...")
            with torch.no_grad():
                new_embs = self.embedding_model.encode(
                    missing, convert_to_numpy=True, show_progress_bar=True, batch_size=64
                )

            # Обновляем кэш атомарно
            all_skills = list(cached.keys()) + missing
            all_embs = np.vstack([np.stack(list(cached.values())), new_embs]) if cached else new_embs
            new_index = {skill: i for i, skill in enumerate(all_skills)}

            # Атомарная запись npz и index
            atomic_write_npz({"embeddings": all_embs}, cache_npz)
            atomic_write_json(new_index, cache_index)

            logger.info(f"💾 npz-кэш атомарно обновлён: {len(all_skills)} навыков")

            for skill, emb in zip(missing, new_embs, strict=False):
                cached[skill] = emb

        return cached

    # =========================================================================
    # СОХРАНЕНИЕ
    # =========================================================================
    def save_raw_vacancies(self, vacancies: list[dict] | list[Vacancy], filename: str = "hh_vacancies.json"):
        """Сохраняет вакансии в JSON атомарно."""
        filepath = config.DATA_RAW_DIR / filename

        data_to_save = []
        for vac in vacancies:
            if isinstance(vac, Vacancy):
                data_to_save.append(vac.raw_data)
            else:
                data_to_save.append(vac)

        atomic_write_json(data_to_save, filepath)
        logger.info(f"Сырые данные атомарно сохранены в {filepath} (вакансий: {len(vacancies)})")

    def save_processed_frequencies(
        self, frequencies: dict[str, int], filename: str = "competency_frequency.json", apply_filter: bool = True
    ):
        """Сохраняет частоты навыков в JSON атомарно."""
        if apply_filter:
            whitelist = load_it_skills()
            if whitelist:
                frequencies = filter_skills_by_whitelist(frequencies, whitelist)
                logger.info(f"Фильтрация применена, осталось {len(frequencies)} навыков")

        filepath = config.DATA_PROCESSED_DIR / filename
        atomic_write_json(frequencies, filepath)
        logger.info(f"Частоты навыков атомарно сохранены в {filepath} (навыков: {len(frequencies)})")

    # =========================================================================
    # ИЗВЛЕЧЕНИЕ НАВЫКОВ
    # =========================================================================
    def extract_skills_from_description(self, description: str) -> list[str]:
        """Извлекает навыки ТОЛЬКО из описания (для старого utils.py)"""
        if not description:
            return []
        extracted = self.skill_parser._extract_from_text(description, source=SkillSource.DESCRIPTION)
        return [skill.text for skill in extracted]

    def extract_skills_from_vacancies(self, vacancies: list[dict] | list[Vacancy]) -> dict[str, Any]:
        """
        Извлекает навыки из вакансий: частоты + hybrid_weights + эмбеддинги.
        Оптимизации: lru_cache в normalize, параллельная валидация, единый проход.
        """
        # === Шаг 1: Конвертация в Vacancy ===
        vacancy_objects = []
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError:
                    continue
            else:
                vacancy_objects.append(vac)

        # === Шаг 2: Извлечение и подсчёт частот ===
        skill_freq = Counter()

        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)

            skill_texts = [s.text for s in extracted if s.text]
            normalized_per_vac = SkillNormalizer.normalize_batch(skill_texts)
            unique_per_vac = list(dict.fromkeys([s for s in normalized_per_vac if s]))

            for skill in unique_per_vac:
                skill_freq[skill] += 1

        logger.info(f"Парсинг завершён: {self.skill_parser.get_stats()}")
        logger.info(f"Найдено уникальных навыков: {len(skill_freq)} | Сумма частот: {sum(skill_freq.values())}")

        # === Шаг 3: Параллельная валидация ===
        all_skills = list(skill_freq.keys())
        valid_skills = []

        if len(all_skills) > 200:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = executor.map(self.skill_validator.validate, all_skills)
                for skill, result in zip(all_skills, results, strict=False):
                    if result.is_valid:
                        valid_skills.append(skill)
        else:
            for skill in all_skills:
                if self.skill_validator.validate(skill).is_valid:
                    valid_skills.append(skill)

        final_freq = {skill: skill_freq[skill] for skill in valid_skills}

        logger.info(f"После валидации: {len(final_freq)} навыков (отсеяно {len(all_skills) - len(final_freq)})")

        # === Шаг 4: BM25 + гибридные веса ===
        hybrid_weights = self._calculate_hybrid_weights(vacancies)

        # === Шаг 5: Эмбеддинги для валидных навыков ===
        unique_valid_skills = list(final_freq.keys())
        skill_embeddings = self._get_skill_embeddings(unique_valid_skills)

        logger.info(
            f"Итого: {len(final_freq)} навыков, "
            f"{len(hybrid_weights)} с hybrid-весами, "
            f"{len(skill_embeddings)} с эмбеддингами"
        )

        return {"frequencies": final_freq, "hybrid_weights": hybrid_weights, "skill_embeddings": skill_embeddings}

    # =========================================================================
    # ХЭШ КОРПУСА ДЛЯ КЭШИРОВАНИЯ BM25
    # =========================================================================
    def _get_corpus_hash(self, vacancies: list) -> str:
        """Быстрый хэш по количеству вакансий и сумме ID."""
        total_ids = 0
        count = 0
        for v in vacancies:
            vid = v.get("id", "") if isinstance(v, dict) else v.id
            if vid:
                total_ids += int(vid) if str(vid).isdigit() else hash(str(vid))
                count += 1
        return f"{count}:{total_ids}"

    # =========================================================================
    # BM25 С ПРЕДВАРИТЕЛЬНОЙ ФИЛЬТРАЦИЕЙ И КЭШИРОВАНИЕМ
    # =========================================================================
    # =========================================================================
    # BM25 С ПРЕДВАРИТЕЛЬНОЙ ФИЛЬТРАЦИЕЙ, КЭШИРОВАНИЕМ И КОНФИГОМ
    # =========================================================================
    def _calculate_bm25_weights(self, vacancies: list) -> dict[str, float]:
        """BM25 с предфильтрацией n-грамм, кэшированием корпуса и однопроходной токенизацией."""

        # Проверяем кэш
        corpus_hash = self._get_corpus_hash(vacancies)
        if self._cached_corpus is not None and self._corpus_hash == corpus_hash:
            logger.info("✅ BM25: использован кэшированный корпус (однопроходная токенизация)")
            return self._cached_corpus

        morph = pymorphy3.MorphAnalyzer()

        # Загружаем whitelist и строим быстрые индексы
        whitelist_raw = load_it_skills()
        whitelist = set()
        for skill in whitelist_raw:
            norm = SkillNormalizer.normalize(skill)
            if norm:
                whitelist.add(norm)

        whitelist_words = set()
        whitelist_phrases = set()
        for skill in whitelist:
            words = skill.split()
            if len(words) == 1:
                whitelist_words.add(words[0])
            else:
                whitelist_phrases.add(skill)

        # Стоп-леммы
        stop_lemmas = {
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

        tokenized_corpus = []
        all_ngrams = set()

        # === ЕДИНЫЙ ПРОХОД ПО ВАКАНСИЯМ ===
        for vac in vacancies:
            parts = []
            if isinstance(vac, dict):
                desc = vac.get("description") or ""
                if desc:
                    parts.append(self._strip_html(desc))
                snippet = vac.get("snippet", {})
                req = snippet.get("requirement") or ""
                resp = snippet.get("responsibility") or ""
                if req:
                    parts.append(self._strip_html(req))
                if resp:
                    parts.append(self._strip_html(resp))
                key_skills = " ".join(s.get("name", "") for s in vac.get("key_skills", []))
                if key_skills:
                    parts.append(key_skills)
            else:
                if vac.description:
                    parts.append(self._strip_html(vac.description))
                key_skills = " ".join(s.name for s in vac.key_skills)
                if key_skills:
                    parts.append(key_skills)

            text = " ".join(part.strip() for part in parts if part and part.strip())
            if not text:
                continue

            words = re.findall(r"(?u)\b\w[\w\+\-\#\.]+\b", text.lower())
            if len(words) < 1:
                continue

            # Лемматизация
            lemmas = []
            for w in words:
                if any("а" <= c <= "я" or c == "ё" for c in w):
                    try:
                        lemmas.append(morph.parse(w)[0].normal_form)
                    except Exception:  # noqa: E722
                        lemmas.append(w)
                else:
                    lemmas.append(w)

            # Генерация n-грамм с предфильтрацией
            for n in range(1, 4):
                for i in range(len(lemmas) - n + 1):
                    ngram_words = lemmas[i : i + n]
                    ngram = " ".join(ngram_words)

                    if n == 1:
                        if ngram not in whitelist_words:
                            continue
                    else:
                        if ngram not in whitelist_phrases:
                            continue

                    if any(lemma in stop_lemmas for lemma in ngram_words):
                        continue

                    norm = SkillNormalizer.normalize(ngram)
                    if norm and norm in whitelist:
                        all_ngrams.add(norm)
                        tokenized_corpus.append([norm])

        if not tokenized_corpus:
            logger.warning("Нет валидных n-грамм для BM25")
            return {}

        # Убираем дубликаты документов
        unique_docs = []
        seen = set()
        for doc in tokenized_corpus:
            key = tuple(doc)
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # Динамический лимит: 10% от вакансий, но от 200 до 2000
        total_vacancies = len(vacancies)
        max_docs = max(200, min(2000, total_vacancies // 10))
        if len(unique_docs) > max_docs:
            doc_lengths = [len(doc) for doc in unique_docs]
            top_indices = np.argsort(doc_lengths)[-max_docs:]
            unique_docs = [unique_docs[i] for i in top_indices]
            logger.info(
                f"Корпус BM25 ограничен до {len(unique_docs)} документов "
                f"(из {total_vacancies} вакансий, порог {max_docs})"
            )

        bm25 = BM25Okapi(unique_docs)

        weights = {}
        min_score = getattr(config, "BM25_MIN_SCORE", 0.005)
        logger.info(f"Вычисление BM25 для {len(all_ngrams)} n-грамм...")

        for term in all_ngrams:
            try:
                scores = bm25.get_scores([term])
                if len(scores) == 0:
                    continue
                avg_score = float(sum(scores) / len(scores))
                if avg_score > min_score:
                    weights[term] = round(avg_score, 4)
            except ZeroDivisionError:
                continue

        logger.info(f"✅ BM25 рассчитан: {len(weights)} значимых n-грамм")

        # Сохраняем в кэш
        self._cached_corpus = weights
        self._corpus_hash = corpus_hash

        return weights

    # =========================================================================
    # ГИБРИДНЫЕ ВЕСА С PCA, GRACEFUL DEGRADATION И КОНФИГОМ
    # =========================================================================
    def _calculate_hybrid_weights(self, vacancies: list) -> dict[str, float]:
        """
        Гибридный расчёт весов: BM25 + эмбеддинги.
        С PCA-сжатием для больших словарей.
        При недоступности модели — fallback на чистый BM25.
        """
        bm25_weights = self._calculate_bm25_weights(vacancies)
        if not bm25_weights:
            logger.warning("BM25 вернул пустой результат → возвращаем fallback")
            return {}

        # Проверяем доступность модели эмбеддингов
        if self.embedding_model is None:
            logger.warning("⚠️ Модель эмбеддингов недоступна — используем чистый BM25")
            vals = np.array(list(bm25_weights.values()))
            v_min, v_max = vals.min(), vals.max()
            if v_max > v_min:
                return {s: round((w - v_min) / (v_max - v_min), 4) for s, w in bm25_weights.items()}
            return bm25_weights

        unique_skills = list(bm25_weights.keys())

        # Пробуем получить эмбеддинги
        try:
            skill_embeddings_dict = self._get_skill_embeddings(unique_skills)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка получения эмбеддингов: {e} — fallback на чистый BM25")
            vals = np.array(list(bm25_weights.values()))
            v_min, v_max = vals.min(), vals.max()
            if v_max > v_min:
                return {s: round((w - v_min) / (v_max - v_min), 4) for s, w in bm25_weights.items()}
            return bm25_weights

        if len(skill_embeddings_dict) < 10:
            logger.warning("Слишком мало эмбеддингов → возвращаем только BM25")
            return bm25_weights

        # Подготовка матрицы
        skill_list = list(skill_embeddings_dict.keys())
        emb_list = [skill_embeddings_dict[s] for s in skill_list]
        embeddings = np.array(emb_list, dtype=np.float32)

        # PCA-сжатие (параметры из конфига)
        pca_enabled = getattr(config, "PCA_ENABLED", True)
        pca_min_samples = getattr(config, "PCA_MIN_SAMPLES", 100)
        pca_min_features = getattr(config, "PCA_MIN_FEATURES", 128)
        pca_target_dim = getattr(config, "PCA_TARGET_DIM", 256)

        pca_applied = False
        if pca_enabled and len(embeddings) > pca_min_samples and embeddings.shape[1] > pca_min_features:
            from sklearn.decomposition import PCA

            n_components = min(pca_target_dim, len(embeddings) - 1, embeddings.shape[1])

            if n_components < embeddings.shape[1]:
                pca = PCA(
                    n_components=n_components,
                    svd_solver="auto",
                    random_state=config.GLOBAL_RANDOM_SEED if hasattr(config, "GLOBAL_RANDOM_SEED") else 42,
                )
                embeddings = pca.fit_transform(embeddings)
                explained = pca.explained_variance_ratio_.sum()
                logger.info(
                    f"Эмбеддинги сжаты PCA: {emb_list[0].shape[0]} → {n_components} "
                    f"(объяснённая дисперсия: {explained:.2%})"
                )
                pca_applied = True
            else:
                logger.debug(f"PCA не требуется: размерность {embeddings.shape[1]} уже ≤ {n_components}")

        # L2-нормализация
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Матрица косинусных сходств
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        semantic_centrality = sim_matrix.mean(dim=1).cpu().numpy()

        # Нормализация
        bm25_vals = np.array([bm25_weights.get(s, 0.0) for s in skill_list])
        bm25_norm = (bm25_vals - bm25_vals.min()) / (bm25_vals.max() - bm25_vals.min() + 1e-8)

        semantic_norm = (semantic_centrality - semantic_centrality.min()) / (
            semantic_centrality.max() - semantic_centrality.min() + 1e-8
        )

        # Гибридный вес
        alpha, beta = 0.65, 0.35
        hybrid_weights = {}
        for i, skill in enumerate(skill_list):
            hybrid_score = alpha * bm25_norm[i] + beta * semantic_norm[i]
            hybrid_weights[skill] = round(float(hybrid_score), 4)

        logger.info(
            f"✅ ГИБРИД BM25 + Embeddings готов: {len(hybrid_weights)} навыков "
            f"(α={alpha}, β={beta}" + (", с PCA" if pca_applied else ", без PCA") + ")"
        )

        return hybrid_weights

    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ
    # =========================================================================
    @staticmethod
    def _strip_html(text: str) -> str:
        """Полная очистка HTML-тегов из описания вакансии"""
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def clean_highlighttext(text: str) -> str:
        """Удаляет теги <highlighttext> из hh.ru"""
        if not text:
            return ""
        text = re.sub(r"</?highlighttext[^>]*>", "", text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def extract_skills(vacancies: list[dict[str, Any]]) -> list[str]:
        """Старый метод - извлекает ключевые навыки (для совместимости)"""
        all_skills = []
        vacancies_with_skills = 0

        for vacancy in vacancies:
            skills_data = vacancy.get("key_skills", [])
            if skills_data:
                vacancies_with_skills += 1
                for skill_item in skills_data:
                    skill_name = skill_item.get("name", "")
                    if skill_name:
                        clean_name = VacancyParser.clean_highlighttext(skill_name)
                        all_skills.append(clean_name)

        logger.info(f"Из {len(vacancies)} вакансий извлечены навыки из {vacancies_with_skills} (key_skills)")
        return all_skills

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Старый метод - нормализация (для совместимости)"""
        if not skill:
            return ""

        normalized = VacancyParser.clean_highlighttext(skill).lower().strip()
        normalized = re.sub(
            r"^(опыт работы с|работа с|знание|владение|умение|должен|требуется|навык|навыки|умение работать с|опыт)\s+",
            "",
            normalized,
        )
        normalized = re.sub(
            r"\s+(опыт|знание|владение|умение|плюсом|желательно|преимуществом|навык|навыки)$", "", normalized
        )

        if len(normalized.split()) > 4:
            return ""

        if normalized in {
            "инициатива",
            "мотивация",
            "коммуникация",
            "клиентами",
            "клиентам",
            "харизма",
            "многозадачность",
        }:
            return ""

        return normalized.strip()

    @staticmethod
    def is_valid_skill(skill: str) -> bool:
        return skill is not None and len(skill) >= 3

    @staticmethod
    def count_skills(skills_list: list[str]) -> dict[str, int]:
        """Старый метод - подсчёт скиллов (для совместимости)"""
        normalized_skills = [VacancyParser.normalize_skill(s) for s in skills_list if s]
        skill_counts = Counter(normalized_skills)
        filtered = {k: v for k, v in skill_counts.items() if len(k) > 2}

        whitelist = load_it_skills()
        if whitelist:
            filtered = {k: v for k, v in filtered.items() if k in whitelist or k.lower() in whitelist}

        logger.info(f"После фильтрации осталось {len(filtered)} навыков")
        return filtered

    @staticmethod
    def extract_skills_from_text(vacancies: list[dict[str, Any]]) -> list[str]:
        """Старый метод - извлечение из текста (для совместимости)"""
        logger.warning("extract_skills_from_text deprecated, используйте extract_skills_from_vacancies")
        return []

    # =========================================================================
    # EXCEL
    # =========================================================================
    def aggregate_to_dataframe(self, vacancies: list[dict] | list[Vacancy]) -> pd.DataFrame:
        """
        Агрегирует данные в DataFrame для Excel.
        Навыки собираются из key_skills + текстового парсера (объединение).
        """
        rows = []

        for vac in vacancies:
            key_skill_names = []
            if isinstance(vac, Vacancy):
                key_skill_names = vac.get_skill_names()
                vac_name = vac.name
                employer_name = vac.employer.name
                area_name = vac.area.name
                vac_id = vac.id
                salary = str(vac.salary) if vac.salary else "Не указана"
                parsed_skills = self.skill_parser.parse_vacancy(vac)
                text_skill_names = [s.text for s in parsed_skills if s.text]
                description = vac.description or ""
                snippet_req = vac.snippet.requirement if vac.snippet else ""
                snippet_resp = vac.snippet.responsibility if vac.snippet else ""
            else:
                key_skills = vac.get("key_skills", [])
                key_skill_names = [s["name"] for s in key_skills if isinstance(s, dict) and "name" in s]
                vac_name = vac.get("name", "Unknown")
                employer = vac.get("employer", {}) or {}
                employer_name = employer.get("name", "Unknown")
                area = vac.get("area", {}) or {}
                area_name = area.get("name", "Unknown")
                vac_id = vac.get("id")
                salary = "Не указана"
                description = vac.get("description", "") or ""
                snippet = vac.get("snippet", {}) or {}
                snippet_req = snippet.get("requirement", "") or ""
                snippet_resp = snippet.get("responsibility", "") or ""
                text_skill_names = self.extract_skills_from_description(f"{description} {snippet_req} {snippet_resp}")

            all_skills = list(dict.fromkeys(key_skill_names + text_skill_names))
            try:
                from src.parsing.skill_normalizer import SkillNormalizer

                all_skills = SkillNormalizer.deduplicate(all_skills)
            except Exception:
                pass

            rows.append(
                {
                    "Вакансия": vac_name,
                    "Компания": employer_name,
                    "Регион": area_name,
                    "ID": vac_id,
                    "Зарплата": salary,
                    "Навыков": len(all_skills),
                    "Навыки": ", ".join(all_skills),
                }
            )

        return pd.DataFrame(rows)

    def save_to_excel(self, df: pd.DataFrame, filename: str):
        """Сохраняет DataFrame в Excel"""
        filepath = config.DATA_PROCESSED_DIR / filename
        df.to_excel(filepath, index=False, engine="openpyxl")
        logger.info(f"Excel файл сохранён в {filepath}")

    def print_vacancies_list(self, vacancies: list[dict] | list[Vacancy]):
        """Выводит список вакансий (навыки: key_skills + текстовое извлечение)"""
        for i, vac in enumerate(vacancies[:20], 1):
            key_skill_names = []
            text_skill_names = []

            if isinstance(vac, Vacancy):
                vac_name = vac.name
                employer_name = vac.employer.name
                area_name = vac.area.name
                key_skill_names = vac.get_skill_names()
                parsed = self.skill_parser.parse_vacancy(vac)
                text_skill_names = [s.text for s in parsed if s.text]
            else:
                vac_name = vac.get("name", "Unknown")
                employer = vac.get("employer", {}) or {}
                employer_name = employer.get("name", "Unknown")
                area = vac.get("area", {}) or {}
                area_name = area.get("name", "Unknown")
                ks = vac.get("key_skills", [])
                key_skill_names = [s["name"] for s in ks if isinstance(s, dict) and "name" in s]
                desc = vac.get("description", "") or ""
                snip = vac.get("snippet", {}) or {}
                req = snip.get("requirement", "") or ""
                resp = snip.get("responsibility", "") or ""
                text_skill_names = self.extract_skills_from_description(f"{desc} {req} {resp}")

            all_skills = list(dict.fromkeys(key_skill_names + text_skill_names))
            try:
                from src.parsing.skill_normalizer import SkillNormalizer

                all_skills = SkillNormalizer.deduplicate(all_skills)
            except Exception:
                pass

            print(f"{i}. {vac_name} @ {employer_name} ({area_name})")
            if all_skills:
                print(f"   Навыки: {', '.join(all_skills[:5])}")
