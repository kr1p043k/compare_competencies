"""
Парсер вакансий с поддержкой как старых dict, так и новых типизированных моделей.
"""

import json
import re
from collections import Counter
from typing import List, Dict, Any, Union
import logging
import pandas as pd
import pymorphy3
import torch
from rank_bm25 import BM25Okapi
import numpy as np   # если ещё нет
from src import config
from src.parsing.utils import load_it_skills, filter_skills_by_whitelist
from src.models.vacancy import Vacancy
from src.parsing.skill_parser import SkillParser, SkillSource
from src.parsing.skill_validator import SkillValidator
from src.parsing.skill_normalizer import SkillNormalizer
from src.parsing.embedding_loader import get_embedding_model

logger = logging.getLogger(__name__)


class VacancyParser:
    """
    Парсер вакансий - совместим с обоими форматами (dict и Vacancy объекты).
    Теперь использует новые парсер, валидатор и нормализатор.
    Исправлен подсчёт частот навыков (считает вхождения, а не уникальные навыки).
    """

    def __init__(self):
        self.skill_parser = SkillParser()
        self.skill_validator = SkillValidator(
            whitelist=load_it_skills()
        )

        # === ЗАГРУЗКА МОДЕЛИ ЭМБЕДДИНГОВ (один раз при создании парсера) ===
        logger.info(f"🚀 Загрузка модели эмбеддингов: {config.EMBEDDING_MODEL}")
        try:
            self.embedding_model = get_embedding_model()
            self.embedding_model.eval()   # режим inference
            logger.info("✅ Модель эмбеддингов успешно загружена")
        except Exception as e:
            logger.error(f"❌ Не удалось загрузить модель эмбеддингов: {e}")
            self.embedding_model = None   # чтобы не падало дальше
    def _get_skill_embeddings(self, skills: List[str]) -> Dict[str, List[float]]:
        """Генерирует эмбеддинги для списка навыков с кэшированием."""
        if not skills:
            return {}

        cache_file = config.EMBEDDINGS_CACHE_DIR / "skill_embeddings.json"
        
        # Пытаемся загрузить кэш
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                logger.info(f"Кэш эмбеддингов загружен ({len(cache)} навыков)")
                # Возвращаем только нужные
                return {s: cache[s] for s in skills if s in cache}
            except Exception:
                pass

        # Вычисляем новые эмбеддинги
        logger.info(f"Вычисление эмбеддингов для {len(skills)} навыков...")
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                skills, 
                convert_to_numpy=True, 
                show_progress_bar=True,
                batch_size=32
            )

        # Сохраняем в кэш
        cache = {skill: emb.tolist() for skill, emb in zip(skills, embeddings)}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        logger.info("Эмбеддинги успешно вычислены и закэшированы")
        return cache
    # =========================================================================
    # СОХРАНЕНИЕ
    # =========================================================================

    def save_raw_vacancies(
        self,
        vacancies: Union[List[Dict], List[Vacancy]],
        filename: str = "hh_vacancies.json"
    ):
        """Сохраняет вакансии в JSON (работает с dict и Vacancy)"""
        filepath = config.DATA_RAW_DIR / filename
        
        data_to_save = []
        for vac in vacancies:
            if isinstance(vac, Vacancy):
                data_to_save.append(vac.raw_data)
            else:
                data_to_save.append(vac)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Сырые данные сохранены в {filepath} (вакансий: {len(vacancies)})")

    def save_processed_frequencies(
        self,
        frequencies: Dict[str, int],
        filename: str = "competency_frequency.json",
        apply_filter: bool = True
    ):
        """Сохраняет частоты навыков в JSON"""
        if apply_filter:
            whitelist = load_it_skills()
            if whitelist:
                frequencies = filter_skills_by_whitelist(frequencies, whitelist)
                logger.info(f"Фильтрация применена, осталось {len(frequencies)} навыков")

        filepath = config.DATA_PROCESSED_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Частоты навыков сохранены в {filepath} (навыков: {len(frequencies)})")

    # =========================================================================
    # ИЗВЛЕЧЕНИЕ НАВЫКОВ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    # =========================================================================
    def extract_skills_from_description(self, description: str) -> List[str]:
        """Извлекает навыки ТОЛЬКО из описания (для старого utils.py)"""
        if not description:
            return []
        extracted = self.skill_parser._extract_from_text(
            description, source=SkillSource.DESCRIPTION
        )
        return [skill.text for skill in extracted]
    
    def extract_skills_from_vacancies(
        self, vacancies: Union[List[Dict], List[Vacancy]]
    ) -> Dict[str, Any]:
        """
        Исправленная версия: частоты считаются корректно (по количеству вакансий,
        где навык встречается хотя бы раз).
        """
        vacancy_objects = []
        for vac in vacancies:
            if isinstance(vac, dict):
                try:
                    vacancy_objects.append(Vacancy.from_api(vac))
                except ValueError:
                    continue
            else:
                vacancy_objects.append(vac)

        # === Правильный подсчёт частот ===
        skill_freq = Counter()
        all_extracted_for_embeddings = []   # для эмбеддингов и валидации

        for vacancy in vacancy_objects:
            extracted = self.skill_parser.parse_vacancy(vacancy)
            
            # Нормализуем и убираем дубли ТОЛЬКО внутри одной вакансии
            skill_texts = [s.text for s in extracted if s.text]
            normalized_per_vac = SkillNormalizer.normalize_batch(skill_texts)  # или лучше отдельно normalize + unique
            
            # Убираем дубли внутри вакансии (сохраняем порядок)
            unique_per_vac = list(dict.fromkeys([s for s in normalized_per_vac if s]))
            
            for skill in unique_per_vac:
                skill_freq[skill] += 1                    # ← +1 за каждую вакансию, где навык есть
                all_extracted_for_embeddings.append(skill)

        logger.info(f"Парсинг завершён: {self.skill_parser.get_stats()}")
        logger.info(f"Найдено уникальных навыков: {len(skill_freq)} | Сумма частот: {sum(skill_freq.values())}")

        # Валидация
        valid_skills, validation_results = self.skill_validator.validate_batch(list(skill_freq.keys()))
        
        # Фильтруем частоты только валидными навыками
        final_freq = {skill: skill_freq[skill] for skill in valid_skills if skill in skill_freq}

        # BM25 (оставляем как было)
        hybrid_weights = self._calculate_hybrid_weights(vacancies)

        # Эмбеддинги только для валидных уникальных навыков
        unique_valid_skills = list(final_freq.keys())
        skill_embeddings = self._get_skill_embeddings(unique_valid_skills)

        logger.info(f"Итого после валидации: {len(final_freq)} навыков")

        return {
            "frequencies": final_freq,           # ← теперь будут нормальные частоты (например python: 412)
            "hybrid_weights": hybrid_weights,
            "skill_embeddings": skill_embeddings
        }
        
    def _calculate_bm25_weights(self, vacancies: List) -> Dict[str, float]:
        """BM25 на n-граммах (1-3 слова) с фильтрацией по белому списку IT-навыков."""
        morph = pymorphy3.MorphAnalyzer()

        # Загружаем белый список IT-навыков (нормализованный)
        whitelist_raw = load_it_skills()
        whitelist = set()
        for skill in whitelist_raw:
            norm = SkillNormalizer.normalize(skill)
            if norm:
                whitelist.add(norm)

        # Стоп-леммы (предлоги, союзы, местоимения, общие слова)
        STOP_LEMMAS = {
            "в", "без", "до", "из", "к", "на", "по", "о", "от", "перед", "при",
            "через", "с", "у", "за", "над", "об", "под", "про", "для", "и",
            "да", "или", "либо", "не", "ни", "как", "так", "то", "что", "чтобы",
            "если", "хотя", "пока", "когда", "где", "который", "этот", "тот",
            "мой", "твой", "свой", "наш", "ваш", "весь", "всякий", "любой",
            "человек", "год", "раз", "дело", "жизнь", "день", "время", "работа",
            "сила", "рука", "слово", "место", "часть", "город", "страна",
            "опыт", "знание", "умение", "владение", "навык", "разработка",
            "программирование", "язык", "технология", "система", "решение",
            "задача", "проект", "команда", "компания", "клиент", "сервер",
            "поддержка", "сопровождение", "настройка", "обеспечение", "анализ",
            "тестирование", "отладка", "документация", "обучение", "мониторинг",
            "управление", "процесс", "функция", "модуль", "архитектура",
            "инфраструктура", "платформа", "среда", "код", "данные",
            "алгоритм", "модель", "метод", "подход", "практика", "стандарт",
            "версия", "релиз", "сборка", "деплой", "интеграция", "миграция",
            "контроль", "планирование", "оценка", "риск", "качество",
            "производительность", "масштабирование", "безопасность",
            "сеть", "база", "хранилище", "облако", "кластер", "контейнер",
            "виртуализация", "оркестрация", "автоматизация", "интерфейс",
            "пользователь", "администратор", "разработчик", "специалист",
            "инженер", "аналитик", "менеджер", "руководитель",
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "person", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us", "is", "was", "are", "been", "has",
            "had", "were", "said", "did", "get", "may", "am"
        }

        tokenized_corpus = []
        all_ngrams = set()

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

            words = re.findall(r'(?u)\b\w[\w\+\-\#\.]+\b', text.lower())
            if len(words) < 1:
                continue

            # Лемматизация русских слов
            lemmas = []
            for w in words:
                if any('а' <= c <= 'я' or c == 'ё' for c in w):
                    try:
                        lemmas.append(morph.parse(w)[0].normal_form)
                    except:
                        lemmas.append(w)
                else:
                    lemmas.append(w)

            # Генерация n-грамм (1-3 слова)
            for n in range(1, 4):
                for i in range(len(lemmas) - n + 1):
                    ngram = " ".join(lemmas[i:i+n])
                    # Проверка на стоп-леммы
                    if any(lemma in STOP_LEMMAS for lemma in lemmas[i:i+n]):
                        continue
                    norm = SkillNormalizer.normalize(ngram)
                    # Оставляем только n-граммы из белого списка
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

        bm25 = BM25Okapi(unique_docs)

        weights = {}
        logger.info(f"Вычисление BM25 для {len(all_ngrams)} n-грамм...")

        for term in all_ngrams:
            try:
                scores = bm25.get_scores([term])
                if len(scores) == 0:
                    continue
                avg_score = float(sum(scores) / len(scores))
                if avg_score > 0.005:   # умеренный порог
                    weights[term] = round(avg_score, 4)
            except ZeroDivisionError:
                continue

        logger.info(f"✅ BM25 рассчитан: {len(weights)} значимых n-грамм")
        return weights
            
    def _calculate_hybrid_weights(self, vacancies: List) -> Dict[str, float]:
        """
        ГИБРИДНЫЙ РАСЧЁТ ВЕСОВ:
        BM25 (lexical) + Sentence-Transformer embeddings (semantic centrality)
        """
        # 1. Сначала считаем чистый BM25
        bm25_weights = self._calculate_bm25_weights(vacancies)
        if not bm25_weights:
            logger.warning("BM25 вернул пустой результат → возвращаем fallback")
            return {}

        # 2. Получаем эмбеддинги только для навыков, которые есть в BM25
        unique_skills = list(bm25_weights.keys())
        skill_embeddings_dict = self._get_skill_embeddings(unique_skills)

        if len(skill_embeddings_dict) < 10:
            logger.warning("Слишком мало эмбеддингов → возвращаем только BM25")
            return bm25_weights

        # 3. Подготовка матрицы эмбеддингов
        skill_list = list(skill_embeddings_dict.keys())
        emb_list = [skill_embeddings_dict[s] for s in skill_list]
        embeddings = torch.tensor(emb_list, dtype=torch.float32)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2-нормализация

        # 4. Матрица косинусных сходств (O(n²) — для 2000 навыков ~0.1 сек)
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        semantic_centrality = sim_matrix.mean(dim=1).cpu().numpy()  # средняя схожесть с остальными

        # 5. Нормализация обоих компонентов в [0, 1]
        bm25_vals = np.array([bm25_weights.get(s, 0.0) for s in skill_list])
        bm25_norm = (bm25_vals - bm25_vals.min()) / (bm25_vals.max() - bm25_vals.min() + 1e-8)

        semantic_norm = (semantic_centrality - semantic_centrality.min()) / (
            semantic_centrality.max() - semantic_centrality.min() + 1e-8
        )

        # 6. ГИБРИДНЫЙ ВЕС (настраиваемые коэффициенты)
        alpha = 0.65   # вес BM25
        beta = 0.35    # вес семантики 

        hybrid_weights = {}
        for i, skill in enumerate(skill_list):
            hybrid_score = alpha * bm25_norm[i] + beta * semantic_norm[i]
            hybrid_weights[skill] = round(float(hybrid_score), 4)

        logger.info(f"✅ ГИБРИД BM25 + Embeddings готов: {len(hybrid_weights)} навыков "
                    f"(α={alpha}, β={beta})")

        return hybrid_weights

    # Добавь этот статический метод в класс VacancyParser (рядом с clean_highlighttext)
    @staticmethod
    def _strip_html(text: str) -> str:
        """Полная очистка HTML-тегов из описания вакансии"""
        if not text:
            return ""
        # Удаляем все теги
        text = re.sub(r'<[^>]+>', ' ', text)
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # =========================================================================
    # СТАРЫЕ МЕТОДЫ (для обратной совместимости)
    # =========================================================================

    @staticmethod
    def clean_highlighttext(text: str) -> str:
        """Удаляет теги <highlighttext> из hh.ru"""
        if not text:
            return ""
        text = re.sub(r'</?highlighttext[^>]*>', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def extract_skills(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Старый метод - извлекает ключевые навыки (для совместимости)"""
        all_skills = []
        vacancies_with_skills = 0

        for vacancy in vacancies:
            skills_data = vacancy.get('key_skills', [])
            if skills_data:
                vacancies_with_skills += 1
                for skill_item in skills_data:
                    skill_name = skill_item.get('name', '')
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
        normalized = re.sub(r'^(опыт работы с|работа с|знание|владение|умение|должен|требуется|навык|навыки|умение работать с|опыт)\s+', '', normalized)
        normalized = re.sub(r'\s+(опыт|знание|владение|умение|плюсом|желательно|преимуществом|навык|навыки)$', '', normalized)

        if len(normalized.split()) > 4:
            return ""

        if normalized in {"инициатива", "мотивация", "коммуникация", "клиентами", "клиентам", "харизма", "многозадачность"}:
            return ""

        return normalized.strip()

    @staticmethod
    def is_valid_skill(skill: str) -> bool:
        """Старый метод - проверка валидности (для совместимости)"""
        if not skill or len(skill) < 3:
            return False
        return True

    @staticmethod
    def count_skills(skills_list: List[str]) -> Dict[str, int]:
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
    def extract_skills_from_text(vacancies: List[Dict[str, Any]]) -> List[str]:
        """Старый метод - извлечение из текста (для совместимости)"""
        logger.warning("extract_skills_from_text deprecated, используйте extract_skills_from_vacancies")
        return []

    # =========================================================================
    # EXCEL
    # =========================================================================

    def aggregate_to_dataframe(self, vacancies: Union[List[Dict], List[Vacancy]]) -> pd.DataFrame:
        """Агрегирует данные в DataFrame для Excel"""
        rows = []

        for vac in vacancies:
            if isinstance(vac, Vacancy):
                row = {
                    'Вакансия': vac.name,
                    'Компания': vac.employer.name,
                    'Регион': vac.area.name,
                    'ID': vac.id,
                    'Зарплата': str(vac.salary) if vac.salary else 'Не указана',
                    'Навыков': len(vac.key_skills),
                    'Навыки': ', '.join(vac.get_skill_names())
                }
            else:
                vac_name = vac.get('name', 'Unknown')
                employer_name = vac.get('employer', {}).get('name', 'Unknown')
                area_name = vac.get('area', {}).get('name', 'Unknown')
                skills = [s['name'] for s in vac.get('key_skills', [])]

                row = {
                    'Вакансия': vac_name,
                    'Компания': employer_name,
                    'Регион': area_name,
                    'ID': vac.get('id'),
                    'Зарплата': 'Не указана',
                    'Навыков': len(skills),
                    'Навыки': ', '.join(skills)
                }
            
            rows.append(row)

        return pd.DataFrame(rows)

    def save_to_excel(self, df: pd.DataFrame, filename: str):
        """Сохраняет DataFrame в Excel"""
        filepath = config.DATA_PROCESSED_DIR / filename
        df.to_excel(filepath, index=False, engine='openpyxl')
        logger.info(f"Excel файл сохранён в {filepath}")

    def print_vacancies_list(self, vacancies: Union[List[Dict], List[Vacancy]]):
        """Выводит список вакансий"""
        for i, vac in enumerate(vacancies[:20], 1):
            if isinstance(vac, Vacancy):
                print(f"{i}. {vac.name} @ {vac.employer.name} ({vac.area.name})")
                if vac.key_skills:
                    print(f"   Навыки: {', '.join(vac.get_skill_names()[:5])}")
            else:
                vac_name = vac.get('name', 'Unknown')
                employer = vac.get('employer', {}).get('name', 'Unknown')
                area = vac.get('area', {}).get('name', 'Unknown')
                print(f"{i}. {vac_name} @ {employer} ({area})")