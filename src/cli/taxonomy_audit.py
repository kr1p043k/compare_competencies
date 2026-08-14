"""Аудит и заполнение пробелов таксономии навыков.

Находит навыки из it_skills.json, у которых нет категории в skill_taxonomy.json,
и предлагает категорию по эмбеддинг-сходству с прототипом каждой категории.

Usage:
    python -m src.cli taxonomy-audit                 # показать некатегоризованные
    python -m src.cli taxonomy-audit --apply          # применить предложения в JSON
    python -m src.cli taxonomy-audit --threshold 0.55 --apply
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import structlog

from src import config

logger = structlog.get_logger(__name__)

TAXONOMY_PATH = config.SKILL_TAXONOMY_PATH
IT_SKILLS_PATH = config.IT_SKILLS_PATH
DEFAULT_THRESHOLD = 0.55
# Для авто-категоризации новых навыков (extend) — чуть ниже, т.к. nearest-neighbor
# даёт меньшие absolute scores, но при < 0.50 навык оставляем в "other" (неоднозначно).
EXTEND_THRESHOLD = 0.50

# Категории, куда не стоит автоматически относить навыки (пустые/служебные).
SKIP_CATEGORIES = {"business_tools", "methodologies", "abstract_concepts"}

# Ручной маппинг навык → категория (приоритетнее эмбеддинг-предложений).
# Составлен по смыслу: языки → programming_languages, фреймворки → frameworks и т.д.
MANUAL_OVERRIDES = {
    "1c": "enterprise",
    "1c предприятие": "enterprise",
    "a/b тестирование": "testing_qa",
    "airflow": "data_science",
    "apache spark java": "data_science",
    "apache spark python": "data_science",
    "api gateway": "devops",
    "arduino c++": "embedded",
    "asterisk": "devops",
    "asyncio": "frameworks",
    "backpropagation": "ml_advanced",
    "big data python": "data_science",
    "c++ embedded": "embedded",
    "c++ gpu": "ml_advanced",
    "c++ разработка": "programming_languages",
    "c/c++": "programming_languages",
    "chaos engineering": "devops",
    "cmake": "devops",
    "cpp для встроенных систем": "embedded",
    "cuda": "ml_advanced",
    "cuda c++": "ml_advanced",
    "data science python": "data_science",
    "deep learning basics": "ml_advanced",
    "deep learning research": "ml_advanced",
    "deep neural networks": "ml_advanced",
    "devops": "devops",
    "elm": "programming_languages",
    "embedded software": "embedded",
    "enterprise java": "enterprise",
    "event sourcing": "methodologies_concepts",
    "f1 мера": "ml_advanced",
    "feature preprocessing": "data_science",
    "feature store": "data_science",
    "federated learning": "ml_advanced",
    "feedforward neural network": "ml_advanced",
    "few-shot fine-tuning": "ml_advanced",
    "flink java": "data_science",
    "gradle": "frameworks",
    "hadoop java": "data_science",
    "hugo": "frameworks",
    "hyperledger": "frameworks",
    "java big data": "data_science",
    "java разработка": "programming_languages",
    "jhipster": "frameworks",
    "jupyter": "data_science",
    "knowledge graph": "ml_advanced",
    "llvm": "programming_languages",
    "mlp": "ml_advanced",
    "model serving": "ml_advanced",
    "nextauth": "frameworks",
    "observability": "devops",
    "odata": "frameworks",
    "opencl": "ml_advanced",
    "pine script": "programming_languages",
    "pyspark sql": "data_science",
    "python programming": "programming_languages",
    "python разработка": "programming_languages",
    "real-time failover": "devops",
    "relu": "ml_advanced",
    "research": "ml_advanced",
    "reverse proxy": "devops",
    "solidity": "programming_languages",
    "stl": "programming_languages",
    "stripe api": "frameworks",
    "svd": "mathematics",
    "threat hunting": "security",
    "threat intelligence": "security",
    "time series": "data_science",
    "transformer": "ml_advanced",
    "turborepo": "frameworks",
    "vector database": "databases",
    "vectordb": "databases",
    "webrtc": "frontend",
    "zapier": "frameworks",
    "zeromq": "frameworks",
    "автоматизация развёртывания": "devops",
    "автоматическое доказательство теорем": "mathematics",
    "автоматическое машинное обучение": "ml_advanced",
    "адаптация моделей": "ml_advanced",
    "анализ данных на python": "data_science",
    "базы знаний": "ml_advanced",
    "байесовская оптимизация гиперпараметров": "mathematics",
    "безопасность алгоритмов": "security",
    "вариационные автокодировщики": "ml_advanced",
    "веб-аналитика": "data_science",
    "верификация моделей": "ml_advanced",
    "видео-языковые модели": "llm_ai",
    "визуализация python": "data_science",
    "визуализация данных": "data_science",
    "встроенное по c++": "embedded",
    "вычисления на gpu": "ml_advanced",
    "генеративно-состязательные сети": "ml_advanced",
    "генерация изображений": "llm_ai",
    "глубокие нейронные сети": "ml_advanced",
    "градиентное скрытие": "ml_advanced",
    "графическое представление данных": "data_science",
    "графовые нейронные сети": "ml_advanced",
    "дашборды": "data_science",
    "дедуктивные системы": "mathematics",
    "диффузионные модели": "llm_ai",
    "защита моделей": "security",
    "извлечение информации": "data_science",
    "инженерия знаний": "ml_advanced",
    "инициализация весов": "ml_advanced",
    "инновационные методы ml": "ml_advanced",
    "инфраструктура бд": "databases",
    "инфраструктура данных": "data_science",
    "искусственный интеллект": "ml_advanced",
    "искусственный нейрон": "ml_advanced",
    "исследование dl": "ml_advanced",
    "исследование llm": "llm_ai",
    "исследования в ml": "ml_advanced",
    "классические алгоритмы машинного обучения": "ml_advanced",
    "кодирование категорий": "data_science",
    "контейнеризация": "devops",
    "контейнеризация python": "devops",
    "логическое программирование": "methodologies_concepts",
    "масштабирование признаков": "data_science",
    "машинное обучение python": "ml_advanced",
    "миграция обучения": "ml_advanced",
    "микроконтроллеры c++": "embedded",
    "микросервисы python": "methodologies_concepts",
    "микросервисы на java": "methodologies_concepts",
    "многопоточность c++": "programming_languages",
    "многослойный перцептрон": "ml_advanced",
    "модальная логика": "mathematics",
    "мониторинг моделей": "ml_advanced",
    "мультимодальные трансформеры": "llm_ai",
    "надежность мл": "ml_advanced",
    "нейронные сети": "ml_advanced",
    "нейросетевые инновации": "ml_advanced",
    "непрерывная интеграция данных": "data_science",
    "непрерывная поставка ml": "ml_advanced",
    "новые алгоритмы мл": "ml_advanced",
    "новые архитектуры нейросетей": "ml_advanced",
    "обработка больших данных python": "data_science",
    "обработка выбросов": "data_science",
    "обработка данных": "data_science",
    "обработка потоков на java": "data_science",
    "обучение на децентрализованных данных": "ml_advanced",
    "обучение на малых данных": "ml_advanced",
    "обучение нейросетей": "ml_advanced",
    "объединение модальностей": "llm_ai",
    "онтологии": "ml_advanced",
    "ооп python": "programming_languages",
    "оптимизаторы": "ml_advanced",
    "оптимизация обучения": "ml_advanced",
    "оркестрация контейнеров": "devops",
    "основы глубокого обучения": "ml_advanced",
    "оценка распределений": "mathematics",
    "параллельное обучение": "ml_advanced",
    "параллельное программирование": "methodologies_concepts",
    "параллельное программирование c++": "programming_languages",
    "парсинг данных": "data_science",
    "перекрёстная проверка": "ml_advanced",
    "перцептрон": "ml_advanced",
    "полносвязные сети": "ml_advanced",
    "предварительный анализ данных": "data_science",
    "предобработка данных": "data_science",
    "предобученные модели": "ml_advanced",
    "прогнозирование временных рядов": "data_science",
    "промышленная разработка ии": "ml_advanced",
    "развёртывание бд в k8s": "devops",
    "разработка алгоритмов": "ml_advanced",
    "разработка архитектур": "methodologies_concepts",
    "разработка языковых моделей": "llm_ai",
    "распределённое обучение": "ml_advanced",
    "распределённый градиентный спуск": "ml_advanced",
    "реверс-инжиниринг": "security",
    "ревью кода": "management",
    "рекурсивное исключение": "ml_advanced",
    "сверточные сети": "ml_advanced",
    "семантические сети": "ml_advanced",
    "сертификация ии": "security",
    "сетевые протоколы": "devops",
    "сигмоида": "ml_advanced",
    "системное программирование": "programming_languages",
    "скрипты python": "programming_languages",
    "скрытые слои": "ml_advanced",
    "современный c++": "programming_languages",
    "статистический анализ": "mathematics",
    "стек больших данных": "data_science",
    "теория алгоритмов": "mathematics",
    "тестирование на проникновение ии": "security",
    "топология": "mathematics",
    "трансформеры": "ml_advanced",
    "управление знаниями": "management",
    "управление качеством данных": "data_science",
    "ускорение обучения": "ml_advanced",
    "ускорение сходимости": "ml_advanced",
    "функции активации": "ml_advanced",
    "частотная статистика": "mathematics",
    "численные методы оптимизации": "mathematics",
    "этика разработки": "soft_skills",
    "эффективное обучение": "ml_advanced",
}


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_it_skills() -> set[str]:
    with open(IT_SKILLS_PATH, encoding="utf-8") as f:
        return {s.strip().lower() for s in json.load(f) if s.strip()}


def taxonomy_skill_set(taxonomy: dict) -> set[str]:
    """Все навыки + алиасы таксономии в нижнем регистре."""
    result: set[str] = set()
    for cat in taxonomy.get("categories", {}).values():
        for s in cat.get("skills", []):
            result.add(s.strip().lower())
        for a in cat.get("aliases", {}):
            result.add(a.strip().lower())
    return result


def build_prototypes(taxonomy: dict) -> dict[str, np.ndarray]:
    """Возвращает {cat_id: (n_cat × d) матрица нормализованных векторов навыков}.

    Каждая категория представлена матрицей эталонных навыков — сходство
    кандидата = максимальное косинусное сходство с ближайшим эталоном.
    """
    from src.parsing.api.embedding_loader import get_embedding_model

    model = get_embedding_model()
    prototypes: dict[str, np.ndarray] = {}
    for cat_id, cat in taxonomy.get("categories", {}).items():
        if cat_id in SKIP_CATEGORIES:
            continue
        names = [s.strip() for s in cat.get("skills", []) if s.strip()]
        if not names:
            continue
        embs = model.encode(names, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        prototypes[cat_id] = embs / norms
    return prototypes


def suggest_categories(skills: list[str], prototypes: dict[str, np.ndarray]) -> dict[str, tuple[str, float]]:
    """skill → (category_id, score). Ручные оверрайды приоритетны; остальное — эмбеддинг.

    Сходство с категорией = максимальное косинусное сходство кандидата
    с любым эталонным навыком категории (nearest-neighbor).
    """
    from src.parsing.api.embedding_loader import get_embedding_model

    model = get_embedding_model()
    cat_ids = list(prototypes.keys())
    cat_mats = {c: prototypes[c] for c in cat_ids}
    result: dict[str, tuple[str, float]] = {}
    to_embed = [s for s in skills if s not in MANUAL_OVERRIDES]
    embedded = {s: (MANUAL_OVERRIDES[s], 1.0) for s in skills if s in MANUAL_OVERRIDES}
    for i in range(0, len(to_embed), 32):
        batch = to_embed[i:i + 32]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        for j, skill in enumerate(batch):
            best_cat, best_score = "other", 0.0
            for cat_id, mat in cat_mats.items():
                sims = embs[j:j + 1] @ mat.T
                score = float(sims.max())
                if score > best_score:
                    best_cat, best_score = cat_id, score
            embedded[skill] = (best_cat, best_score)
    for s in skills:
        result[s] = embedded[s]
    return result


def apply_assignments(assignments: dict[str, tuple[str, float]], threshold: float) -> int:
    """Дописывает навыки в skill_taxonomy.json. Ручные оверрайды применяются всегда."""
    taxonomy = load_taxonomy()
    added = 0
    for skill, (cat_id, score) in assignments.items():
        if score < threshold and skill not in MANUAL_OVERRIDES:
            continue
        cats = taxonomy.get("categories", {})
        if cat_id not in cats:
            continue
        skills_list = cats[cat_id].setdefault("skills", [])
        existing = {s.strip().lower() for s in skills_list}
        if skill in existing:
            continue
        skills_list.append(skill)
        added += 1
    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    return added


def categorize_new_skills(skills: list[str]) -> dict[str, str]:
    """Возвращает {skill: category_id} для списка навыков (ручные оверрайды + эмбеддинг).

    Используется extend_skills: новые навыки сразу попадают в таксономию,
    а не остаются в категории "other". Безопасен для пустого списка.
    """
    if not skills:
        return {}
    taxonomy = load_taxonomy()
    prototypes = build_prototypes(taxonomy)
    suggestions = suggest_categories(skills, prototypes)
    result: dict[str, str] = {}
    for skill in skills:
        cat_id, score = suggestions.get(skill, ("other", 0.0))
        if skill in MANUAL_OVERRIDES:
            result[skill] = MANUAL_OVERRIDES[skill]
        elif score >= EXTEND_THRESHOLD:
            result[skill] = cat_id
        else:
            result[skill] = "other"
    return result


def main(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy()
    it_skills = load_it_skills()
    known = taxonomy_skill_set(taxonomy)
    uncategorized = sorted(it_skills - known)

    print(f"it_skills: {len(it_skills)} | taxonomy known: {len(known)} | "
          f"uncategorized: {len(uncategorized)}")

    if not uncategorized:
        print("Нет некатегоризованных навыков.")
        return

    prototypes = build_prototypes(taxonomy)
    assignments = suggest_categories(uncategorized, prototypes)
    threshold = getattr(args, "threshold", DEFAULT_THRESHOLD)

    print(f"\nПредложения (порог {threshold:.2f}):")
    for skill in uncategorized:
        cat_id, score = assignments.get(skill, ("other", 0.0))
        manual = skill in MANUAL_OVERRIDES
        mark = "✓" if (score >= threshold or manual) else " "
        src = "manual" if manual else "emb"
        print(f"  [{mark}] {skill:55s} → {cat_id:24s} ({score:.3f}) [{src}]")

    if getattr(args, "apply", False):
        added = apply_assignments(assignments, threshold)
        print(f"\nДобавлено навыков в skill_taxonomy.json: {added}")
        # Пересчёт
        new_known = taxonomy_skill_set(load_taxonomy())
        still = len(it_skills - new_known)
        print(f"Осталось некатегоризованных: {still}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аудит пробелов таксономии навыков")
    parser.add_argument("--apply", action="store_true", help="Применить предложения в JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    main(parser.parse_args())
