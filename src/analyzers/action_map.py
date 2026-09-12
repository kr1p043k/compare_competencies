"""ACTION -> TOOL map: RPD verbs resolve to market tools (v31).

Two variants in one table (user-approved design):
- grounded: tools VERIFIED present in market - always active.
- expert: user's additions, applied ONLY if the tool appears in market later
  (dormant until data supports it - never dead matches, auto-activates).
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\w+", flags=re.UNICODE)

ACTION_TOOLS: list[dict] = [
    {"id": 'normalize', "triggers": ['нормализация', 'нормализовать', 'масштабирование', 'масштабировать', 'стандартизация', 'стандартизировать'],
     "grounded": ['pandas', 'scikit-learn', 'numpy'], "expert": []},
    {"id": 'encoding', "triggers": ['кодирование', 'кодировать', 'one-hot', 'label encoding', 'get_dummies', 'ordinalencoder', 'labelencoder'],
     "grounded": ['pandas', 'scikit-learn'], "expert": []},
    {"id": 'featureselection', "triggers": ['отбор признаков', 'feature selection', 'селекция признаков'],
     "grounded": ['scikit-learn'], "expert": ['boruta', 'shap', 'eli5', 'feature-engine']},
    {"id": 'cv', "triggers": ['кросс-валидация', 'кроссвалидация', 'гиперпараметр', 'подбор гиперпараметров', 'gridsearch'],
     "grounded": ['scikit-learn'], "expert": ['optuna', 'hyperopt', 'ray']},
    {"id": 'regularization', "triggers": ['регуляризация', 'регуляризировать', 'ridge', 'lasso'],
     "grounded": ['scikit-learn'], "expert": []},
    {"id": 'featureeng', "triggers": ['feature engineering', 'генерация признаков', 'конструирование признаков', 'инженерия признаков'],
     "grounded": ['pandas', 'scikit-learn'], "expert": []},
    {"id": 'preprocessing', "triggers": ['предобработка', 'предобработать', 'очистка данных', 'пропуски', 'выбросы', 'missing values'],
     "grounded": ['pandas', 'numpy'], "expert": ['pyod']},
    {"id": 'refactoring', "triggers": ['рефакторинг', 'рефакторить', 'оптимизация кода', 'оптимизировать код'],
     "grounded": ['python'], "expert": []},
    {"id": 'vectorization', "triggers": ['векторизация', 'векторизовать'],
     "grounded": ['numpy'], "expert": []},
    {"id": 'gpu', "triggers": ['gpu', 'cuda', 'параллельные вычисления', 'распараллеливание', 'распределенные вычисления'],
     "grounded": ['pytorch', 'tensorflow', 'cuda'], "expert": []},
    {"id": 'visualization', "triggers": ['визуализация', 'визуализировать', 'eda', 'разведочный анализ'],
     "grounded": ['matplotlib', 'seaborn', 'plotly'], "expert": []},
    {"id": 'pipelines', "triggers": ['пайплайн', 'pipeline', 'конвейер данных'],
     "grounded": ['scikit-learn', 'prefect'], "expert": ['luigi', 'dagster']},
    {"id": 'deeplearning', "triggers": ['нейросеть', 'нейронная сеть', 'deep learning', 'глубокое обучение'],
     "grounded": ['pytorch', 'tensorflow'], "expert": []},
    {"id": 'nlp', "triggers": ['nlp', 'обработка естественного языка', 'обработка текстов', 'spacy'],
     "grounded": ['transformers', 'spacy'], "expert": ['nltk', 'razdel', 'pymorphy']},
    {"id": 'cv2', "triggers": ['computer vision', 'компьютерное зрение', 'opencv', 'детекция'],
     "grounded": ['opencv', 'pytorch'], "expert": []},
    {"id": 'sql', "triggers": ['sql', 'субд', 'реляционные базы', 'реляционная база'],
     "grounded": ['sql', 'postgresql'], "expert": []},
    {"id": 'deploy', "triggers": ['api', 'деплой', 'fastapi', 'развертывание модели', 'инференс'],
     "grounded": ['fastapi', 'docker', 'kubernetes'], "expert": []},
    {"id": 'testing', "triggers": ['юнит-тест', 'unit test', 'pytest', 'модульное тестирование'],
     "grounded": ['pytest'], "expert": []},
    {"id": 'versioning', "triggers": ['git', 'версионирование', 'контроль версий', 'github'],
     "grounded": ['git'], "expert": []},
    {"id": 'linux', "triggers": ['linux', 'терминал', 'командная строка', 'bash'],
     "grounded": ['linux', 'bash'], "expert": []},
    {"id": 'jupyter', "triggers": ['jupyter', 'colab', 'google colab'],
     "grounded": ['jupyter'], "expert": ['colab']},
    {"id": 'mlops', "triggers": ['mlops', 'трекинг экспериментов', 'мониторинг моделей', 'mlflow'],
     "grounded": ['mlflow'], "expert": ['wandb', 'clearml', 'dvc']},
    {"id": 'boosting', "triggers": ['xgboost', 'lightgbm', 'catboost', 'бустинг', 'градиентный бустинг'],
     "grounded": ['xgboost', 'lightgbm', 'catboost'], "expert": []},
    {"id": 'kafka', "triggers": ['kafka', 'брокер сообщений', 'очереди сообщений', 'очередь сообщений'],
     "grounded": ['kafka', 'apache kafka'], "expert": []},
    {"id": 'redis', "triggers": ['redis', 'кэширование', 'кэш'],
     "grounded": ['redis'], "expert": []},
    {"id": 'orm', "triggers": ['orm', 'sqlalchemy', 'миграции', 'alembic'],
     "grounded": ['sqlalchemy'], "expert": ['alembic']},
]


_TRIGGER_CACHE: list[tuple[frozenset, dict]] | None = None


def _trigger_sets():
    """Lazy (trigger-lemma-set, entry) pairs; lemmas via ru_morph with fallback."""
    global _TRIGGER_CACHE
    if _TRIGGER_CACHE is not None:
        return _TRIGGER_CACHE
    try:
        from src.text.ru_morph import lemmas as _lemmas
    except Exception:
        _lemmas = None
    out = []
    for entry in ACTION_TOOLS:
        for trig in entry["triggers"]:
            if _lemmas is not None:
                try:
                    toks = frozenset(_lemmas(trig))
                except Exception:
                    toks = frozenset(_WORD.findall(trig.lower()))
            else:
                toks = frozenset(_WORD.findall(trig.lower()))
            if toks:
                out.append((toks, entry))
    _TRIGGER_CACHE = out
    return out


def phrase_lemmas(text: str) -> set[str]:
    """Lemma set of a phrase (ru_morph with plain-token fallback)."""
    try:
        from src.text.ru_morph import lemmas as _lemmas
        return set(_lemmas(text))
    except Exception:
        return set(_WORD.findall((text or "").lower()))


def resolve_action_tools(lemmas: set[str], market: dict[str, int]):
    """Return (tool, origin) or None. Grounded pool first, dormant expert second.
    Within a pool the highest-frequency present tool wins (name tiebreak)."""
    best = None
    for toks, entry in _trigger_sets():
        if not toks or not (toks <= lemmas):
            continue
        for pool, origin in ((entry["grounded"], "grounded"), (entry["expert"], "expert")):
            cands = [(t, market[t]) for t in pool if t in market]
            if not cands:
                continue
            cands.sort(key=lambda x: (-x[1], x[0]))
            tool, freq = cands[0]
            if best is None or (freq, tool) > (best[2], best[0]):
                best = (tool, origin, freq)
            break
    if best is None:
        return None
    return (best[0], best[1])

