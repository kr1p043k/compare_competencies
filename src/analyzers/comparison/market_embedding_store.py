"""Общие хелперы маркет-эмбеддингов: один хэш данных, subset/merge без перекодирования.

До: SkillMatcher.set_market и EmbeddingComparator.build_market_index
кодировали одни и те же ~4000 навыков независимо (холодный старт — минуты),
а comparator грузил joblib вслепую. После: обе точки проверяют
model_version + data_hash(sorted(names)); при расхождении состава —
докодируют только недостающее и атомарно перезаписывают.

I/O намеренно НЕ здесь: тесты мокируют joblib/ArtifactManifest
в неймспейсах call sites. Здесь только чистая логика.
Храним СЫРЫЕ векторы; нормализацию делают call sites как раньше.
"""

import hashlib

import numpy as np


def market_data_hash(names) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:16]


def split_cached(cached_skills, cached_embs, names):
    """Разделить запрос на имеющееся (в порядке names) и недостающее.

    Возвращает (ordered_names, mat_or_None, missing).
    mat_or_None — готовый массив для имен, найденных в кэше.
    """
    names = list(names)
    row_by_skill = {s: i for i, s in enumerate(cached_skills)}
    have_idx = [row_by_skill[n] for n in names if n in row_by_skill]
    missing = [n for n in names if n not in row_by_skill]
    have_names = [names[i] for i, n in enumerate(names) if n in row_by_skill]
    mat = np.asarray(cached_embs)[np.fromiter(have_idx, dtype=np.int64, count=len(have_idx))] \
        if have_idx else None
    return have_names, mat, missing


def merge_matrix(have_names, have_mat, missing_names, missing_mat):
    """Слить кэшированное и докодированное в один массив (порядок: have, missing)."""
    have_mat = np.asarray(have_mat) if have_mat is not None else None
    missing_mat = np.asarray(missing_mat) if missing_mat is not None else None
    parts, order = [], list(have_names)
    if have_mat is not None and len(have_names):
        parts.append(have_mat)
    if missing_mat is not None and len(missing_names):
        parts.append(missing_mat)
        order.extend(missing_names)
    if not parts:
        return [], np.zeros((0, 0), dtype=np.float32)
    return order, np.vstack(parts)
