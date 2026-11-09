"""Проверка манифестов/версий моделей на консистентность.

Для каждого LTR-артефакта в ``data/models/`` сверяет:
  * сохранённые ``feature_names`` с рантайм-набором признаков;
  * наличие записанной метрики NDCG@10 (манифест/метрики);
  * попадание в список несовместимых моделей.

    python -m scripts.check_model_manifests [--models-dir DIR]
Код выхода 1, если есть проблемы (кофаундер/CI-гейт).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictors.ltr_recommendation_engine import (  # noqa: E402
    _RUNTIME_FEATURES,
    LTRRecommendationEngine,
)

_NDCG_KEYS = ("human_ndcg_at_10", "ndcg@10", "ndcg_at_10")


def _manifest_ndcg(manifest: Mapping[str, Any] | None) -> float | None:
    metrics = (manifest or {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    for key in _NDCG_KEYS:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None
    return None


def validate(
    name: str,
    feature_names: Sequence[str] | None,
    manifest: Mapping[str, Any] | None,
    runtime_features: Sequence[str],
    incompatible: Mapping[str, str],
) -> list[str]:
    """Возвращает список проблем для одного артефакта (пустой = всё ок)."""
    problems: list[str] = []
    if name in incompatible:
        return [f"в списке несовместимых: {incompatible[name]}"]
    if feature_names is None:
        problems.append("в артефакте нет feature_names")
    elif tuple(feature_names) != tuple(runtime_features):
        problems.append(
            f"feature_names не совпадают с рантаймом: {list(feature_names)} != {list(runtime_features)}"
        )
    if _manifest_ndcg(manifest) is None:
        problems.append("нет записанной метрики NDCG@10 (манифест/метрики)")
    return problems


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _artifact_feature_names(path: Path) -> tuple[str, ...] | None:
    try:
        import joblib

        data = joblib.load(path)
    except Exception:
        return None
    names = data.get("feature_names") if isinstance(data, dict) else None
    return tuple(names) if names else None


def main(models_dir: Path | None = None) -> int:
    from src import config

    md = Path(models_dir) if models_dir else config.MODELS_DIR
    artifacts = sorted(md.glob("*.joblib"))
    if not artifacts:
        print(f"Нет артефактов в {md} - нечего проверять")
        return 0

    runtime = _RUNTIME_FEATURES
    incompatible = LTRRecommendationEngine.INCOMPATIBLE_MODELS
    failed = False
    for path in artifacts:
        name = path.name
        manifest = _read_json(path.with_suffix(".manifest.json"))
        problems = validate(name, _artifact_feature_names(path), manifest, runtime, incompatible)
        if problems:
            failed = True
            print(f"PROBLEM {name}:")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"ok {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка манифестов моделей")
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()
    sys.exit(main(args.models_dir))
