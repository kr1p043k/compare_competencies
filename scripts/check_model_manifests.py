"""Validate that every shipped LTR artifact carries a manifest/metrics sidecar.

An artifact is considered "shipped" when it sits in ``data/models`` and matches
``ltr_ranker_*.joblib``. Each one must be accompanied by EITHER

  * ``<name>.manifest.json`` (``ArtifactManifest`` format), or
  * ``<name>.joblib.metrics.json`` (plain sidecar),

and that sidecar must record

  * the artifact's ``feature_names`` (so the feature space it was trained on is
    documented next to the bytes), and
  * an NDCG@10 value under one of ``human_ndcg_at_10`` / ``ndcg@10`` /
    ``ndcg_at_10`` (the keys ``LTRRecommendationEngine`` understands).

The recorded ``feature_names`` are cross-checked against the names stored inside
the ``.joblib`` itself; a mismatch means the sidecar is stale.

``--write`` (re)creates the sidecars from the artifact and the frozen report, so
a retrain can refresh them without hand-editing JSON. It never touches the
``.joblib`` itself.

Usage:
    python scripts/check_model_manifests.py
    python scripts/check_model_manifests.py --write
    python scripts/check_model_manifests.py --models-dir data/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

# Mirrors LTRRecommendationEngine._NDCG_KEYS; kept local so the checker does not
# import the whole predictor stack (sentence-transformers, xgboost) just to read
# two lists.
NDCG_KEYS: tuple[str, ...] = ("human_ndcg_at_10", "ndcg@10", "ndcg_at_10")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = ROOT / "data" / "models"
DEFAULT_REPORT = ROOT / "data" / "reports" / "frozen_eval.json"


def read_json(path: Path) -> Any | None:
    """Return parsed JSON or ``None`` (missing / invalid / unreadable)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def pick_ndcg(mapping: Any) -> float | None:
    """First NDCG value found under the known keys, or ``None``."""
    if not isinstance(mapping, dict):
        return None
    for key in NDCG_KEYS:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def artifact_feature_names(path: Path) -> list[str] | None:
    """Feature names stored inside the artifact, or ``None`` if unreadable."""
    try:
        data = joblib.load(path)
    except Exception:
        return None
    raw = data.get("feature_names") if isinstance(data, dict) else None
    return list(raw) if raw else None


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def metrics_path(path: Path) -> Path:
    return Path(f"{path}.metrics.json")


def collect_sidecar(path: Path) -> tuple[dict[str, Any] | None, list[str] | None, float | None]:
    """Return (sidecar_dict, feature_names, ndcg) from manifest OR metrics.json.

    The manifest wins when both exist (matches the engine's read order).
    """
    for candidate in (manifest_path(path), metrics_path(path)):
        data = read_json(candidate)
        if not isinstance(data, dict):
            continue
        names = data.get("feature_names")
        names = list(names) if isinstance(names, list) else None
        ndcg = pick_ndcg(data.get("metrics")) if candidate == manifest_path(path) else pick_ndcg(data)
        if ndcg is None:
            ndcg = pick_ndcg(data)
        return data, names, ndcg
    return None, None, None


def frozen_ndcg(report: Any, name: str) -> float | None:
    if not isinstance(report, dict):
        return None
    entry = (report.get("results") or {}).get(name)
    return pick_ndcg(entry)


def validate(models_dir: Path, report: Any) -> list[str]:
    """Return a list of human-readable problems (empty means OK)."""
    artifacts = sorted(models_dir.glob("ltr_ranker_*.joblib")) if models_dir.exists() else []
    if not artifacts:
        print(f"no ltr_ranker_*.joblib under {models_dir}; nothing to validate")
        return []

    problems: list[str] = []
    for path in artifacts:
        stored = artifact_feature_names(path)
        _sidecar, names, ndcg = collect_sidecar(path)
        if names is None and ndcg is None:
            problems.append(f"{path.name}: no .manifest.json / .metrics.json sidecar")
            continue
        if names is None:
            problems.append(f"{path.name}: sidecar has no feature_names")
        elif stored is not None and names != stored:
            problems.append(
                f"{path.name}: sidecar feature_names ({len(names)}) != artifact ({len(stored)})"
            )
        if ndcg is None:
            problems.append(f"{path.name}: sidecar has no NDCG@10")
        elif frozen_ndcg(report, path.name) is None:
            problems.append(f"{path.name}: NDCG recorded but absent from the frozen report")
        status = "ok" if not problems or not problems[-1].startswith(path.name) else "FAIL"
        print(
            f"{status:4} {path.name:44} features={len(names) if names else '-':>2} "
            f"ndcg@10={ndcg if ndcg is not None else '-'}"
        )
    return problems


def write_sidecars(models_dir: Path, report: Any) -> int:
    """(Re)create manifest + metrics sidecars for every artifact. Returns count."""
    artifacts = sorted(models_dir.glob("ltr_ranker_*.joblib")) if models_dir.exists() else []
    written = 0
    for path in artifacts:
        names = artifact_feature_names(path)
        if not names:
            print(f"skip {path.name}: no feature_names readable from the artifact")
            continue
        ndcg = None
        existing = read_json(manifest_path(path)) or read_json(metrics_path(path))
        if isinstance(existing, dict):
            ndcg = pick_ndcg(existing.get("metrics")) or pick_ndcg(existing)
        if ndcg is None:
            ndcg = frozen_ndcg(report, path.name)
        metrics = {"feature_names": names, "ndcg@10": ndcg}
        if ndcg is not None:
            metrics["human_ndcg_at_10"] = ndcg

        manifest = {
            "artifact": path.name,
            "data_hash": None,
            "model_version": None,
            "feature_names": names,
            "metrics": {k: v for k, v in metrics.items() if k != "feature_names"},
        }
        manifest_path(path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metrics_path(path).write_text(
            json.dumps(
                {
                    "artifact": path.name,
                    "source": "data/reports/frozen_eval.json"
                    if frozen_ndcg(report, path.name) is not None
                    else "artifact sidecar",
                    "split": "tests/fixtures/gold_frozen.csv seed=42 test_size=0.25",
                    **metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written += 1
        print(f"wrote {manifest_path(path).name} + {metrics_path(path).name}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--write",
        action="store_true",
        help="(re)create sidecars from the artifact + frozen report instead of validating",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    report = read_json(Path(args.report))

    if args.write:
        written = write_sidecars(models_dir, report)
        print(f"\n{written} sidecar(s) written")
        return 0

    problems = validate(models_dir, report)
    if problems:
        print("\nFAIL: model manifest problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: every shipped artifact has feature_names + NDCG@10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
