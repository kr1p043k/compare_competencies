"""Сравнить два снапшота частот навыков и сообщить о дрейфе (PSI).

    python -m scripts.check_skill_drift                       # два последних freq_market_*
    python -m scripts.check_skill_drift A.json B.json         # явные пути
    python -m scripts.check_skill_drift --threshold 0.25 --json

Код выхода 1, если дрейф значимый (удобно для CI/крона).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzers.skills.drift import detect_drift  # noqa: E402

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_two(history_dir: Path = HISTORY_DIR) -> tuple[Path, Path]:
    snaps = sorted(history_dir.glob("freq_market_*.json"))
    if len(snaps) < 2:
        raise SystemExit(f"Нужно >=2 снапшота freq_market_* в {history_dir}")
    return snaps[-2], snaps[-1]


def main(expected: Path | None = None, actual: Path | None = None,
         threshold: float = 0.25, as_json: bool = False) -> int:
    if expected is None or actual is None:
        expected, actual = _latest_two()
    report = detect_drift(_load(expected), _load(actual), psi_threshold=threshold)
    if as_json:
        print(json.dumps({"expected": str(expected), "actual": str(actual), **report}, ensure_ascii=False, indent=2))
    else:
        print(f"drift {expected.name} -> {actual.name}: PSI={report['psi']:.4f} ({report['level']})")
        for c in report["top_contributors"]:
            print(f"  {c['skill']}: psi={c['psi']:.4f} "
                  f"share {c['expected_share']:.4f} -> {c['actual_share']:.4f}")
    return 1 if report["drifted"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дрейф распределения навыков (PSI)")
    parser.add_argument("expected", nargs="?", type=Path)
    parser.add_argument("actual", nargs="?", type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.expected, args.actual, args.threshold, args.json))
