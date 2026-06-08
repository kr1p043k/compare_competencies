"""CLI: run teacher analysis. Usage: python -m src.cli teacher-analysis"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline.teacher_analysis_runner import run_teacher_analysis
from src.result import Ok


def main(direction: str = "09.03.02", discipline: str | None = None) -> None:
    result = asyncio.run(run_teacher_analysis(
        direction_code=direction,
        discipline_filter=discipline,
    ))
    if isinstance(result, Ok):
        d = result.unwrap()
        cov = f"{d['average_coverage']:.1%}"
        print(f"OK — {d['direction']}: {d['total_disciplines']} disciplines, "
              f"coverage {cov}, gaps {d['total_gaps_across_all']}")
    else:
        print(f"ERROR: {result.err()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", default="09.03.02")
    args, _ = parser.parse_known_args()
    main(direction=args.direction)
