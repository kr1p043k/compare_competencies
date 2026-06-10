"""Export vacancies from JSON files to PostgreSQL.

Usage:
    python -m src.cli export-vacancies [--basic FILE] [--detailed FILE]
"""

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from src import config
from src.pipeline.db_writer import export_vacancies_from_json


async def main(
    basic_path: str | None = None,
    detailed_path: str | None = None,
) -> None:
    basic = Path(basic_path) if basic_path else config.DATA_RAW_DIR / "hh_vacancies_basic.json"
    detailed = Path(detailed_path) if detailed_path else config.DATA_PROCESSED_DIR / "hh_vacancies_detailed.json"

    print(f"Importing vacancies from:\n  basic: {basic}\n  detailed: {detailed}")
    total = await export_vacancies_from_json(basic_path=basic, detailed_path=detailed)
    print(f"Done. {total} vacancies saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic", help="Path to basic vacancies JSON")
    parser.add_argument("--detailed", help="Path to detailed vacancies JSON")
    args = parser.parse_args()
    asyncio.run(main(basic_path=args.basic, detailed_path=args.detailed))
