"""Export existing JSON pipeline results to PostgreSQL.

Usage:
    python -m src.cli export-results
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline.db_writer import export_file_results_to_db


async def main() -> None:
    print("Exporting file results to DB...")
    count = await export_file_results_to_db()
    print(f"Done. {count} records written.")


if __name__ == "__main__":
    asyncio.run(main())
