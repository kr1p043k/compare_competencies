"""Export existing JSON pipeline results to PostgreSQL.

Usage:
    python -m src.cli export-results
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")

async def main() -> None:
    from src.pipeline.db_writer import export_vacancies_from_json
    print("Exporting file results to DB...")
    count = await export_vacancies_from_json()
    print(f"Done. {count} records written.")


if __name__ == "__main__":
    asyncio.run(main())
