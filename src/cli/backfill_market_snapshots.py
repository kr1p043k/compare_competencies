"""Построить недостающие месячные снимки рынка из parsed_skills вакансий.

Проблема: история прогнозов строится из freq_market_YYYY-MM.json файлов, но
снимок сохраняется только в текущий месяц при сборе. Если какой-то месяц был
пропущен (например июль), прогнозы не видят его точку.

Решение: этот скрипт считает месячные частоты навыков из vacancies.parsed_skills
(по published_at) и сохраняет freq_market_YYYY-MM.json для месяцев, у которых
файла ещё нет. Существующие снимки не перезаписываются.

Usage:
    python -m src.cli backfill-market-snapshots [--force]
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime

from sqlalchemy import text

from src import config
from src.database import async_session_factory


async def _build_monthly_freqs() -> dict[str, dict[str, float]]:
    """Считает частоты навыков по месяцам published_at из parsed_skills."""
    monthly: dict[str, Counter] = {}
    async with async_session_factory() as session:
        rows = await session.execute(text(
            """
            SELECT
                to_char(date_trunc('month', v.published_at::timestamp), 'YYYY-MM') AS month,
                LOWER(TRIM(ps::text)) AS skill,
                COUNT(DISTINCT v.id) AS freq
            FROM vacancies v
            CROSS JOIN LATERAL jsonb_array_elements_text(v.parsed_skills::jsonb) AS ps
            WHERE v.parsed_skills IS NOT NULL
              AND v.parsed_skills::text != '[]'
              AND v.published_at IS NOT NULL
            GROUP BY month, skill
            """
        ))
        for row in rows:
            month = row.month
            if month not in monthly:
                monthly[month] = Counter()
            monthly[month][row.skill] = int(row.freq)
    return {m: {k: float(v) for k, v in c.items()} for m, c in monthly.items()}


def _existing_months() -> set[str]:
    existing: set[str] = set()
    for f in sorted(config.HISTORY_DIR.glob("freq_market_*.json")):
        # имя файла: freq_market_2026-04.json -> 2026-04
        existing.add(f.stem.replace("freq_market_", ""))
    return existing


def _write_month_snapshot(month: str, freqs: dict[str, float]) -> None:
    """Пишет freq_market_YYYY-MM.json в формате _meta (как save_snapshot)."""
    # whitelist-фильтр, как в save_snapshot(apply_whitelist=True)
    try:
        from src.parsing.skills.skill_validator import SkillValidator
        validator = SkillValidator()
        filtered = {}
        for skill, freq in freqs.items():
            r = validator.validate(skill)
            if r.is_ok() and r.ok().is_valid:
                filtered[skill] = freq
    except Exception:
        filtered = freqs

    path = config.HISTORY_DIR / f"freq_market_{month}.json"
    out = {
        "_meta": {
            "type": "full_market",
            "snapshot_date": f"{month}-01",
            "vacancy_count": None,
            "source": "it_sector",
        }
    }
    for k, v in filtered.items():
        out[k] = v
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    wrote {path.name}: {len(filtered)} skills")


def main(force: bool = False) -> None:
    monthly = asyncio.run(_build_monthly_freqs())
    if not monthly:
        print("No monthly data found in vacancies.parsed_skills")
        return

    existing = _existing_months()
    print(f"Months in DB: {sorted(monthly)}")
    print(f"Existing freq_market files: {sorted(existing)}")

    missing = sorted(set(monthly) - existing)
    if force:
        missing = sorted(monthly)

    if not missing:
        print("Nothing to backfill")
        return

    print(f"Backfilling {len(missing)} months: {missing}")
    for month in missing:
        print(f"  {month}: {len(monthly[month])} skills")
        _write_month_snapshot(month, monthly[month])

    print("Done.")


if __name__ == "__main__":
    import sys
    main("--force" in sys.argv)
