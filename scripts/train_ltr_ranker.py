"""Train the real LTR ranker (XGBRanker) on human/weak gold labels.

Usage:
    python scripts/train_ltr_ranker.py [--gold labeling/gold_v1.csv]
        [--vacancies 3000] [--seed 42] [--out data/models/ltr_ranker_xgb_rank.joblib]

- vacancies: sampled from Postgres (parsed_skills), lowercased copies
  (gold skills are lowercase-normalized; DB keeps original case).
- metrics -> <out>.metrics.json ; manifest saved by the engine.
"""
import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import asyncpg

sys.modules.setdefault("shap", __import__("unittest.mock", fromlist=["x"]).MagicMock())
sys.modules.setdefault("cv2", __import__("unittest.mock", fromlist=["x"]).MagicMock())

from src import Ok  # noqa: E402
from src.config import settings  # noqa: E402
from src.predictors.ltr_recommendation_engine import LTRRecommendationEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# HH experience id -> level (DB stores plain ids; the engine maps only dict-form)
EXP_MAP = {"noexperience": "junior", "between1and3": "middle",
           "between3and6": "middle", "morethan6": "senior", "morethan10": "senior"}


def read_gold(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        reader.fieldnames = [c.replace("\ufeff", "").strip() for c in (reader.fieldnames or [])]
        gcol = "grade" if "grade" in (reader.fieldnames or []) else "gold"
        for r in reader:
            try:
                g = int(r[gcol])
            except (KeyError, ValueError, TypeError):
                continue
            if g in (0, 1, 2):
                rows.append({"role": r["role"].strip(),
                             "skill": r["skill"].strip().lower(), "grade": g})
    return rows


async def load_vacancies(limit: int) -> list[dict]:
    url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    pool = await asyncpg.create_pool(url, min_size=1, max_size=2, command_timeout=300)
    async with pool.acquire() as c:
        dbrows = await c.fetch(
            "SELECT parsed_skills, experience, (SELECT jsonb_agg(el->>'id') "
            "FROM jsonb_array_elements(raw->'professional_roles') el) AS role_ids "
            "FROM vacancies WHERE parsed_skills IS NOT NULL LIMIT $1",
            limit,
        )
    await pool.close()
    out = []
    for r in dbrows:
        try:
            vals = json.loads(r["parsed_skills"]) if isinstance(r["parsed_skills"], str) else r["parsed_skills"]
        except Exception:
            continue
        if not isinstance(vals, list):
            continue
        skills = sorted({s.strip().lower() for s in vals if isinstance(s, str) and s.strip()})
        if skills:
            try:
                rids = r["role_ids"]
                rids = json.loads(rids) if isinstance(rids, str) else (rids or [])
                rids = [str(x) for x in rids if x is not None]
            except Exception:
                rids = []
            # key_skills + experience feed _extract_skills_from_vacancy /
            # _prepare_vacancies_for_levels (level_encoded feature); DB keeps
            # original case, we pass normalized copies.
            out.append({"extracted_skills": skills,
                        "key_skills": [{"name": s} for s in skills],
                        "experience": EXP_MAP.get(str(r["experience"] or "").strip().lower(), "middle"),
                        "hh_roles": rids})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="labeling/gold_v1.csv")
    ap.add_argument("--vacancies", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/models/ltr_ranker_xgb_rank.joblib")
    ap.add_argument("--drop", default="",
                    help="comma-separated features to drop (ablation)")
    args = ap.parse_args()

    gold = read_gold(ROOT / args.gold)
    print(f"gold rows: {len(gold)}, roles: {sorted({g['role'] for g in gold})}")
    vacancies = asyncio.run(load_vacancies(args.vacancies))
    print(f"vacancies: {len(vacancies)}")

    engine = LTRRecommendationEngine()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drop = tuple(s.strip() for s in args.drop.split(",") if s.strip())
    res = engine.fit_ranker(gold, vacancies, seed=args.seed, model_out=out_path,
                            drop_features=drop)
    match res:
        case Ok(metrics):
            print("METRICS:", json.dumps(metrics, ensure_ascii=False, indent=2))
            with open(str(out_path) + ".metrics.json", "w", encoding="utf-8") as f:
                json.dump({"gold": args.gold, "n_gold": len(gold),
                           "n_vacancies": len(vacancies), **metrics}, f, ensure_ascii=False, indent=2)
            print("artifact:", out_path)
        case _:
            print("TRAINING FAILED:", res)
            sys.exit(1)


if __name__ == "__main__":
    main()
