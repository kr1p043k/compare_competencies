"""Map KSA text to it_skills. Three tiers: JSON > substring > semantic.

Usage:
    python -m src.cli.map_ksa_to_skills [--json-map data/reference/krm_it_skill_map.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import async_session_factory
from src.models.krm_models import Competency, CompetencySkill, Direction, Discipline, Skill

logger = structlog.get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
KRM_PATH = DATA_DIR / "reference" / "krm_disciplines_09.03.02.json"
JSON_MAP_PATH = DATA_DIR / "reference" / "krm_it_skill_map.json"

SEMANTIC_THRESHOLD = 0.45
MIN_SUBSTRING_LEN = 3


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def tier_explicit_json(session, json_map_path, disc_map, comp_map, skill_map):
    """Tier 1: Explicit JSON mapping."""
    if not json_map_path.exists():
        print(f"  [tier1] skip: {json_map_path} not found")
        return 0
    mapping = load_json(json_map_path)
    count = 0
    for disc_name, comps in mapping.items():
        disc_id = disc_map.get(disc_name)
        if not disc_id:
            continue
        for comp_code, skill_names in comps.items():
            comp_id = comp_map.get((disc_id, comp_code))
            if not comp_id:
                continue
            for sn in skill_names:
                sk_id = skill_map.get(sn.strip().lower())
                if not sk_id:
                    continue
                ex = await session.execute(
                    select(CompetencySkill).where(
                        CompetencySkill.competency_id == comp_id,
                        CompetencySkill.skill_id == sk_id))
                if ex.scalar_one_or_none():
                    continue
                session.add(CompetencySkill(
                    competency_id=comp_id, skill_id=sk_id,
                    ksa_type="flat", source_text=sn, match_type="explicit"))
                count += 1
    await session.flush()
    print(f"  [tier1] Explicit: {count}")
    return count


async def tier_substring(session, krm, disc_map, comp_map):
    """Tier 2: Substring scan of KSA text."""
    it_res = await session.execute(select(Skill).where(Skill.source == "it_skills"))
    it_skills = {s.name.lower(): s.id for s in it_res.scalars().all()}
    count = 0
    for dn, dd in krm.get("09.03.02", {}).get("disciplines", {}).items():
        did = disc_map.get(dn)
        if not did:
            continue
        for cc, ksa in dd.get("ksa", {}).items():
            cid = comp_map.get((did, cc))
            if not cid:
                continue
            txt = " ".join(t for kt in ("knowledge", "abilities", "skills") for t in ksa.get(kt, [])).lower()
            if len(txt) < 10:
                continue
            for iname, iid in it_skills.items():
                if len(iname) < MIN_SUBSTRING_LEN or iname not in txt:
                    continue
                ex = await session.execute(
                    select(CompetencySkill).where(
                        CompetencySkill.competency_id == cid,
                        CompetencySkill.skill_id == iid))
                if ex.scalar_one_or_none():
                    continue
                session.add(CompetencySkill(
                    competency_id=cid, skill_id=iid,
                    ksa_type="flat", source_text=iname, match_type="substring"))
                count += 1
    await session.flush()
    print(f"  [tier2] Substring: {count}")
    return count


async def tier_semantic(session, krm, disc_map, comp_map):
    """Tier 3: Embedding cosine similarity."""
    try:
        from src.analyzers.comparison.embedding_provider import EmbeddingProviderFactory
        prov = EmbeddingProviderFactory.get()
    except Exception as e:
        print(f"  [tier3] skip: {e}")
        return 0

    it_res = await session.execute(select(Skill).where(Skill.source == "it_skills"))
    it_list = [(s.name.lower(), s.id) for s in it_res.scalars().all()]
    if not it_list:
        return 0
    it_names = [n for n, _ in it_list]
    it_ids = [i for _, i in it_list]

    it_embs = prov.encode(it_names, show_progress_bar=False)
    it_norms = np.linalg.norm(it_embs, axis=1, keepdims=True)
    it_norms[it_norms == 0] = 1.0
    it_embs_n = it_embs / it_norms

    count = 0
    for dn, dd in krm.get("09.03.02", {}).get("disciplines", {}).items():
        did = disc_map.get(dn)
        if not did:
            continue
        for cc, ksa in dd.get("ksa", {}).items():
            cid = comp_map.get((did, cc))
            if not cid:
                continue
            txt = " ".join(t for kt in ("knowledge", "abilities", "skills") for t in ksa.get(kt, [])).lower()
            if len(txt) < 10:
                continue
            qemb = prov.encode([txt], show_progress_bar=False)
            qn = np.linalg.norm(qemb)
            if qn > 0:
                qemb = qemb / qn
            sims = it_embs_n @ qemb.T
            top = np.argsort(sims)[::-1][:3]
            for idx in top:
                if float(sims[idx]) < SEMANTIC_THRESHOLD:
                    break
                ex = await session.execute(
                    select(CompetencySkill).where(
                        CompetencySkill.competency_id == cid,
                        CompetencySkill.skill_id == it_ids[idx]))
                if ex.scalar_one_or_none():
                    continue
                session.add(CompetencySkill(
                    competency_id=cid, skill_id=it_ids[idx],
                    ksa_type="flat", source_text=it_names[idx], match_type="semantic"))
                count += 1
    await session.flush()
    print(f"  [tier3] Semantic: {count}")
    return count


async def run_mapping(json_map_path: Path | None = None) -> dict:
    """Run all three tiers."""
    krm = load_json(KRM_PATH)
    map_path = json_map_path or JSON_MAP_PATH

    async with async_session_factory() as session:
        dir_result = await session.execute(select(Direction).where(Direction.code == "09.03.02"))
        direction = dir_result.scalar_one_or_none()
        if not direction:
            print("ERROR: Direction 09.03.02 not found")
            return {}

        disc_result = await session.execute(select(Discipline).where(Discipline.direction_id == direction.id))
        disc_map = {d.name: d.id for d in disc_result.scalars().all()}

        comp_result = await session.execute(select(Competency).where(Competency.discipline_id.in_(disc_map.values())))
        comp_map = {(c.discipline_id, c.code): c.id for c in comp_result.scalars().all()}

        skill_result = await session.execute(select(Skill))
        skill_map = {s.name.lower(): s.id for s in skill_result.scalars().all()}

        print(f"Maps: {len(disc_map)} disciplines, {len(comp_map)} competencies, {len(skill_map)} skills")

        t1 = await tier_explicit_json(session, map_path, disc_map, comp_map, skill_map)
        t2 = await tier_substring(session, krm, disc_map, comp_map)
        t3 = await tier_semantic(session, krm, disc_map, comp_map)

        await session.commit()

        total = await session.execute(
            select(CompetencySkill).where(CompetencySkill.competency_id.in_(comp_map.values())))
        all_links = total.scalars().all()

        stats = {"tier1": t1, "tier2": t2, "tier3": t3, "total_new": t1 + t2 + t3, "total_links": len(all_links)}
        print(f"\nDone: +{stats['total_new']} new ({t1} explicit + {t2} substring + {t3} semantic)")
        print(f"Total competency_skills: {stats['total_links']}")
        return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Map KSA to it_skills (3-tier)")
    parser.add_argument("--json-map", type=str, default=None)
    args = parser.parse_args()
    path = Path(args.json_map) if args.json_map else None
    asyncio.run(run_mapping(json_map_path=path))
