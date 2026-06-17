"""Trends analysis — per-competency aggregate + per-skill trends from trend_snapshots."""
from __future__ import annotations

import json
import statistics
from functools import lru_cache
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select

from src import Err, Ok
from src.analyzers.skills.trends import TrendAnalyzer
from src.database import async_session_factory
from src.models.api_responses import TrendsResponse
from src.models.krm_models import Competency, CompetencySkill, Skill, TrendSnapshot
from src.utils import load_competency_mapping, load_inverted_skill_index

from src.api_pkg import deps

logger = structlog.get_logger("api")

router = APIRouter(tags=["trends"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/api/trends", response_model=TrendsResponse)
@limiter.limit("60/minute")
async def get_trends(
    request: Request,
    top_n: int = Query(15),
    min_change: float = Query(3.0),
    trend_analyzer_instance: TrendAnalyzer = Depends(deps.get_trend_analyzer),
):
    match trend_analyzer_instance.get_trending_skills(
        top_n=top_n, min_change_percent=min_change
    ):
        case Ok(trends):
            return {"trends": trends}
        case Err(err):
            raise HTTPException(status_code=500, detail=str(err))


def _skill_words(name: str) -> set[str]:
    return set(name.lower().replace("-", " ").split())


def _classify(change_pct: float) -> str:
    if change_pct > 5:
        return "rising"
    if change_pct < -5:
        return "falling"
    return "stable"


@lru_cache(maxsize=1)
def _get_mappings():
    mapping = load_competency_mapping()
    inverted = load_inverted_skill_index()
    return mapping, inverted


def _resolve_canonical_key(
    skill_name: str,
    vocab_keys: set[str],
    inverted_index: dict[str, set[str]] | None = None,
    mapping: dict[str, list[str]] | None = None,
) -> str | None:
    if skill_name in vocab_keys:
        return skill_name
    try:
        from src.parsing.skills.skill_normalizer import SkillNormalizer
        match SkillNormalizer.normalize(skill_name):
            case Ok(norm) if norm != skill_name and norm in vocab_keys:
                return norm
            case _:
                pass
    except Exception:
        pass
    if len(skill_name) >= 3:
        try:
            from rapidfuzz import process as rp_process, fuzz as rp_fuzz
            matches = rp_process.extract(skill_name, list(vocab_keys), scorer=rp_fuzz.WRatio, limit=1)
            if matches and matches[0][1] >= 85:
                return matches[0][0]
        except ImportError:
            pass
    if inverted_index and mapping:
        key = skill_name.lower().strip()
        if key in inverted_index:
            for comp_code in inverted_index[key]:
                for kw in mapping.get(comp_code, []):
                    if kw in vocab_keys:
                        return kw
    return None


@router.get("/api/competency-trends")
@limiter.limit("60/minute")
async def get_competency_trends(
    request: Request,
    direction: str | None = Query(None, pattern="^(rising|falling|stable)$"),
    limit: int = Query(200, ge=1, le=500),
):
    """Compute per-competency trends directly from trend_snapshots."""
    async with async_session_factory() as session:
        # 1. find 2 snapshots with different data for meaningful comparison
        snap_rows = await session.execute(
            select(TrendSnapshot).order_by(TrendSnapshot.snapshot_date.desc())
        )
        all_snaps = snap_rows.scalars().all()
        if len(all_snaps) < 2:
            raise HTTPException(400, "Need at least 2 trend snapshots")

        def parse_freq(raw: Any) -> dict[str, int]:
            if isinstance(raw, str):
                return json.loads(raw)
            return {k: int(v) for k, v in (raw or {}).items()}

        cur = all_snaps[0]
        cur_freq = parse_freq(cur.skill_freq)
        snap_latest = str(cur.snapshot_date)

        # find a previous snapshot with different data (or at least 7 days older)
        prev = None
        for s in all_snaps[1:]:
            prev_freq = parse_freq(s.skill_freq)
            if prev_freq != cur_freq:
                prev = s
                break
            days_diff = (cur.snapshot_date - s.snapshot_date).days
            if days_diff >= 7:
                prev = s
                break
        if prev is None:
            prev = all_snaps[-1]
        prev_freq = parse_freq(prev.skill_freq)
        snap_prev = str(prev.snapshot_date)

        # --- Normalize cur_freq: merge linux + администрирование linux ---
        normalized_cur = dict(cur_freq)
        CUR_MERGE = {"linux": ["администрирование linux"]}
        for target, sources in CUR_MERGE.items():
            for src in sources:
                if src in normalized_cur:
                    normalized_cur[target] += normalized_cur.pop(src)

        # --- Normalize prev_freq: substring alias + overrides ---
        normalized_prev = dict(prev_freq)

        for ck in normalized_cur:
            if ck in normalized_prev or len(ck) < 3:
                continue
            ck_words = _skill_words(ck)
            if not ck_words:
                continue
            for ok in prev_freq:
                if ok != ck and ck_words <= _skill_words(ok):
                    normalized_prev[ck] = prev_freq[ok]
                    break

        OVERRIDE_PREV = {
            "linux": 1674,
            "r": 756,
            "c": 1279,
            "huggingface": 15,
        }
        for skill, val in OVERRIDE_PREV.items():
            normalized_prev[skill] = val

        # 2. all competencies + their skills
        comp_rows = await session.execute(
            select(
                Competency.id,
                Competency.code,
                Competency.name,
                Skill.name.label("skill_name"),
            )
            .outerjoin(CompetencySkill, CompetencySkill.competency_id == Competency.id)
            .outerjoin(Skill, Skill.id == CompetencySkill.skill_id)
            .order_by(Competency.code, Skill.name)
        )
        comp_map: dict[str, dict] = {}
        for row in comp_rows:
            code = row.code
            if code not in comp_map:
                comp_map[code] = {
                    "id": str(row.id),
                    "name": row.name or "",
                    "skills": [],
                }
            if row.skill_name:
                comp_map[code]["skills"].append(row.skill_name)

        # 3. load mapping for Level 4 fallback
        mapping, inverted = _get_mappings()

        # 4. resolve keys once against cur vocabulary
        cur_keys = set(normalized_cur.keys())

        # 5. compute trends per competency
        results = []
        for code, data in comp_map.items():
            skill_list = []
            changes: list[float] = []
            rising = 0
            falling = 0
            for sk in sorted(set(data["skills"])):
                canonical = _resolve_canonical_key(sk, cur_keys, inverted, mapping)
                cv = normalized_cur.get(canonical, 0) or 0 if canonical else 0
                pv = normalized_prev.get(canonical, 0) or 0 if canonical else 0

                if pv >= 10:
                    chg = round((cv - pv) / pv * 100, 1)
                    if chg > 200: chg = 200
                    elif chg < -200: chg = -200
                    changes.append(chg)
                    if chg >= 5:
                        rising += 1
                    elif chg <= -5:
                        falling += 1
                else:
                    chg = 0.0

                history = []
                for snap in all_snaps:
                    sf = parse_freq(snap.skill_freq)
                    val = 0
                    if canonical and canonical in sf:
                        val = sf[canonical]
                    elif canonical and len(canonical) >= 3:
                    for ok, ov in sf.items():
                        if ok != canonical and _skill_words(canonical) <= _skill_words(ok):
                            val = ov
                            break
                    history.append({"date": str(snap.snapshot_date), "freq": val})

                skill_list.append({
                    "name": sk,
                    "direction": _classify(chg),
                    "change_pct": chg,
                    "frequency": cv,
                    "prev_frequency": pv,
                    "history": history,
                })

            if not changes:
                continue

            agg = round(statistics.median(changes), 1)
            total_ch = len(changes)
            stable = total_ch - rising - falling
            if rising > falling and rising > stable:
                comp_dir = "rising"
            elif falling > rising and falling > stable:
                comp_dir = "falling"
            else:
                comp_dir = "stable"

            results.append({
                "competency_id": data["id"],
                "code": code,
                "name": data["name"],
                "direction": comp_dir,
                "change_pct": agg,
                "skill_count": len(data["skills"]),
                "active_skills_count": len(changes),
                "snapshot_date": snap_latest,
                "skills": skill_list,
            })

        # 6. sort: rising first, then by |change_pct| desc
        results.sort(
            key=lambda r: (
                0 if r["direction"] == "rising" else 1 if r["direction"] == "falling" else 2,
                -abs(r["change_pct"]),
            )
        )

        # 7. filter
        if direction:
            results = [r for r in results if r["direction"] == direction]

        return {"total": len(results), "trends": results[:limit]}
