"""Find near-duplicate skills by embedding similarity (report only by default).

String normalisation is unsafe here: stripping punctuation maps ``c``, ``c#``
and ``c++`` onto the same key, which are three different skills. Embeddings
separate them, so candidates are ranked by cosine similarity instead.

Nothing is modified unless ``--apply`` is passed, and even then only pairs the
operator explicitly lists in ``--pair`` are merged.

Usage:
    python scripts/find_skill_duplicates.py
    python scripts/find_skill_duplicates.py --threshold 0.95 --top 40
    python scripts/find_skill_duplicates.py --apply --pair <keep_id> <drop_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import psycopg

DEFAULT_DSN = "postgresql://postgres:Admin_123!@localhost:5432/compare_competencies"
ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "reports" / "skill_duplicates.json"


def load_skills(conn) -> tuple[list[str], list[str], np.ndarray]:
    rows = conn.execute(
        "select id::text, name, embedding::text from skills "
        "where embedding is not null order by name"
    ).fetchall()
    ids = [r[0] for r in rows]
    names = [r[1] for r in rows]
    vectors = np.array(
        [np.fromstring(r[2].strip("[]"), sep=",", dtype=np.float32) for r in rows],
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, names, vectors / norms


def find_pairs(ids, names, unit_vectors, threshold: float, top: int) -> list[dict]:
    sims = unit_vectors @ unit_vectors.T
    np.fill_diagonal(sims, 0.0)
    upper = np.triu(sims)
    idx = np.argwhere(upper >= threshold)
    pairs = [
        {"i": int(i), "j": int(j), "cosine": round(float(sims[i, j]), 4)}
        for i, j in idx
    ]
    pairs.sort(key=lambda p: -p["cosine"])
    return pairs[:top]


def _strict_key(name: str) -> str:
    """Alphanumerics only, lowercased: punctuation/spacing variants collapse.

    Morphological variants ("методология" / "методологии") stay distinct, and
    so do different skills that only share a stem ("c" / "c#" / "c++").
    """
    return "".join(ch for ch in name.lower() if ch.isalnum())


def safe_merges(conn, pairs: list[dict], ids, names) -> list[tuple[str, str]]:
    """Pairs whose names are identical once punctuation and spacing are removed."""
    out = []
    for p in pairs:
        keep_id, drop_id = ids[p["i"]], ids[p["j"]]
        if _strict_key(names[p["i"]]) == _strict_key(names[p["j"]]):
            out.append((keep_id, drop_id))
    return out


def merge(conn, keep_id: str, drop_id: str) -> None:
    """Repoint foreign keys onto the survivor, dropping rows that would collide."""
    with conn.cursor() as cur:
        cur.execute(
            """
            delete from market_skill_mappings d
            where d.skill_id = %s
              and exists (
                select 1 from market_skill_mappings k
                where k.skill_id = %s
                  and k.period is not distinct from d.period
                  and k.source is not distinct from d.source
              )
            """,
            (drop_id, keep_id),
        )
        cur.execute(
            "update market_skill_mappings set skill_id = %s where skill_id = %s",
            (keep_id, drop_id),
        )
        cur.execute(
            """
            delete from competency_skills d
            where d.skill_id = %s
              and exists (
                select 1 from competency_skills k
                where k.skill_id = %s
                  and k.competency_id = d.competency_id
                  and k.ksa_type = d.ksa_type
                  and coalesce(k.parse_version_id, '00000000-0000-0000-0000-000000000000'::uuid)
                    = coalesce(d.parse_version_id, '00000000-0000-0000-0000-000000000000'::uuid)
              )
            """,
            (drop_id, keep_id),
        )
        cur.execute(
            "update competency_skills set skill_id = %s where skill_id = %s",
            (keep_id, drop_id),
        )
        cur.execute("delete from skills where id = %s", (drop_id,))
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--apply-safe", action="store_true")
    parser.add_argument("--pair", nargs=2, metavar=("KEEP_ID", "DROP_ID"))
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    with psycopg.connect(args.dsn, connect_timeout=15) as conn:
        if args.apply:
            if not args.pair:
                print("--apply requires --pair KEEP_ID DROP_ID")
                return 2
            merge(conn, *args.pair)
            print(f"merged {args.pair[1]} into {args.pair[0]}")
            return 0

        ids, names, unit = load_skills(conn)
        pairs = find_pairs(ids, names, unit, args.threshold, args.top)

        if args.apply_safe:
            targets = safe_merges(conn, pairs, ids, names)
            print(f"safe merges: {len(targets)}")
            for keep_id, drop_id in targets:
                name = names[ids.index(drop_id)]
                try:
                    merge(conn, keep_id, drop_id)
                except Exception as exc:  # keep going, report the failure
                    conn.rollback()
                    print(f"  FAILED {name!r}: {exc}")
                    continue
                print(f"  merged {name!r} -> {names[ids.index(keep_id)]!r}")
            return 0

        links = {}
        for row in conn.execute(
            "select id::text, "
            "(select count(*) from market_skill_mappings m where m.skill_id = s.id), "
            "(select count(*) from competency_skills cs where cs.skill_id = s.id) "
            "from skills s"
        ).fetchall():
            links[row[0]] = (row[1], row[2])

    print(f"skills with embeddings: {len(ids)}")
    print(f"pairs with cosine >= {args.threshold}: {len(pairs)}\n")
    print(f"{'keep':28} {'drop':28} {'cos':>6} {'msm':>10} {'cs':>8}")
    for p in pairs:
        a, b = ids[p["i"]], ids[p["j"]]
        print(
            f"{names[p['i']][:27]:28} {names[p['j']][:27]:28} {p['cosine']:6.3f} "
            f"{str(links.get(a)):>10} {str(links.get(b)):>8}"
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "keep": {"id": ids[p["i"]], "name": names[p["i"]], "links": links.get(ids[p["i"]])},
            "drop": {"id": ids[p["j"]], "name": names[p["j"]], "links": links.get(ids[p["j"]])},
            "cosine": p["cosine"],
        }
        for p in pairs
    ]
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
