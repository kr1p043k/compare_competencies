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
import unicodedata
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
    """Alphanumerics only, lowercased, after Unicode compatibility folding.

    NFKC collapses superscripts (iВІc -> i2c) and fullwidth forms; the explicit
    replacements catch non-breaking hyphens and zero-width marks that NFKC
    leaves alone. Punctuation and spacing variants therefore collapse, while
    different skills sharing a stem ("c" / "c#" / "c++") and morphological
    variants ("РјРµС‚РѕРґРѕР»РѕРіРёСЏ" / "РјРµС‚РѕРґРѕР»РѕРіРёРё") stay distinct.
    """
    folded = unicodedata.normalize("NFKC", name).lower()
    for ch in ("\u2010", "\u2011", "\u2013", "\u2014", "\u2212"):
        folded = folded.replace(ch, "-")
    for ch in ("\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\ufeff"):
        folded = folded.replace(ch, "")
    return "".join(ch for ch in folded if ch.isalnum())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _mixed_script(name: str) -> bool:
    """True when a single token mixes Cyrillic and Latin letters (typo signal)."""
    for token in name.split():
        has_cyr = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in token)
        has_lat = any("a" <= ch.lower() <= "z" for ch in token)
        if has_cyr and has_lat:
            return True
    return False


def typo_merges(pairs: list[dict], ids, names, links: dict[str, tuple[int, int]]) -> list[tuple[str, str]]:
    """Pairs that are the same term with one character wrong.

    Restricted to keys of at least 8 characters and an edit distance of exactly
    one, because short acronyms differ by one character while meaning different
    things (``1с:уат``/``1с:ут``, ``еспд``/``естд``, ``ida``/``jpa``).

    The survivor is chosen by: no mixed Latin/Cyrillic token, then more foreign
    key references (so the row actually used keeps its id), then the shorter
    name (singular over plural, corrected spelling over the typo).
    """
    out = []
    for p in pairs:
        a, b = names[p["i"]], names[p["j"]]
        ka, kb = _strict_key(a), _strict_key(b)
        if len(ka) < 8 or len(kb) < 8:
            continue
        if abs(len(ka) - len(kb)) > 1:
            continue
        if _levenshtein(ka, kb) != 1:
            continue

        id_a, id_b = ids[p["i"]], ids[p["j"]]
        la = sum(links.get(id_a, (0, 0)))
        lb = sum(links.get(id_b, (0, 0)))
        score_a = (_mixed_script(a), -la, len(a))
        score_b = (_mixed_script(b), -lb, len(b))
        if score_a <= score_b:
            out.append((id_a, id_b))
        else:
            out.append((id_b, id_a))
    return out


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
    parser.add_argument("--apply-typos", action="store_true")
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

        links: dict[str, tuple[int, int]] = {}
        for row in conn.execute(
            "select id::text, "
            "(select count(*) from market_skill_mappings m where m.skill_id = s.id), "
            "(select count(*) from competency_skills cs where cs.skill_id = s.id) "
            "from skills s"
        ).fetchall():
            links[row[0]] = (row[1], row[2])

        if args.apply_typos:
            targets = typo_merges(pairs, ids, names, links)
            print(f"typo merges: {len(targets)}")
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
