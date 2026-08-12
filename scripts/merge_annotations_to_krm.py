"""Слияние реальных компетенций и содержания из аннотаций ЮФУ в KRM-файлы.

Для каждого направления из TARGETS: для каждой дисциплины, которой найдена
аннотация на Yandex Disk, заменяем у неё:
  - competencies  — реальные коды (УК/ОПК/ПК/ВПК) из текста аннотации;
  - ksa           — содержание (темы) аннотации → knowledge;
  - skills        — оставляем старые (связь competency_skills не используется,
                    покрытие считается по ksa_entries).

Файлы перезаписываются (non-clean krm_disciplines_<code>.json).
Использование:
    python scripts/merge_annotations_to_krm.py [--dir data/reference] [--only 09.03.04]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
ANN_ALL_PATH = REFERENCE_DIR / "sfu_annotations_all.json"

DIR_CODES = [
    "01.03.01",
    "01.03.02",
    "02.03.02_och",
    "02.03.03",
    "09.03.01_bim",
    "09.03.02",
    "09.03.04",
]

COMP_RE = re.compile(r"\b(?:УК|ОПК|ПК|ВПК|UK|OPK|PK|VPK)[-\s]?\s*\d+(?:\.\d+)*")

RU = {"UK": "УК", "OPK": "ОПК", "PK": "ПК", "VPK": "ВПК"}


def normalize_comp(code: str) -> str:
    m = re.match(r"\b(УК|ОПК|ПК|ВПК|UK|OPK|PK|VPK)[\s-]*(\d+(?:\.\d+)*)", code, flags=re.I)
    if not m:
        return code
    base = m.group(1).upper()
    return f"{RU.get(base, base)}-{m.group(2)}"


def norm_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"^(аннотация|анн|апп -?|а -)\s*", "", s)
    s = re.sub(r"^\d{2}\.\d{2}\.\d{2}-?(анн)?\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def extract_content(text: str) -> list[str]:
    """Темы из секции «Содержание дисциплины» / «Course contents»."""
    sections = [
        (
            r"\d+\.\s*Содержание\s+дисциплины",
            r"\d+\.\s*(?:Дополнительная|Основные\s+образовательные|Формы\s+контроля|Компетенции|Учебно-методическое)",
        ),
        (r"\d+\.\s*Course\s+contents", r"\d+\.\s*Learning\s+outcomes"),
    ]
    topics: list[str] = []
    for start_pat, end_pat in sections:
        sm = re.search(start_pat, text, flags=re.I)
        if not sm:
            continue
        em = re.search(end_pat, text[sm.end() :], flags=re.I)
        seg = text[sm.end() : sm.end() + em.start()] if em else text[sm.end() :]
        raw = [ln.strip() for ln in seg.splitlines() if len(ln.strip()) >= 3]
        for ln in raw:
            ln = re.sub(r"^\d+[.)]\s*", "", ln).strip()
            if ln and ln not in topics:
                topics.append(ln)
        break
    return topics


def build_ksa(comp_codes: list[str], topics: list[str], old_ksa: dict) -> dict:
    """Темы (knowledge) равномерно по компетенциям + старые KSA актуальных кодов."""
    ksa: dict[str, dict] = {}
    for i, code in enumerate(comp_codes):
        old = old_ksa.get(code, {})
        knowledge = list(old.get("knowledge", []))
        chunk = topics[i :: len(comp_codes)]
        known = set(knowledge)
        for t in chunk:
            if t not in known:
                known.add(t)
                knowledge.append(t)
        ksa[code] = {
            "knowledge": knowledge,
            "abilities": list(old.get("abilities", [])),
            "skills": list(old.get("skills", [])),
        }
    return ksa


def enrich_direction(dir_code: str, anns: dict) -> tuple[int, int, int]:
    krm_path = REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"
    if not krm_path.exists():
        print(f"{dir_code}: KRM-файл не найден, пропуск")
        return 0, 0, 0

    data = json.loads(krm_path.read_text(encoding="utf-8"))
    sub = next(iter(data.values()))
    disc = sub.get("disciplines", {})
    if not isinstance(disc, dict):
        print(f"{dir_code}: нет disciplines")
        return 0, 0, 0

    known = {norm_name(d): d for d in disc}
    matched = has_comp = has_content = 0
    for a in anns:
        if not a.get("ok"):
            continue
        name = norm_name(a.get("name", ""))
        hit = None
        if not name or not a.get("text"):
            continue
        if name in known:
            hit = known[name]
        else:
            for kd, orig in known.items():
                if len(name) >= 5 and len(kd) >= 5 and (kd in name or name in kd):
                    hit = orig
                    break
        if not hit:
            continue
        comps = sorted({normalize_comp(c) for c in a.get("competencies", [])})
        topics = extract_content(a.get("text", ""))
        if not comps:
            comps = disc[hit].get("competencies", [])  # остаёмся на прежних кодах
        if not comps:
            continue
        disc[hit]["competencies"] = comps
        old_ksa = disc[hit].get("ksa", {})
        ksa = build_ksa(comps, topics, old_ksa)
        disc[hit]["ksa"] = ksa if ksa else old_ksa
        if topics:
            has_content += 1
        if a.get("competencies"):
            has_comp += 1
        if "skills" not in disc[hit]:
            disc[hit]["skills"] = {}
        matched += 1

    krm_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{dir_code}: matched={matched} new_comp={has_comp} with_content={has_content}")
    return matched, has_comp, has_content


def main() -> None:
    global REFERENCE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(REFERENCE_DIR))
    parser.add_argument("--only", default=None, help="dir_code; все если не указан")
    args = parser.parse_args()
    REFERENCE_DIR = Path(args.dir)

    anns_all = json.loads(ANN_ALL_PATH.read_text(encoding="utf-8"))
    codes = [args.only] if args.only else DIR_CODES
    for code in codes:
        if args.only and args.only not in DIR_CODES:
            print(f"{args.only}: вне списка покрытых аннотациями направлений")
        enrich_direction(code, anns_all.get(code, []))


if __name__ == "__main__":
    main()
