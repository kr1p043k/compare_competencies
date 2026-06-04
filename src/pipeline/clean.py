"""Clean RPD skills: extract parseable skills matching it_skills taxonomy."""

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rapidfuzz import fuzz
from src.parsing.utils import load_it_skills
from src.parsing.skills.skill_validator import SkillValidator

it_skills = load_it_skills()
it_set = {s.strip().lower() for s in it_skills if s.strip()}

rpd_skills_path = os.path.join(os.path.dirname(__file__), "../../data/reference/rpd_skills.json")
rpd_set = set()
if os.path.exists(rpd_skills_path):
    rpd_skills = json.load(open(rpd_skills_path, "r", encoding="utf-8"))
    rpd_set = {s.strip().lower() for s in rpd_skills if s.strip()}
    print(f"it_skills: {len(it_set)}, rpd_skills: {len(rpd_set)}")

combined_set = it_set | rpd_set
combined_list = sorted(combined_set, key=len, reverse=True)

validator = SkillValidator(whitelist=combined_set, max_length=80, max_words=10)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" ,;.:-\"'«»()")


def find_skills(text: str) -> set[str]:
    tl = norm(text)
    if len(tl) < 3:
        return set()
    found = set()
    text_tokens = set(tl.split())
    for entry in combined_list:
        if entry in tl:
            found.add(entry)
            continue
        if any(c in entry for c in ("-", "(", ")")):
            if entry in tl:
                found.add(entry)
                continue
        entry_tokens = set(entry.split())
        overlap = entry_tokens & text_tokens
        if len(overlap) >= len(entry_tokens):
            found.add(entry)
            continue
        if len(entry_tokens) >= 2:
            ratio = fuzz.token_set_ratio(entry, tl)
            if ratio >= 85:
                found.add(entry)
        elif len(entry_tokens) == 1:
            ratio = fuzz.partial_ratio(entry, tl)
            if ratio >= 92:
                found.add(entry)
    return found


def clean_competency_skills(data: dict) -> dict:
    for disc_name, disc in data.get("disciplines", {}).items():
        for comp_code in disc.get("competencies", []):
            ksa = disc.get("ksa", {}).get(comp_code, {})
            skills_flat = set()
            for kt in ("knowledge", "abilities", "skills"):
                raw_list = ksa.get(kt, [])
                for text in raw_list:
                    skills_flat |= find_skills(text)
            disc.setdefault("skills", {})[comp_code] = sorted(skills_flat)
    return data


if __name__ == "__main__":
    import sys
    krm_path = sys.argv[1] if len(sys.argv) > 1 else "data/reference/krm_disciplines_09.03.02.json"
    with open(krm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = clean_competency_skills(data)
    with open(krm_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Cleaned: {krm_path}")
