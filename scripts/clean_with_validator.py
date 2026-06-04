"""Clean RPD skills: extract parseable skills matching it_skills taxonomy"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rapidfuzz import fuzz
from src.parsing.utils import load_it_skills
from src.parsing.skills.skill_validator import SkillValidator
from src import Ok

it_skills = load_it_skills()
it_set = {s.strip().lower() for s in it_skills if s.strip()}

# Also load RPD-specific skills (complementary taxonomy)
rpd_skills_path = "data/reference/rpd_skills.json"
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
        # 1. Direct substring match — exact, no false positives
        if entry in tl:
            found.add(entry)
            continue

        entry_tokens = entry.split()

        # For single-token fuzzy matches: require very high score (>=92) to avoid
        # false positives like electron <-> selection, jest <-> тестирование
        if len(entry_tokens) == 1:
            score_pr = fuzz.partial_ratio(entry, tl)
            if score_pr >= 92:
                found.add(entry)
            continue

        # For multi-token: each token must appear literally or via stem
        text_stems = {t[:5] for t in text_tokens if len(t) > 4}
        tokens_ok = 0
        for tok in entry_tokens:
            if tok in text_tokens:
                tokens_ok += 1
            elif len(tok) > 4:
                tok_stem = tok[:5]
                if any(ts.startswith(tok_stem) for ts in text_stems):
                    tokens_ok += 1
        if tokens_ok < max(1, len(entry_tokens) * 0.6):
            continue

        # 3. Partial ratio — handles Russian case forms
        score_pr = fuzz.partial_ratio(entry, tl)
        if score_pr >= 85:
            found.add(entry)
            continue

        # 4. Token sort ratio — handles word order
        score_ts = fuzz.token_sort_ratio(entry, tl)
        if score_ts >= 85:
            found.add(entry)
            continue

    return found

# --- Process ---
with open("data/reference/krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

disciplines = data["09.03.02"]["disciplines"]
total_before = 0
total_after = 0
stats_by_comp = {}

for dname, disc in list(disciplines.items()):
    for comp in list(disc["skills"].keys()):
        original = disc["skills"].get(comp, [])
        if not original:
            continue
        total_before += len(original)

        cleaned = set()
        for skill_text in original:
            matches = find_skills(skill_text)
            for m in matches:
                r = validator.validate(m)
                if r.is_ok() and r.unwrap().is_valid:
                    cleaned.add(m)

        disc["skills"][comp] = sorted(cleaned)
        total_after += len(cleaned)
        stats_by_comp[f"{dname[:30]}::{comp}"] = (len(original), len(cleaned))

    if "ksa" in disc:
        for comp in list(disc["ksa"].keys()):
            for ksa_key in ["knowledge", "abilities", "skills"]:
                ksa_list = disc["ksa"][comp].get(ksa_key, [])
                if not ksa_list:
                    continue
                cleaned_ksa = set()
                for skill_text in ksa_list:
                    matches = find_skills(skill_text)
                    for m in matches:
                        r = validator.validate(m)
                        if r.is_ok() and r.unwrap().is_valid:
                            cleaned_ksa.add(m)
                disc["ksa"][comp][ksa_key] = sorted(cleaned_ksa)

with open("data/reference/krm_disciplines_09.03.02.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

zeros = [(n, disc) for n, disc in disciplines.items() if not any(disc["skills"].values())]

print(f"Before: {total_before}")
print(f"After: {total_after}")
print(f"Zero-skill: {len(zeros)}")
for n, _ in zeros:
    print(f"  {n}")

# Show top losses
print("\nBiggest losses:")
big_loss = sorted(stats_by_comp.items(), key=lambda x: x[1][0]-x[1][1], reverse=True)[:5]
for key, (before, after) in big_loss:
    print(f"  {key}: {before} -> {after}")

# Sample
for n, disc in disciplines.items():
    if "Python" in n and "науч" in n.lower():
        print(f"\n=== {n} ===")
        for comp in sorted(disc["skills"].keys()):
            skills = disc["skills"][comp]
            print(f"  {comp} ({len(skills)})")
            for s in skills:
                print(f"    - {s}")
        break
