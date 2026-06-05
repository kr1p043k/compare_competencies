"""Quick check of extracted RPD data quality."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

with open(r"C:\Users\291A~1\workvs\COMPAR~1\data\reference\krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

discs = data["09.03.02"]["disciplines"]
print(f"Дисциплин: {len(discs)}")

all_skills = []
for name, d in list(discs.items())[:5]:
    comp_skills = []
    for comp, skills in d["skills"].items():
        comp_skills.extend(skills)
    all_skills.extend(comp_skills)
    print(f"\n  {name}:")
    print(f"    Компетенций: {len(d['competencies'])}")
    print(f"    Навыков: {len(comp_skills)}")
    if comp_skills:
        print(f"    Первые 5: {comp_skills[:5]}")

print(f"\nУникальных навыков (первые 50):")
unique = list(set(
    s for d in discs.values()
    for sk in d["skills"].values()
    for s in sk
))
print("  " + "\n  ".join(unique[:50]))
