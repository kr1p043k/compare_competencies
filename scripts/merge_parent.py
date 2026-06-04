import json

with open("data/reference/krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

d = data["09.03.02"]["disciplines"]

# Merge indicator skills into parent competencies
# E.g., ОПК-2.2 skills -> also add to ОПК-2
new_disciplines = {}
for dname, info in d.items():
    comp_skills = {}
    for comp in info["competencies"]:
        skills = info["skills"].get(comp, [])
        comp_skills[comp] = list(skills)

    # Merge child indicators into parent
    for comp in info["competencies"]:
        if "." in comp:
            parent = comp.rsplit(".", 1)[0]
            if parent in comp_skills and parent != comp:
                child_skills = info["skills"].get(comp, [])
                for s in child_skills:
                    if s not in comp_skills[parent]:
                        comp_skills[parent].append(s)

    new_info = {
        "competencies": info["competencies"],
        "skills": comp_skills
    }
    new_disciplines[dname] = new_info

data["09.03.02"]["disciplines"] = new_disciplines

with open("data/reference/krm_disciplines_09.03.02.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Stats
total_skills = sum(len(v) for disc in new_disciplines.values() for v in disc["skills"].values())
zeros = [(n, i) for n, i in new_disciplines.items() if not any(i["skills"].values())]
print(f"Total: {len(new_disciplines)} disciplines, {total_skills} skills")
print(f"Disciplines with 0 skills: {len(zeros)}")
for n, i in zeros:
    print(f"  {n}: {i['competencies']}")

# Show sample
bd = new_disciplines.get("Базы данных и СУБД", {})
if bd:
    print("\nSample: Базы данных и СУБД")
    for comp in ["ОПК-2", "ОПК-4"]:
        sk = bd["skills"].get(comp, [])
        print(f"  {comp}: {len(sk)} skills")
        for s in sk[:3]:
            print(f"    - {s[:100]}")
