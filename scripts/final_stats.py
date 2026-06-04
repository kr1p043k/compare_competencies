import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("data/reference/krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

d = data["09.03.02"]["disciplines"]
print(f"Disciplines: {len(d)}")
total_sk = sum(len(v) for di in d.values() for v in di["skills"].values())
print(f"Total skills: {total_sk}")
print()

for name, info in sorted(d.items()):
    sc = sum(len(v) for v in info["skills"].values())
    print(f"  {name}: {len(info['competencies'])} comps, {sc} skills")

print("\nMissing (in rpd_pdfs but not in JSON):")
existing = set(d.keys())
for fname in sorted(os.listdir("temp/rpd_pdfs")):
    if fname.endswith(".pdf"):
        clean = fname.replace(".pdf", "").replace("РПД_", "").strip()
        if clean not in existing:
            print(f"  {clean}")

# Show a sample of clean data
print("\n\nSample: Базы данных и СУБД -> ОПК-2 skills:")
bd = d.get("Базы данных и СУБД", {})
for s in bd.get("skills", {}).get("ОПК-2", [])[:5]:
    print(f"  - {s[:100]}")
