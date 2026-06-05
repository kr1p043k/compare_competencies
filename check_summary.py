import json
p = r"C:\Users\максим\workvs\compare_competencies\data\result\teacher\09.03.02\_summary.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)
ds = data.get("disciplines", [])
print(f"Disciplines in summary: {len(ds)}")
for d in ds:
    nm = d["name"][:50]
    lvl = d["coverage_level"]
    cov = d["coverage_ratio"] * 100
    print(f"  {nm} — {lvl} ({cov:.0f}%)")
