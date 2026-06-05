"""Read OПОП to find target profession."""
import json
with open(r"C:\Users\291A~1\workvs\COMPAR~1\data\reference\krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

discs = data["09.03.02"]["disciplines"]

# Check ОПОП discipline
opop = discs.get("ОПОП 09.03.02 Перспективные информационные технологии", {})
print("ОПОП competencies:")
for c in opop.get("competencies", [])[:10]:
    print(f"  {c}")
print(f"\nОПОП skills: {sum(len(s) for s in opop.get('skills', {}).values())}")

# List what the loader knows about direction
print(f"\nDirection: {data['09.03.02'].get('direction_name', 'N/A')}")
print(f"Profile: {data['09.03.02'].get('profile', 'N/A')}")

# Check ГИА
gia = discs.get("ИИ 09.03.02", {})
print(f"\nГИА competencies: {gia.get('competencies', [])}")
print(f"ГИА skills count: {sum(len(s) for s in gia.get('skills', {}).values())}")
