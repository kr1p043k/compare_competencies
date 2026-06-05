import json
p = "data/reference/krm_disciplines_09.03.02.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)
key = "09.03.02"
d = data.get(key, {})
print("type:", type(data))
print("top keys:", list(data.keys())[:5])
print("disciplines type:", type(d.get("disciplines", {})))
print("disciplines count:", len(d.get("disciplines", {})))
discs = d.get("disciplines", {})
if discs:
    for k in sorted(discs.keys())[:5]:
        print(f"  - {k}")
else:
    print("  EMPTY!")
