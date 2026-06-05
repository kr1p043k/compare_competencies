import os

base = r'C:\Users\максим\workvs\compare_competencies'

checks = [
    ("src/config.py", ["KRM_DISCIPLINES_PATH", "TEACHER_RECOMMENDATIONS_PATH"]),
    ("src/api_pkg/routers/auth.py", ["get_pool", "asyncpg", "fetchrow", "crypt"]),
    ("src/api_pkg/startup.py", ["create_pool", "asyncpg pool ready"]),
    ("src/api_pkg/request_logger.py", ["replace(tzinfo=None)"]),
    ("data/reference/skill_types.json", ["academic", "professional"]),
    ("src/predictors/curriculum_recommender.py", ["foundational", "_filter_relevant"]),
    ("src/models/teacher_analysis.py", ["foundational"]),
    ("frontend/src/app/components/TeacherDashboard.tsx", ["KRM_API", "encodeURIComponent"]),
    ("users.json", ["teacher@compare-competencies.local", "teacher123"]),
]

all_ok = True
for path, keywords in checks:
    full = os.path.join(base, path)
    if not os.path.exists(full):
        print(f"FAIL: {path} — file not found")
        all_ok = False
        continue
    with open(full, "r", encoding="utf-8") as f:
        content = f.read()
    missing = [kw for kw in keywords if kw not in content]
    if missing:
        print(f"FAIL: {path} — missing: {missing}")
        all_ok = False
    else:
        print(f"OK: {path}")

print()
print("ALL OK" if all_ok else "SOME FAILURES")
