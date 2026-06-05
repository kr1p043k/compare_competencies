import os, sys

# Reset sys.path to have compare_competencies ONLY
new_path = [r"C:\Users\максим\workvs\compare_competencies"]
for p in sys.path:
    if "forcode" not in p.lower() and "forCode" not in p:
        new_path.append(p)
sys.path = new_path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from src.api_pkg.routers import teacher as teacher_mod
print(f"Module file: {teacher_mod.__file__}")
print(f"PyCache file: {teacher_mod.__cached__}")

# Check the actual route registration
for route in teacher_mod.router.routes:
    methods = getattr(route, "methods", None)
    path = getattr(route, "path", "")
    if "recommend" in path.lower() or "rec_" in path:
        import inspect
        sig = inspect.signature(route.endpoint)
        print(f"  {methods} {path} -> sig={sig}")
        if hasattr(route.endpoint, "__wrapped__"):
            print(f"    WRAPPED! original={route.endpoint.__wrapped__.__name__}")
