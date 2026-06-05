import importlib.util
spec = importlib.util.spec_from_file_location(
    "src.api_pkg.routers.teacher",
    r"C:\Users\максим\workvs\compare_competencies\src\api_pkg\routers\teacher.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for route in mod.router.routes:
    print(f"  {route.methods} {route.path}")
print()
post_count = sum(1 for r in mod.router.routes if hasattr(r, "methods") and "POST" in r.methods)
print(f"POST routes: {post_count}")
# Check for old rec parameter
import inspect
for r in mod.router.routes:
    if hasattr(r, "methods") and "POST" in r.methods:
        sig = inspect.signature(r.endpoint)
        print(f"  sig: {sig}")
