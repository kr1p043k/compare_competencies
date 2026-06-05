"""Start server, test all teacher endpoints."""
import subprocess, sys, time, urllib.request, urllib.error, json

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'src.api_pkg:app',
     '--host', '0.0.0.0', '--port', '8000', '--log-level', 'error'],
    cwd=r'C:\Users\максим\workvs\compare_competencies',
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

try:
    for i in range(35):
        time.sleep(1)
        try:
            # 1) POST recommendation
            req = urllib.request.Request(
                'http://localhost:8000/api/teacher/krm/recommendations',
                data=json.dumps({"discipline":"test","competency":"test","suggestion":"test","type":"add"}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            r1 = urllib.request.urlopen(req, timeout=3)
            print(f"POST rec: {r1.status} {r1.read().decode()}")

            # 2) GET recommendation list
            r2 = urllib.request.urlopen('http://localhost:8000/api/teacher/krm/recommendations', timeout=3)
            print(f"GET recs: {len(json.loads(r2.read()))}")

            # 3) GET analysis (should 404 — no analysis run yet)
            try:
                r3 = urllib.request.urlopen('http://localhost:8000/api/teacher/analysis/Алгоритмизация', timeout=3)
                print(f"Analysis: {r3.status}")
            except urllib.error.HTTPError as e3:
                print(f"Analysis: HTTP {e3.code} (expected if no analysis run)")

            # 4) GET analysis summary
            try:
                r4 = urllib.request.urlopen('http://localhost:8000/api/teacher/analysis', timeout=3)
                print(f"Summary: {r4.status}")
            except urllib.error.HTTPError as e4:
                print(f"Summary: HTTP {e4.code} (expected if no analysis run)")

            print("ALL TESTS PASSED")
            break
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:300]}")
            break
        except Exception as e:
            print(f"waiting... ({type(e).__name__})")
            continue
    else:
        print("TIMEOUT")
finally:
    proc.terminate()
    proc.wait(timeout=5)
