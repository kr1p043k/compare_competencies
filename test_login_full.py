"""Start uvicorn, wait for ready, test login, report."""
import subprocess, sys, time, urllib.request, urllib.error, json

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'src.api_pkg:app',
     '--host', '0.0.0.0', '--port', '8000', '--log-level', 'error'],
    cwd=r'C:\Users\максим\workvs\compare_competencies',
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

try:
    # Wait up to 45s for server to come up
    for i in range(45):
        time.sleep(1)
        try:
            req = urllib.request.Request(
                'http://localhost:8000/api/auth/login',
                data=json.dumps({"email": "admin@compare-competencies.local", "password": "admin"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=3)
            print(f"OK: {resp.status}")
            print(resp.read().decode()[:500])
            sys.exit(0)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            print(f"HTTP {e.code}: {body}")
            sys.exit(0)
        except (urllib.error.URLError, ConnectionRefusedError):
            continue
    print("TIMEOUT: server never started")
    sys.exit(1)
finally:
    proc.terminate()
    proc.wait(timeout=5)
