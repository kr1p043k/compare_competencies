"""Test with PYTHONDONTWRITEBYTECODE set in the subprocess."""
import subprocess, sys, time, http.client, json
import os

env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"

proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'src.api_pkg:app',
     '--host', '0.0.0.0', '--port', '8000', '--log-level', 'error'],
    cwd=r'C:\Users\максим\workvs\compare_competencies',
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env=env,
)
try:
    for i in range(35):
        time.sleep(1)
        if proc.poll() is not None:
            break
        try:
            conn = http.client.HTTPConnection('localhost', 8000, timeout=3)
            conn.request('POST', '/api/teacher/krm/rec_add',
                body=json.dumps({"discipline":"t","competency":"t","suggestion":"t","type":"add"}),
                headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            body = r.read().decode()[:300]
            conn.close()
            print(f"POST /rec_add: {r.status} {body}")
            break
        except Exception:
            continue
    else:
        print("TIMEOUT")
finally:
    if proc.poll() is None:
        proc.terminate()
        proc.wait(5)
