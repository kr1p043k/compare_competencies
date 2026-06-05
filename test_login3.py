import urllib.request, urllib.error, json

req = urllib.request.Request(
    "http://localhost:8000/api/auth/login",
    data=json.dumps({"email": "teacher@compare-competencies.local", "password": "teacher123"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    resp = urllib.request.urlopen(req, timeout=10)
    print(resp.status, resp.read().decode()[:500])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:800]}")
