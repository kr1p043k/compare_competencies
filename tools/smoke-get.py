"""Smoke: пройти по всем GET-эндпоинтам живого API и найти 5xx.

Запуск (нужен поднятый uvicorn на :8000 и рабочая БД):
    python tools/smoke-get.py
    python tools/smoke-get.py --base http://127.0.0.1:8000

Не входит в pytest: эндпоинты ходят в БД, и TestClient на них зависает
(проверено на 30-минутном прогоне). Зато этот скрипт поймал реальный дефект:
500 на /teacher/export/vacancies из-за UnboundLocalError (локальный импорт
HTTPException внутри функции делал имя локальным для всей функции).
"""
import argparse
import os
import time

import requests

SKIP = {"/docs", "/redoc", "/openapi.json", "/metrics", "/docs/oauth2-redirect"}


def login(base: str, email: str, password: str, attempts: int = 40) -> str | None:
    for _ in range(attempts):
        try:
            r = requests.post(
                f"{base}/api/auth/login",
                json={"email": email, "password": password},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json().get("token")
        except Exception:
            pass
        time.sleep(3)
    return None


def wait_health(base: str, attempts: int = 60) -> bool:
    for _ in range(attempts):
        try:
            if requests.get(f"{base}/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--email", default=os.environ.get("SMOKE_EMAIL", ""))
    ap.add_argument("--password", default=os.environ.get("SMOKE_PASSWORD", ""))
    ap.add_argument("--timeout", type=int, default=40)
    args = ap.parse_args()

    # Учётные данные не храним в репозитории: только аргументы или окружение.
    if not args.email or not args.password:
        print("Укажите --email/--password или SMOKE_EMAIL/SMOKE_PASSWORD")
        return 2

    if not wait_health(args.base):
        print("API не поднялся")
        return 2

    token = login(args.base, args.email, args.password)
    if not token:
        print("login failed")
        return 2
    headers = {"Authorization": f"Bearer {token}"}

    spec = requests.get(f"{args.base}/openapi.json", timeout=60).json()
    paths = sorted(
        p for p, ops in spec.get("paths", {}).items()
        if "get" in ops and "{" not in p and p not in SKIP
    )
    print(f"GET без path-параметров: {len(paths)}")

    bad = []
    for p in paths:
        try:
            code = requests.get(f"{args.base}{p}", headers=headers, timeout=args.timeout).status_code
        except Exception as exc:
            print(f"  skip {p}: {type(exc).__name__}")
            continue
        if code >= 500:
            bad.append((p, code))
            print(f"  5xx: {p} {code}")

    print(f"\n5xx найдено: {len(bad)}")
    for p, c in bad:
        print(f"  {c} {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
