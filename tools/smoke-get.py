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


def login(base: str, email: str, password: str, attempts: int = 40,
          totp_secret: str = "") -> str | None:
    """Вход с учётом 2FA.

    Ответы /auth/login при включённом мастер-переключателе:
      * token - обычный вход;
      * {status: totp_required, challenge_id} - нужен код из TOTP
        (totp_secret берётся из TOTP_SECRET или --totp-secret);
      * {status: totp_setup_required, enroll_id} - админ ещё не подключил
        TOTP. Настраиваем автоматически: setup -> confirm, подтверждение
        сразу отдаёт токен.
    """
    for _ in range(attempts):
        try:
            r = requests.post(
                f"{base}/api/auth/login",
                json={"email": email, "password": password},
                timeout=10,
            )
            if r.status_code == 200:
                body = r.json()
                if body.get("token"):
                    return body["token"]
                if body.get("status") == "totp_required":
                    if not totp_secret:
                        print(
                            "Нужен TOTP-код: укажите --totp-secret или TOTP_SECRET"
                        )
                        return None
                    import pyotp

                    v = requests.post(
                        f"{base}/api/auth/2fa/verify",
                        json={
                            "challenge_id": body["challenge_id"],
                            "code": pyotp.TOTP(totp_secret).now(),
                        },
                        timeout=10,
                    )
                    if v.status_code == 200:
                        return v.json().get("token")
                    print(f"2fa verify: {v.status_code} {v.text[:200]}")
                    return None
                if body.get("status") == "totp_setup_required":
                    # Смоук не должен молча падать: админ ещё не подключил
                    # второй фактор, настраиваем его тем же путём, что человек.
                    return _enroll_and_login(base, body.get("enroll_id"))
        except Exception:
            pass
        time.sleep(3)
    return None


def _enroll_and_login(base: str, enroll_id: str | None) -> str | None:
    """Первичная настройка TOTP по одноразовому пропуску из /auth/login."""
    if not enroll_id:
        print("2fa: сервер требует настройку, но не выдал enroll_id")
        return None
    try:
        import pyotp
    except ImportError:
        print("2fa: нужен pyotp для автонастройки (pip install pyotp)")
        return None

    setup = requests.post(
        f"{base}/api/auth/2fa/setup", json={"enroll_id": enroll_id}, timeout=10
    )
    if setup.status_code != 200:
        print(f"2fa setup: {setup.status_code} {setup.text[:200]}")
        return None
    secret = setup.json().get("secret")
    if not secret:
        print("2fa setup: в ответе нет secret")
        return None

    confirm = requests.post(
        f"{base}/api/auth/2fa/confirm",
        json={"code": pyotp.TOTP(secret).now(), "enroll_id": enroll_id},
        timeout=10,
    )
    if confirm.status_code != 200:
        print(f"2fa confirm: {confirm.status_code} {confirm.text[:200]}")
        return None
    print("2fa: админ подключён автоматически, второй фактор включён")
    return confirm.json().get("token")


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
    ap.add_argument("--warmup-retry", type=float, default=5.0)
    ap.add_argument("--totp-secret", default=os.environ.get("TOTP_SECRET", ""))
    args = ap.parse_args()

    # Учётные данные не храним в репозитории: только аргументы или окружение.
    if not args.email or not args.password:
        print("Укажите --email/--password или SMOKE_EMAIL/SMOKE_PASSWORD")
        return 2

    if not wait_health(args.base):
        print("API не поднялся")
        return 2

    token = login(args.base, args.email, args.password, totp_secret=args.totp_secret)
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
        if code == 503:
            # 503 бывает и «честным» (интеграция не настроена), и временным:
            # зависимости (эмбеддинги, кластеры) поднимаются после старта.
            # Даём второй шанс, иначе первый же прогрев даёт ложные находки.
            time.sleep(args.warmup_retry)
            try:
                code = requests.get(
                    f"{args.base}{p}", headers=headers, timeout=args.timeout
                ).status_code
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
