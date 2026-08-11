"""Нормализация названий дисциплин в KRM-файлах.

Правила:
  - Название трогаем только если вне скобок есть латинские слова.
  - Если есть русская группа в скобках (перевод) -> берём её.
  - Иначе переводим через YandexGPT (при --no-llm оставляем как есть).
  - При совпадении ключей (дубль дисциплины на англ. и рус.) записи объединяются.

Использование:
    python scripts/normalize_krm_names.py [--dir data/reference] [--dry-run] [--no-llm]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUS_RE = re.compile(r"[А-Яа-яЁё]")
LAT_WORD_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{3,}(?![A-Za-z])")

YC_API_KEY = ""
YC_FOLDER_ID = ""


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _top_level_groups(s: str) -> list[str]:
    """Возвращает содержимое несбалансированных групп (…), включая вложенные скобки."""
    groups: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(s[start:i])
                start = -1
    return groups


def _outside_groups(s: str) -> str:
    """Текст вне всех скобочных групп."""
    parts: list[str] = []
    depth = 0
    buf = []
    for ch in s:
        if ch == "(":
            if depth == 0:
                parts.append("".join(buf))
                buf = []
            depth += 1
        elif ch == ")":
            depth -= 1
        else:
            if depth == 0:
                buf.append(ch)
    parts.append("".join(buf))
    return " ".join(p for p in parts if p.strip())


def extract_ru_group(name: str) -> str | None:
    """Первая скобочная группа, содержащая русские буквы."""
    for group in _top_level_groups(name):
        if RUS_RE.search(group):
            return group
    return None


def has_lat_words(s: str) -> bool:
    return bool(LAT_WORD_RE.search(s))


def yandex_translate(name: str) -> str | None:
    global YC_API_KEY, YC_FOLDER_ID
    if not YC_API_KEY or not YC_FOLDER_ID:
        return None
    body = {
        "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {"temperature": 0.1, "maxTokens": 60},
        "messages": [
            {"role": "system", "text": (
                "Ты переводишь названия учебных дисциплин (курсов) университета с английского "
                "на русский. Не транслитерируй. Не добавляй пояснений и кавычек. "
                "Верни только перевод одним названием."
            )},
            {"role": "user", "text": name},
        ],
    }
    req = urllib.request.Request(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {YC_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        alt = data.get("result", {}).get("alternatives", [])
        if not alt:
            return None
        text = alt[0].get("message", {}).get("text", "").strip()
        return text or None
    except Exception as exc:
        print(f"  [llm-error] {name!r}: {exc}", file=sys.stderr)
        return None


def normalize_name(name: str, use_llm: bool) -> str:
    outside = _outside_groups(name)
    ru_in_outside = bool(RUS_RE.search(outside))
    lat_in_outside = has_lat_words(outside)

    if not lat_in_outside:
        return name
    # Уже русское название (с англицизмом, напр. "Low-code прототипирование") — не трогаем.
    if ru_in_outside:
        return name

    ru_group = extract_ru_group(name)
    if ru_group is not None:
        return ru_group

    if use_llm:
        translated = yandex_translate(name)
        if translated:
            return translated
    return name


def merge_discipline(existing: dict, new: dict) -> dict:
    """Объединяет две записи о дисциплине (одна и та же дисциплина на англ. и рус.)."""
    merged = dict(existing)
    comps = list(existing.get("competencies", []))
    for c in new.get("competencies", []):
        if c not in comps:
            comps.append(c)
    merged["competencies"] = comps

    skills = dict(existing.get("skills", {}))
    for c, slist in (new.get("skills", {}) or {}).items():
        cur = list(skills.get(c, []))
        for s in slist:
            if s not in cur:
                cur.append(s)
        skills[c] = cur
    merged["skills"] = skills

    ksa = dict(existing.get("ksa", {}))
    for c, types in (new.get("ksa", {}) or {}).items():
        cur_types = dict(ksa.get(c, {}))
        for kt, texts in (types or {}).items():
            cur_list = list(cur_types.get(kt, []))
            for t in texts:
                if t not in cur_list:
                    cur_list.append(t)
            cur_types[kt] = cur_list
        ksa[c] = cur_types
    merged["ksa"] = ksa
    return merged


def process_file(path: Path, use_llm: bool, dry_run: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed_any = False
    for dir_code, info in data.items():
        if not isinstance(info, dict):
            continue
        profile = info.get("profile", "")
        new_profile = normalize_name(profile, use_llm)
        if new_profile != profile:
            print(f"  {dir_code}: profile {profile!r} -> {new_profile!r}")
            info["profile"] = new_profile
            changed_any = True

        disciplines = info.get("disciplines", {})
        for old_name in list(disciplines.keys()):
            new_name = normalize_name(old_name, use_llm)
            if new_name == old_name:
                continue
            print(f"  {dir_code}: {old_name!r} -> {new_name!r}")
            entry = disciplines.pop(old_name)
            if new_name in disciplines:
                print(f"    merged into existing key")
                disciplines[new_name] = merge_discipline(disciplines[new_name], entry)
            else:
                disciplines[new_name] = entry
            changed_any = True

    if changed_any and not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'[dry-run]' if dry_run else '[updated]'} {path.name}")


def main() -> None:
    global YC_API_KEY, YC_FOLDER_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/reference")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    env = _load_env(Path(".env"))
    YC_API_KEY = env.get("YC_API_KEY", "")
    YC_FOLDER_ID = env.get("YC_FOLDER_ID", "")

    if not YC_API_KEY and not args.no_llm:
        print("YC_API_KEY not found in .env; LLM disabled (run with --no-llm to allow)", file=sys.stderr)
        args.no_llm = True

    base = Path(args.dir)
    for path in sorted(base.glob("krm_disciplines_*.json")):
        if "_clean" in path.name:
            continue
        process_file(path, use_llm=not args.no_llm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
