"""Слияние дисциплин-аббревиатур в KRM с их официальными названиями.

В KRM рядом с официальным именем может стоять его сокращённая форма:
    «Инструменты и инженерные практики разработки систем ИИ»
    «Инструменты и инженерные практики разработки систем искусственного интеллекта»

Это два ключа одной дисциплины. Данные (competencies/skills/ksa) размазаны по
обоим: у аббревиатуры может лежать 12 skills ПК-18, у официального имени — 2.
Удалять аббревиатуру нельзя (потеряются коды), поэтому записи СЛИВАЮТСЯ в
официальное имя: union competencies, union skills по кодам, union ksa по кодам.

Использование:
    python scripts/merge_krm_aliases.py --direction 09.03.02 --year 2025 [--apply]
Без --apply только показывает план (dry-run).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"

ABBREV = {
    "ии": "искусственного интеллекта",
    "по": "программного обеспечения",
    "субд": "систем управления базами данных",
    "ос": "операционные системы",
    "исок": "информационные системы",
}

THRESHOLD = 0.86


def _norm(name: str) -> str:
    """Нормализация названия: регистр, ё→е, пунктуация, аббревиатуры.

    Всё, что не буква/цифра, становится пробелом, поэтому 'AI_ML' и 'AI/ML',
    '... РПД.pdf' и '...' сходятся к одной форме.
    """
    s = name.lower().replace("ё", "е")
    s = re.sub(r"[^0-9a-zа-я]+", " ", s)
    tokens = [ABBREV.get(t, t) for t in s.split()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def _merge(alias: dict, target: dict) -> None:
    """Слить alias в target: union компетенций, skills и ksa по кодам.

    skills: {код: [тексты]}, ksa: {код: {тип: [тексты]}} - структуры разные,
    union делается по-разному (иначе ksa превращался в список и сид падал).
    """
    target["competencies"] = sorted(set(target.get("competencies") or [])
                                    | set(alias.get("competencies") or []))
    cur_sk = target.setdefault("skills", {}) or {}
    target["skills"] = cur_sk
    for code, items in (alias.get("skills") or {}).items():
        lst = list(cur_sk.get(code) or [])
        for it in items or []:
            if it not in lst:
                lst.append(it)
        cur_sk[code] = lst
    cur_ksa = target.setdefault("ksa", {}) or {}
    target["ksa"] = cur_ksa
    for code, types in (alias.get("ksa") or {}).items():
        if not isinstance(types, dict):
            continue
        tgt_types = cur_ksa.setdefault(code, {})
        for kt, texts in types.items():
            lst = list(tgt_types.get(kt) or [])
            for t in texts or []:
                if t not in lst:
                    lst.append(t)
            tgt_types[kt] = lst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direction", default="09.03.02")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--apply", action="store_true", help="записать изменения в KRM")
    args = p.parse_args()

    off_path = REFERENCE_DIR / f"official_curriculum_{args.direction}_{args.year}.json"
    krm_path = REFERENCE_DIR / f"krm_disciplines_{args.direction}.json"
    off = json.loads(off_path.read_text(encoding="utf-8"))
    official = [d["name"] if isinstance(d, dict) else d for d in off["disciplines"]]
    off_norm = {_norm(n): n for n in official}

    data = json.loads(krm_path.read_text(encoding="utf-8"))
    sub = next(iter(data.values()))
    discs = sub["disciplines"]

    plan: list[tuple[str, str, float]] = []
    for name in list(discs):
        nn = _norm(name)
        if nn in off_norm:
            # То же имя после нормализации, но написание иное (аббревиатура ИИ,
            # артефакт OCR) - сливаем в официальное написание.
            if off_norm[nn] != name:
                plan.append((name, off_norm[nn], 1.0))
            continue
        best, score = None, 0.0
        for on in off_norm:
            r = fuzz.ratio(nn, on) / 100.0
            if r > score:
                best, score = on, r
        if best and score >= THRESHOLD:
            plan.append((name, off_norm[best], score))

    print(f"{args.direction} набор {args.year}: официальных {len(official)}, "
          f"KRM {len(discs)}, к слиянию {len(plan)}")
    for alias, target, score in plan:
        print(f"  {score:.2f}  {alias!r}\n        -> {target!r}")

    if not args.apply:
        print("\n(dry-run; добавьте --apply)")
        return 0

    redirect: dict[str, str] = {}

    def resolve(t: str) -> str:
        seen = set()
        while t in redirect and t not in seen:
            seen.add(t)
            t = redirect[t]
        return t

    merged_n = 0
    renamed_n = 0
    for _pass in range(6):
        changed = 0
        for alias, target, _score in plan:
            if alias not in discs:
                continue
            tgt = resolve(target)
            if alias == tgt:
                continue
            if tgt in discs:
                _merge(discs[alias], discs[tgt])
                del discs[alias]
                redirect[alias] = tgt
                merged_n += 1
            else:
                # официального написания в KRM нет - просто переименовываем
                # алиас в него, сохраняя содержимое.
                discs[tgt] = discs.pop(alias)
                redirect[alias] = tgt
                renamed_n += 1
            changed += 1
        if not changed:
            break
    krm_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nслито {merged_n}, переименовано {renamed_n}; в KRM осталось {len(discs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
