"""Сбор аннотаций дисциплин ЮФУ с Yandex Disk и сравнение компетенций с KRM.

Аннотации (PDF) лежат в публичных папках Yandex Disk, на которые ссылаются
страницы образовательных программ 2024+ на sfedu.ru. В аннотациях указаны
компетенции (УК/ОПК/ПК/ВПК) и содержание дисциплины.

Команды:
    collect                       скачать аннотации всех программ из TARGETS
    collect 09.03.04              скачать одну программу
    compare                       сравнить компетенции аннотаций с KRM-файлами

Пример:
    python scripts/sfu_annotations.py collect
    python scripts/sfu_annotations.py compare
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

requests.packages.urllib3.disable_warnings()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REFERENCE_DIR = DATA_DIR / "reference"
ANN_DIR = REFERENCE_DIR / "sfu_annotations"
ANN_ALL_PATH = REFERENCE_DIR / "sfu_annotations_all.json"
ANN_VS_KRM_PATH = REFERENCE_DIR / "sfu_annotations_vs_krm.json"

YAPI = "https://cloud-api.yandex.net/v1/disk/public/resources"

# dir_code (имя файла KRM) -> публичный ключ папки Yandex Disk программы 2024.
TARGETS: dict[str, str] = {
    "01.03.01": "https://disk.360.yandex.ru/d/-gfJATrttFPwbg",
    "01.03.02": "https://disk.360.yandex.ru/d/QPtzbpVPv1lTMw",
    "02.03.02_och": "https://disk.360.yandex.ru/d/yjqexfiCZ6a8sg",
    "02.03.03": "https://disk.360.yandex.ru/d/rKbsxvuaWL2QdQ",
    "09.03.01_bim": "https://disk.360.yandex.ru/d/U-06vjxeaCmoUA",
    "09.03.02": "https://disk.360.yandex.ru/d/8bYtJpapYwAhmA",
    "09.03.04": "https://disk.360.yandex.ru/d/590LQ114n5xDrQ",
}


def yandex_list(public_key: str, path: str = "") -> dict | None:
    r = requests.get(YAPI, params={"public_key": public_key, "path": path, "limit": 1000}, timeout=40)
    if r.status_code != 200:
        return None
    return r.json()


def yandex_download(public_key: str, path: str) -> bytes | None:
    r = requests.get(YAPI + "/download", params={"public_key": public_key, "path": path}, timeout=40)
    if r.status_code != 200:
        return None
    href = r.json().get("href")
    if not href:
        return None
    d = requests.get(href, timeout=90)
    return d.content if d.status_code == 200 else None


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def parse_competencies(text: str) -> list[str]:
    """Компетенции вида УК-1, ОПК-2.1, ВПК-3, а также англ. OPK-3 / UK-1."""
    codes = re.findall(r"\b(?:УК|ОПК|ПК|ВПК|UK|OPK|PK|VPK)[-\s]?\s*\d+(?:\.\d+)*", text)
    out: list[str] = []
    seen: set[str] = set()
    for c in codes:
        c = re.sub(r"[\s-]+", "-", c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def title_from_text(text: str, fname: str) -> str:
    """Название дисциплины из шапки аннотации («…учебной дисциплины «X»»)."""
    patterns = [
        r"рабочей\s+программы\s+учебной\s+дисциплины\s*[«\"']?\s*([^»\"'\n]{3,90})",
        r"рабочей\s+программы\s+дисциплины\s*[«\"']?\s*([^»\"'\n]{3,90})",
        r"Course description \(annotation\)[\s\S]{0,40}?\n\s*([^\n]{3,90})",
        r"Аннотация[\s\S]{0,40}?\n\s*([^\n]{3,90})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            t = re.sub(r"\s+", " ", m.group(1)).strip(" .;:,«»\"'")
            if 3 <= len(t) < 90:
                return t
    return re.sub(r"\.pdf$", "", fname, flags=re.I).strip()


def safe_cache_name(fname: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", fname)


def collect(public_key: str, cache_dir: Path) -> list[dict]:
    os.makedirs(cache_dir, exist_ok=True)
    root = yandex_list(public_key)
    if not root:
        return []
    items = root.get("_embedded", {}).get("items", [])
    folder_path = None
    for it in items:
        if it.get("type") == "dir" and "ннотац" in it.get("name", ""):
            folder_path = it.get("path")
            break
    if not folder_path:
        return []
    listing = yandex_list(public_key, folder_path)
    if not listing:
        return []
    files = [
        it.get("name")
        for it in listing.get("_embedded", {}).get("items", [])
        if it.get("type") == "file" and it.get("name", "").lower().endswith(".pdf")
    ]
    anns: list[dict] = []
    for fname in files:
        pdf_path = cache_dir / safe_cache_name(fname)
        if pdf_path.exists():
            data = pdf_path.read_bytes()
        else:
            data = yandex_download(public_key, folder_path + "/" + fname)
            if not data:
                anns.append({"file": fname, "ok": False})
                continue
            pdf_path.write_bytes(data)
        text = extract_pdf_text(data)
        anns.append(
            {
                "file": fname,
                "ok": len(text) > 100,
                "text_len": len(text),
                "name": title_from_text(text, fname),
                "competencies": parse_competencies(text),
                "text": text,
            }
        )
        time.sleep(0.2)
    return anns


def cmd_collect(args: argparse.Namespace) -> None:
    targets = {k: v for k, v in TARGETS.items() if args.dir_code is None or k == args.dir_code}
    cache_base = Path(os.environ.get("TMPDIR", Path.home() / "AppData/Local/Temp/opencode")) / "sfu_pdfs"
    for dir_code, pk in targets.items():
        out_json = ANN_DIR / f"{dir_code}.json"
        if out_json.exists():
            print(f"{dir_code}: already done, skip")
            continue
        anns = collect(pk, cache_base / dir_code)
        ok = [a for a in anns if a.get("ok")]
        withcomp = [a for a in ok if a.get("competencies")]
        print(f"{dir_code}: total={len(anns)} ok={len(ok)} with_comp={len(withcomp)}")
        ANN_DIR.mkdir(exist_ok=True)
        out_json.write_text(json.dumps(anns, ensure_ascii=False, indent=1), encoding="utf-8")

    combined: dict[str, list[dict]] = {}
    for path in sorted(ANN_DIR.glob("*.json")):
        combined[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    ANN_ALL_PATH.write_text(json.dumps(combined, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {ANN_ALL_PATH}")


def normalize_comp(code: str) -> str | None:
    m = re.match(r"\b(УК|ОПК|ПК|ВПК|UK|OPK|PK|VPK)[\s-]*(\d+)", code, flags=re.I)
    if not m:
        return None
    ru = {"UK": "УК", "OPK": "ОПК", "PK": "ПК", "VPK": "ВПК"}
    base = m.group(1).upper()
    return f"{ru.get(base, base)}-{m.group(2)}"


def cmd_compare(args: argparse.Namespace) -> None:
    anns = json.loads(ANN_ALL_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for dir_code in TARGETS:
        krm_path = REFERENCE_DIR / f"krm_disciplines_{dir_code}_clean.json"
        if not krm_path.exists():
            print(f"{dir_code}: KRM-файл не найден, пропуск")
            continue
        krm = json.loads(krm_path.read_text(encoding="utf-8"))
        sub = list(krm.values())[0]
        krm_c = set()
        for info in sub.get("disciplines", {}).values():
            for c in info.get("competencies", []):
                n = normalize_comp(c)
                if n:
                    krm_c.add(n)
        sfu_c = set()
        for a in anns.get(dir_code, []):
            if not a.get("ok"):
                continue
            for c in a.get("competencies", []):
                n = normalize_comp(c)
                if n:
                    sfu_c.add(n)
        out[dir_code] = {
            "direction_name": sub.get("direction_name"),
            "profile": sub.get("profile"),
            "krm_competencies": sorted(krm_c),
            "sfu_annotations_competencies": sorted(sfu_c),
            "common": sorted(krm_c & sfu_c),
            "only_krm": sorted(krm_c - sfu_c),
            "only_sfu": sorted(sfu_c - krm_c),
        }
        print(f"{dir_code}: common={len(krm_c & sfu_c)} only_krm={len(krm_c - sfu_c)} only_sfu={len(sfu_c - krm_c)}")
    ANN_VS_KRM_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {ANN_VS_KRM_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect", help="скачать аннотации с Yandex Disk")
    p_collect.add_argument(
        "dir_code",
        nargs="?",
        default=None,
        help="код направления (например 09.03.04); все если не указан",
    )
    p_collect.set_defaults(func=cmd_collect)
    p_compare = sub.add_parser("compare", help="сравнить компетенции аннотаций с KRM")
    p_compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
