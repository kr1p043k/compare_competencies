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


def _http_get(url: str, params: dict | None = None, timeout: int = 60, tries: int = 3,
              stream_to: Path | None = None):
    """GET с ретраями: медленный Диск не должен ронять весь сбор.

    timeout=(connect, read): короткий connect, ограниченный read. Файлы
    аннотаций — сотни КБ, поэтому read в 300с означал 15 минут на один
    зависший шард (2026-10-14: 39/51 PDF, каждый 220 КБ висел 300с на
    s*klg.storage.yandex.net). Теперь 45с + 4 попытки = 3 минуты worst case.
    """
    last_err: Exception | None = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=(15, timeout), stream=stream_to is not None)
            if r.status_code != 200:
                return None
            if stream_to is not None:
                try:
                    tmp = stream_to.with_suffix(stream_to.suffix + ".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=65536):
                            fh.write(chunk)
                    tmp.replace(stream_to)
                except OSError as e:
                    # Имя файла не нравится Windows (хвостовые точки/пробелы,
                    # суррогаты и т.п.) — не роняем весь сбор, файл пропускается.
                    print(f"STREAM-WRITE failed, skip file: {stream_to.name[:60]} ({e})")
                    return None
                return r
            return r
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (i + 1))
    print(f"GET failed after {tries} tries: {url[:120]} ({type(last_err).__name__}: {str(last_err)[:90]})")
    return None


def yandex_list(public_key: str, path: str = "") -> dict | None:
    r = _http_get(YAPI, params={"public_key": public_key, "path": path, "limit": 1000}, timeout=40)
    if r is None:
        return None
    return r.json()


def yandex_download(public_key: str, path: str, dest: Path | None = None) -> bytes | None:
    """Скачать файл. dest задан — стримим на диск (PDF не в памяти)."""
    r = _http_get(YAPI + "/download", params={"public_key": public_key, "path": path}, timeout=40)
    if r is None:
        return None
    href = r.json().get("href")
    if not href:
        return None
    if dest is not None:
        d = _http_get(href, timeout=45, tries=4, stream_to=dest)
        return dest.read_bytes() if d is not None else None
    d = _http_get(href, timeout=45, tries=4)
    return d.content if d is not None and d.status_code == 200 else None


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def parse_competencies(text: str, fname: str = "") -> list[str]:
    """Компетенции вида УК-1, ОПК-2.1, ВПК-3, а также англ. OPK-3 / UK-1.

    Реальные аннотации пишут коды по-разному (замер 2026-10-14 на 09.03.01_bim
    и 09.03.02): 'ОПК -1' (пробел перед дефисом), 'ОПК–2' (en-dash),
    'УК−8' (minus), и иногда код есть ТОЛЬКО в имени файла
    ('Аннотация ВПК-8 (бакалавриат).pdf'). Старый паттерн
    '[-\\s]?\\s*' резал всё это: −15 файлов и −24 уникальных кода.
    """
    # Разделитель: до 3 символов из пробелов/дефисов всех видов/минуса.
    pat = re.compile(r"\b(?:УК|ОПК|ПК|ВПК|UK|OPK|PK|VPK)[\s\-–—−]{0,3}\s*(\d+(?:\.\d+)*)")
    ru = {"UK": "УК", "OPK": "ОПК", "PK": "ПК", "VPK": "ВПК"}
    out: list[str] = []
    seen: set[str] = set()

    def _add(m: re.Match) -> None:
        prefix = re.sub(r"[^A-ZА-Я]", "", m.group(0).upper())
        prefix = ru.get(prefix, prefix)
        c = f"{prefix}-{m.group(1)}"
        if c not in seen:
            seen.add(c)
            out.append(c)

    for m in pat.finditer(text or ""):
        _add(m)
    # Фолбэк: код в имени файла, в тексте его нет.
    for m in pat.finditer(fname or ""):
        _add(m)
    return out


_TITLE_JUNK = re.compile(
    r"^(?:\d+[.)]?\s*)?(?:аннотация|annotation|of the course|name of the course|"
    r"название\s+(?:дисциплины|курса)|рабочей\s+программы|work\s+program|"
    r"course\s+description|учебной\s+дисциплины|бакалавриат)",
    re.I,
)

#: Хвост-слова, на которых предложение обрывается — признак, что строка
#: взята из середины фразы, а не из заголовка ('дисциплины',
#: 'вариативной профессиональной компетенции' — 12 файлов схлопывались
#: в один такой «заголовок», 2026-10-14).
_FRAGMENT_TAIL = re.compile(
    r"\b(?:в|на|и|или|с|к|для|от|по|что|как|при|относительно|в рамках)\s*$", re.I)


def _clean_title(t: str, *, require_capital: bool = False) -> str:
    t = re.sub(r"\s+", " ", t).strip(" .;:,«»\"'")
    if not (3 <= len(t) < 90):
        return ""
    if _TITLE_JUNK.match(t) or re.match(r"^\d", t):
        return ""
    if _FRAGMENT_TAIL.search(t):
        return ""
    if require_capital and t[0].islower():
        return ""
    return t


def title_from_text(text: str, fname: str) -> str:
    """Название дисциплины из шапки аннотации.

    Реальные форматы (замер 2026-10-14): RU «…рабочей программы учебной
    дисциплины «X»» и «…рабочей программы «X»» (ГИА, без 'учебной
    дисциплины'); EN 'Course description (annotation)\\nof the course\\nX'
    и '1. Name of the course\\nX'; EN 'work program of the academic
    discipline\\nX'; и файлы, где название — просто первая строка
    ('Operating systems'). Шаблонную строку '1. Name of the course'
    старый паттерн возвращал как название (мусор-дубликаты).
    """
    # 1-2) RU: рабочей программы (учебной дисциплины)? «X»
    for p in (
        r"рабочей\s+программы\s+учебной\s+дисциплины\s*[«\"']?\s*([^»\"'\n]{3,90})",
        r"рабочей\s+программы\s*[«\"']?\s*([^»\"'\n]{3,90})",
    ):
        m = re.search(p, text)
        if m:
            t = _clean_title(m.group(1))
            if t:
                return t
    # 3) EN-шаблон: '1. Name of the course\nX'
    m = re.search(r"Name of the course\s*:?\s*\n\s*([^\n]{3,90})", text, re.I)
    if m:
        t = _clean_title(m.group(1))
        if t:
            return t
    # 4) EN: Course description (annotation) → первая живая строка после
    m = re.search(r"Course\s+description\s*\(annotation\)", text, re.I)
    if m:
        for line in text[m.end():].splitlines()[:5]:
            t = _clean_title(line)
            if t:
                return t
    # 5) EN: work program of the academic discipline\nX
    m = re.search(
        r"work\s+program\s+of\s+the\s+academic\s+discipline\s*\n\s*([^\n]{3,90})", text, re.I)
    if m:
        t = _clean_title(m.group(1))
        if t:
            return t
    # 6) первая «X» в кавычках
    m = re.search(r"«([^»\n]{5,90})»", text)
    if m:
        t = _clean_title(m.group(1), require_capital=True)
        if t:
            return t
    # 7) первая осмысленная строка — только если это похоже на заголовок
    #    (с заглавной буквы: «Operating systems», «Математика»).
    for line in (text or "").splitlines()[:12]:
        t = _clean_title(line, require_capital=True)
        if t:
            return t
    return re.sub(r"\.pdf$", "", fname, flags=re.I).strip()


def safe_cache_name(fname: str) -> str:
    """Имя файла, безопасное для Windows: без хвостовых точек/пробелов
    (open() даёт FileNotFoundError), длина с запасом, суррогаты режем.
    Коллизии после урезания — суффикс sha1 (2026-10-14, падение 01.03.01)."""
    import hashlib

    name = re.sub(r"[^\w.\-]+", "_", fname or "file", flags=re.UNICODE)
    name = name.strip(" .")
    if len(name) > 120:
        stem, _, ext = name.rpartition(".")
        h = hashlib.sha1(fname.encode("utf-8", "replace")).hexdigest()[:10]
        name = f"{stem[:100]}_{h}.{ext}" if ext else f"{stem[:100]}_{h}"
    return name or "file.pdf"


def list_pdf_files(root: dict) -> list[str]:
    """Имена PDF-файлов в переданном ответе resources API."""
    return [
        it.get("name")
        for it in root.get("_embedded", {}).get("items", [])
        if it.get("type") == "file" and it.get("name", "").lower().endswith(".pdf")
    ]


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
    if folder_path:
        listing = yandex_list(public_key, folder_path)
        if not listing:
            return []
        files = list_pdf_files(listing)
    else:
        # Ссылка может вести прямо на папку с PDF — берём корень.
        files = list_pdf_files(root)
    anns: list[dict] = []
    for fname in files:
        pdf_path = cache_dir / safe_cache_name(fname)
        if pdf_path.exists():
            data = pdf_path.read_bytes()
        else:
            dl_path = (folder_path + "/" if folder_path else "") + fname
            data = yandex_download(public_key, dl_path, dest=pdf_path)
            if not data:
                anns.append({"file": fname, "ok": False})
                continue
            if not pdf_path.exists():
                pdf_path.write_bytes(data)
        text = extract_pdf_text(data)
        anns.append(
            {
                "file": fname,
                "ok": len(text) > 100,
                "text_len": len(text),
                "name": title_from_text(text, fname),
                "competencies": parse_competencies(text, fname),
                "text": text,
            }
        )
        time.sleep(0.2)
    return anns


def cmd_collect(args: argparse.Namespace) -> None:
    targets: dict[str, str] = {}
    if args.url:
        if not args.dir_code:
            print("--url требует указать dir_code")
            sys.exit(1)
        targets[args.dir_code] = args.url
    else:
        targets = {k: v for k, v in TARGETS.items() if args.dir_code is None or k == args.dir_code}
    import tempfile
    cache_base = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "sfu_pdfs"
    for dir_code, pk in targets.items():
        out_json = ANN_DIR / f"{dir_code}.json"
        if out_json.exists() and not args.url:
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
    p_collect.add_argument(
        "--url",
        default=None,
        help="публичная ссылка на папку Yandex Disk (например https://disk.360.yandex.ru/d/...); "
             "пересканирует/переписывает файл для dir_code",
    )
    p_collect.set_defaults(func=cmd_collect)
    p_compare = sub.add_parser("compare", help="сравнить компетенции аннотаций с KRM")
    p_compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
