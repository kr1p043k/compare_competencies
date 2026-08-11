"""SFU (ЮФУ) data collection helpers.

Sources:
  - https://sfedu.ru/www/edu.show_docs_new?p_sel15_id=<id>  (program docs page, CP1251)
  - https://sfedu.ru/www/stat_pages22.show?p=EDU/N...        (program catalogs by year)
  - https://sfedu.sharepoint.com/sites/educational_programs  (РПД files, REST API)
"""
from __future__ import annotations

import json
import re
import time
import warnings
from typing import Any

import urllib3
import requests

# sfedu.ru использует сертификат российского УЦ (нет в системных bundle).
# Данные публичные, read-only — выключаем проверку TLS.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
}
SESSION = requests.Session()
SESSION.headers.update(UA)


def get_cp1251(url: str, timeout: int = 45, retries: int = 3) -> str | None:
    """GET and decode as windows-1251."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, verify=False)
            if r.status_code == 200 and len(r.content) > 500:
                return r.content.decode("windows-1251", errors="replace")
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def get_utf8(url: str, timeout: int = 45, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, verify=False)
            if r.status_code == 200 and len(r.content) > 500:
                return r.content.decode("utf-8", errors="replace")
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def get_bytes(url: str, timeout: int = 90, retries: int = 3) -> bytes | None:
    """GET raw bytes (for PDF download)."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout, verify=False)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def html_to_text(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def parse_program_info(html: str) -> dict[str, Any]:
    """Extract {code, name, profile, year, institute, leader} from docs page heading."""
    body = html_to_text(html)
    idx = body.find("Образовательная программа")
    if idx < 0:
        idx = body.find("образовательная программа")
    seg = body[idx: idx + 600] if idx >= 0 else body[:600]
    code = re.search(r"\b\d{2}\.\d{2}\.\d{2}\b", seg)
    return {
        "code": code.group(0) if code else None,
        "heading": seg[:300],
    }


def parse_discipline_list(html: str) -> list[str]:
    """Extract РПД discipline titles from the docs page.

    The РПД section: "Рабочая программа дисциплины <Name> Аннотация РПД УКД РПД
    <Name2> Аннотация РПД УКД РПД ..." and ends at "Практики:".
    """
    body = html_to_text(html)
    marker = "Рабочая программа дисциплины"
    i = body.find(marker)
    if i < 0:
        return []
    seg = body[i + len(marker):]
    cut = seg.find("Практики:")
    if cut >= 0:
        seg = seg[:cut]
    names: list[str] = []
    for part in re.split(r"Аннотация РПД", seg):
        part = part.strip()
        if part.startswith("УКД РПД"):
            part = part[len("УКД РПД"):]
        part = re.sub(r"\s+", " ", part).strip()
        if part and len(part) >= 3 and part != "РПД":
            names.append(part)
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def fetch_program_ajax(p_sel15_id: int) -> dict[str, Any]:
    """Fetch program data via the AJAX endpoint (works for all programs, JS content).

    Returns dict with:
      id, ok, title, code, name, profile, form, year, disciplines[{name, ze, hours, annot_num}]
    """
    url = f"https://sfedu.ru/www/edu.show_ajax_docs_new?p_sel15_id={p_sel15_id}"
    raw = get_cp1251(url)
    info: dict[str, Any] = {"id": p_sel15_id, "ok": False}
    if not raw:
        return info
    try:
        d = json.loads(raw)
    except Exception:
        return info
    info["ok"] = True
    title = html_to_text(d.get("title_html", ""))
    info["title"] = title[:300]
    m = re.search(
        r"Образовательная программа\s+(\d{2}\.\d{2}\.\d{2})\s+([^|]+?)\s*\|\s*НАПРАВЛЕННОСТЬ:\s*([^|]+?)\s*\|\s*Форма обучения:\s*([^|]+?)\s*\|\s*Год набора:\s*(\d{4})",
        title,
    )
    if m:
        info["code"] = m.group(1).strip()
        info["name"] = m.group(2).strip()
        info["profile"] = m.group(3).strip()
        info["form"] = m.group(4).strip()
        info["year"] = m.group(5).strip()
    dl = d.get("doc_list_html", "")
    info["disciplines"] = parse_doc_list(dl)
    return info


def parse_doc_list(doc_list_html: str) -> list[dict[str, Any]]:
    """Parse doc_list_html: rows = {name, ze, hours, annot_num}."""
    out: list[dict[str, Any]] = []
    for tr in re.findall(r"<tr>([\s\S]*?)</tr>", doc_list_html):
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr)
        if len(tds) < 4:
            continue
        name = re.sub(r"<[^>]+>", " ", tds[1])
        name = re.sub(r"\s+", " ", name).strip()
        ze = re.sub(r"<[^>]+>", " ", tds[2]).strip()
        hours = re.sub(r"<[^>]+>", " ", tds[3]).strip()
        m = re.search(r"doc=annot&num=(\d+)", tr)
        annot_num = int(m.group(1)) if m else None
        if name and name != "Дисциплина":
            out.append({"name": name, "ze": ze, "hours": hours, "annot_num": annot_num})
    return out


def fetch_annotation(p_sel15_id: int, num: int) -> dict[str, Any]:
    """Fetch an annotation (РПД short) for discipline num.

    Returns {ok, title, goals, position, competencies, content}.
    """
    url = f"https://sfedu.ru/www/edu.show_ajax_docs_new?p_sel15_id={p_sel15_id}&doc=annot&num={num}"
    raw = get_cp1251(url)
    info: dict[str, Any] = {"ok": False}
    if not raw:
        return info
    try:
        d = json.loads(raw)
    except Exception:
        return info
    info["ok"] = True
    txt = html_to_text(d.get("description_html", ""))
    info["title"] = html_to_text(d.get("title_html", ""))[:200]
    info["text"] = txt
    comp = re.search(r"Компетенции обучающегося[^\n]{0,40}:\s*(.+)", txt)
    if comp:
        # competencies may run to next numbered section
        seg = comp.group(1)
        seg = re.split(r"\d\.\s*[А-ЯA-Z]", seg)[0]
        info["competencies"] = re.findall(r"(?:УК|ОПК|ПК)-?\d+", seg)
    else:
        info["competencies"] = []
    return info


def fetch_program_docs(p_sel15_id: int) -> dict[str, Any]:
    """Full info for one program docs page."""
    url = f"https://sfedu.ru/www/edu.show_docs_new?p_sel15_id={p_sel15_id}"
    html = get_cp1251(url)
    if not html:
        return {"id": p_sel15_id, "ok": False}
    info = parse_program_info(html)
    disciplines = parse_discipline_list(html)
    return {
        "id": p_sel15_id,
        "ok": True,
        "code": info["code"],
        "heading": info["heading"],
        "disciplines_count": len(disciplines),
        "disciplines": disciplines,
    }


def catalog_programs(year_url: str) -> list[dict[str, Any]]:
    """Parse bachelor (tabs-2) rows of a catalog page."""
    html = get_cp1251(year_url)
    if not html:
        return []
    s = html.find('id="tabs-2"')
    if s < 0:
        return []
    e = html.find('id="tabs-3"', s)
    if e < 0:
        e = len(html)
    bak = html[s:e]
    rows: list[dict[str, Any]] = []
    for tr in re.findall(r"<tr>([\s\S]*?)</tr>", bak):
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr)
        if len(tds) < 2:
            continue
        prog = re.sub(r"<[^>]+>", " ", tds[0])
        prog = re.sub(r"\s+", " ", prog).strip()
        form = re.sub(r"<[^>]+>", " ", tds[1])
        form = re.sub(r"\s+", " ", form).strip()
        unit = re.sub(r"<[^>]+>", " ", tds[3] if len(tds) > 3 else "")
        unit = re.sub(r"\s+", " ", unit).strip()
        m = re.search(r"p_sel15_id=(\d+)", tr)
        if m:
            rows.append({"id": int(m.group(1)), "program": prog, "form": form, "unit": unit})
    return rows


CATALOGS = {
    2022: "https://sfedu.ru/www/stat_pages22.show?p=EDU/N14116/D",
    2023: "https://sfedu.ru/www/stat_pages22.show?p=EDU/N14548/D",
    2024: "https://sfedu.ru/www/stat_pages22.show?p=EDU/N14696/D",
    2025: "https://sfedu.ru/www/stat_pages22.show?p=EDU/N13182/D",
}
