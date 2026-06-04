import json, os, re, sys, urllib.request, urllib.parse
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dst_dir = "temp/rpd_pdfs"
os.makedirs(dst_dir, exist_ok=True)
output_file = "data/reference/krm_disciplines_09.03.02.json"

existing = [f for f in os.listdir(dst_dir) if f.endswith(".pdf")]
if len(existing) < 40:
    print("Downloading PDFs...")
    src_dir = r"C:\forCode\temp\parsed_rpd\pdfs"
    if os.path.exists(src_dir):
        import shutil
        for f in os.listdir(src_dir):
            if f.endswith(".pdf"):
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    else:
        pub_key = "https://disk.360.yandex.ru/d/-5D0p0XfTwL5Qg"
        enc_key = urllib.parse.quote(pub_key, safe="")
        folder = "/РПД, РПП, ГИА/"
        enc_folder = urllib.parse.quote(folder, safe="")
        list_url = "https://cloud-api.yandex.net/v1/disk/public/resources?public_key=" + enc_key + "&path=" + enc_folder + "&limit=100"
        data = json.loads(urllib.request.urlopen(list_url).read())
        items = [it for it in data["_embedded"]["items"] if it["name"].endswith(".pdf") and not it["name"].endswith(".sig")]
        exclude = ["РПП", "ГИА", "Rating", "ECTS", "BPM", "Аннотация"]
        items = [it for it in items if not any(kw in it["name"] for kw in exclude)]
        for item in items:
            name = item["name"]
            out = os.path.join(dst_dir, name)
            if os.path.exists(out): continue
            dl_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + enc_key + "&path=" + enc_folder + urllib.parse.quote(name, safe="")
            dl_data = json.loads(urllib.request.urlopen(dl_url).read())
            urllib.request.urlretrieve(dl_data["href"], out)

# --- Patterns ---
sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
comp_pattern = re.compile(r"(ОПК[-\s]?\d+(?:[.]\d+)?|ПК[-\s]?\d+(?:[.]\d+)?|УК[-\s]?\d+(?:[.]\d+)?|ППК[-\s]?[РС]\d+(?:[.]\d+)?|ИП[-\s]?\d+(?:[.]\d+)?)")

KSA_MAP = {
    "Знания": "knowledge", "Знать": "knowledge",
    "Умения": "abilities", "Уметь": "abilities",
    "Навыки": "skills", "Владеть": "skills",
}
ksa_pattern = re.compile(r"(Знания|Умения|Навыки|Знать|Уметь|Владеть)\s*:", re.IGNORECASE)

result = {
    "09.03.02": {
        "direction_name": "09.03.02 Информационные системы и технологии",
        "profile": "Перспективные информационные технологии",
        "disciplines": {}
    }
}

BULLET_RE = re.compile(r"[\u2012\u2013\-\u2022•‣⁃]\s")

KSA_PREFIXES = {
    "Навыками ", "Владеет ", "Владеет навыком ", "Владеет навыками ",
    "Знает ", "Умеет ", "Знания: ", "Умения: ", "Навыки: ",
    "Знать: ", "Уметь: ", "Владеть: ",
    "Практическими приемами ", "Практическим опытом ",
    "Техниками ", "Методами ", "Инструментами ", "Опытом ",
}

def is_continuation(line: str) -> bool:
    if not line:
        return False
    if line.startswith("(") or line.startswith(")"):
        return True
    if line.startswith(",") or line.startswith(";"):
        return True
    if not line[0].isupper():
        return True
    return False

def clean_skill(text: str) -> str:
    t = text
    # Remove bullet markers
    t = re.sub(r"^[\u2012\u2013\-\u2022•‣⁃]\s*", "", t)
    # Remove KSA prefix verbs (redundant with taxonomy)
    for prefix in KSA_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    # Also handle lowercase prefix forms
    for prefix in [p.lower() for p in ["Навыками ", "Владеет ", "Знает ", "Умеет ", "Знания: ", "Умения: ", "Навыки: "]]:
        if t.lower().startswith(prefix):
            t = t[len(prefix):]
            break
    t = re.sub(r"\s+", " ", t).strip(" ,;.:-")
    return t

def dedup_children(comp_ksa: dict) -> dict:
    cleaned = {}
    for comp in comp_ksa:
        parent = comp.rsplit(".", 1)[0]
        if "." in comp and parent in comp_ksa:
            child_set = set(tuple(comp_ksa[comp].get(k, [])) for k in ["knowledge", "abilities", "skills", "flat"])
            parent_set = set(tuple(comp_ksa[parent].get(k, [])) for k in ["knowledge", "abilities", "skills", "flat"])
            child_flat = set(comp_ksa[comp].get("flat", []))
            parent_flat = set(comp_ksa[parent].get("flat", []))
            # If child is subset of parent (all child skills already in parent), skip
            if child_flat.issubset(parent_flat):
                cleaned[comp] = None  # mark for removal
                continue
        cleaned[comp] = comp_ksa[comp]
    # Remove marked children
    for comp in list(cleaned.keys()):
        if cleaned[comp] is None:
            del cleaned[comp]
    return cleaned

def add_skill(comp_ksa, comp, ksa, text):
    t = clean_skill(text)
    if len(t) > 5:
        comp_ksa[comp][ksa].append(t)
        comp_ksa[comp]["flat"].append(t)

def last_skill(comp_ksa, comp, ksa):
    lst = comp_ksa[comp][ksa]
    return lst[-1] if lst else ""

def set_last_skill(comp_ksa, comp, ksa, val):
    lst = comp_ksa[comp][ksa]
    if lst:
        lst[-1] = val
    flat_lst = comp_ksa[comp]["flat"]
    if flat_lst:
        flat_lst[-1] = val

def extract_ksa_skills(comp_section: str) -> dict:
    lines = comp_section.split("\n")
    current_comp = None
    current_ksa = None
    comp_ksa = {}
    inline_mode = False  # True = capture non-bullet lines as skills

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        cm = comp_pattern.search(line)
        ks = ksa_pattern.search(line)

        if cm:
            comp_code = cm.group(1).upper().replace(" ", "-")
            if comp_code not in comp_ksa:
                comp_ksa[comp_code] = {"knowledge": [], "abilities": [], "skills": [], "flat": []}
            current_comp = comp_code
            current_ksa = KSA_MAP.get(ks.group(1).lower().capitalize()) if ks else None
            inline_mode = False
            continue

        if ks:
            current_ksa = KSA_MAP.get(ks.group(1).lower().capitalize(), "flat")
            after = line[ks.end():].strip()
            if after and len(after) > 5 and current_comp:
                add_skill(comp_ksa, current_comp, current_ksa, after)
            inline_mode = True
            continue

        if not current_comp or not current_ksa:
            continue

        is_bullet = bool(BULLET_RE.match(line))

        if is_bullet:
            inline_mode = False
            add_skill(comp_ksa, current_comp, current_ksa, line)
        elif is_continuation(line):
            nxt = last_skill(comp_ksa, current_comp, current_ksa)
            if nxt:
                set_last_skill(comp_ksa, current_comp, current_ksa, nxt + " " + line)
        elif inline_mode:
            add_skill(comp_ksa, current_comp, current_ksa, line)

    return comp_ksa

files = sorted([f for f in os.listdir(dst_dir) if f.endswith(".pdf")])

for fname in files:
    fpath = os.path.join(dst_dir, fname)
    try:
        reader = PdfReader(fpath)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: full_text += t + "\n"

        sec_matches = list(sec_pattern.finditer(full_text))

        comp_start = None; comp_end = None
        for i, m in enumerate(sec_matches):
            header = m.group().strip()
            if header.startswith("III"):
                end_pos = sec_matches[i+1].start() if i+1 < len(sec_matches) else len(full_text)
                clen = end_pos - m.end()
                if comp_start is None or clen > (comp_end - comp_start if comp_end else 0):
                    comp_start = m.end(); comp_end = end_pos

        if comp_start is None:
            for pat in ["ТРЕБОВАНИЯ К РЕЗУЛЬТАТАМ", "ПЛАНИРУЕМЫЕ РЕЗУЛЬТАТЫ", "КОМПЕТЕНЦИ"]:
                idx = full_text.find(pat)
                if idx >= 0:
                    for m in sec_matches:
                        if m.start() > idx:
                            comp_start = m.end() if m.start() < idx + 300 else idx
                            comp_end = sec_matches[sec_matches.index(m)+1].start() if sec_matches.index(m)+1 < len(sec_matches) else idx + 10000
                            break
                    break

        disc_name = fname.replace(".pdf", "").replace("РПД_", "").strip()

        if comp_start is None:
            comp_ksa = extract_ksa_skills(full_text)
            if not comp_ksa:
                continue
        else:
            comp_section = full_text[comp_start:comp_end]
            comp_ksa = extract_ksa_skills(comp_section)
            if not comp_ksa:
                continue

        # Merge parent from indicators
        for comp in list(comp_ksa.keys()):
            if "." in comp:
                parent = comp.rsplit(".", 1)[0]
                if parent in comp_ksa and parent != comp:
                    for k in ["knowledge", "abilities", "skills", "flat"]:
                        for s in comp_ksa[comp].get(k, []):
                            if s not in comp_ksa[parent].setdefault(k, []):
                                comp_ksa[parent][k].append(s)

        # Remove child indicators whose skills are entirely in parent
        comp_ksa = dedup_children(comp_ksa)

        result["09.03.02"]["disciplines"][disc_name] = {
            "competencies": sorted(comp_ksa.keys()),
            "skills": {c: v["flat"] for c, v in comp_ksa.items()},
            "ksa": {c: {k: v for k, v in comp_ksa[c].items() if k != "flat"} for c in comp_ksa}
        }
    except Exception as e:
        print(f"  {fname}: {e}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

total = len(result["09.03.02"]["disciplines"])
total_skills = sum(len(sk) for d in result["09.03.02"]["disciplines"].values() for sk in d["skills"].values())
zeros = [(n, i) for n, i in result["09.03.02"]["disciplines"].items() if not any(i["skills"].values())]
has_ksa = sum(1 for d in result["09.03.02"]["disciplines"].values() if d.get("ksa"))
print(f"\n{total} disciplines, {total_skills} skills, {has_ksa} with KSA")
print(f"Zero-skill: {len(zeros)}")
if zeros:
    for n, i in zeros:
        print(f"  {n}: {i['competencies']}")

# Show sample with structure
for n, d in result["09.03.02"]["disciplines"].items():
    if "Python" in n:
        print(f"\n=== {n} ===")
        for comp in sorted(d['skills'].keys()):
            flat = d['skills'].get(comp, [])
            if not flat: continue
            ksa_data = d.get("ksa", {}).get(comp, {})
            ksa_str = f"  KSA: k={len(ksa_data.get('knowledge',[]))}, a={len(ksa_data.get('abilities',[]))}, s={len(ksa_data.get('skills',[]))}"
            print(f"  {comp} ({len(flat)} skills) {ksa_str}")
            for s in flat[:4]:
                print(f"    {s[:100]}")
            if len(flat) > 4:
                print(f"    ... +{len(flat)-4} more")
        break
