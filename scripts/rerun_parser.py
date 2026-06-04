import json, os, re, sys, urllib.request, urllib.parse
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dst_dir = "temp/rpd_pdfs"
os.makedirs(dst_dir, exist_ok=True)

# Download PDFs
src_dir = r"C:\forCode\temp\parsed_rpd\pdfs"
need_download = True
if os.path.exists(src_dir):
    pdfs = [f for f in os.listdir(src_dir) if f.endswith(".pdf")]
    if len(pdfs) > 40:
        import shutil
        for f in pdfs:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print(f"Copied {len(pdfs)} PDFs from {src_dir}")
        need_download = False

if need_download:
    existing = [f for f in os.listdir(dst_dir) if f.endswith(".pdf")]
    if len(existing) < 40:
        print("Downloading PDFs from Yandex.Disk...")
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
            if os.path.exists(out):
                continue
            dl_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=" + enc_key + "&path=" + enc_folder + urllib.parse.quote(name, safe="")
            dl_data = json.loads(urllib.request.urlopen(dl_url).read())
            urllib.request.urlretrieve(dl_data["href"], out)

output_file = "data/reference/krm_disciplines_09.03.02.json"

result = {
    "09.03.02": {
        "direction_name": "09.03.02 Информационные системы и технологии",
        "profile": "Перспективные информационные технологии",
        "disciplines": {}
    }
}

# Section header pattern
sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
# Competency codes we care about (KRM codes ONLY, not PL/other standards)
comp_pattern = re.compile(r"(ОПК[-\s]?\d+(?:[.]\d+)?|ПК[-\s]?\d+(?:[.]\d+)?|УК[-\s]?\d+(?:[.]\d+)?|ППК[-\s]?[РС]\d+(?:[.]\d+)?|ИП[-\s]?\d+(?:[.]\d+)?)")
# Knowledge/Skills/Abilities headers
ksa_header = re.compile(r"(Знания|Умения|Навыки)\s*:", re.IGNORECASE)

files = sorted([f for f in os.listdir(dst_dir) if f.endswith(".pdf")])

for fname in files:
    fpath = os.path.join(dst_dir, fname)
    try:
        reader = PdfReader(fpath)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

        # Find all section positions
        sec_matches = list(sec_pattern.finditer(full_text))
        
        # Find the REAL competency section (skip TOC, use section with real content)
        comp_start = None
        comp_end = None
        best_len = 0
        
        for i, m in enumerate(sec_matches):
            header = m.group().strip()
            if header.startswith("III"):
                end_pos = sec_matches[i+1].start() if i+1 < len(sec_matches) else len(full_text)
                content_len = end_pos - m.end()
                if content_len > best_len:
                    best_len = content_len
                    comp_start = m.end()
                    comp_end = end_pos
        
        if comp_start is None or best_len < 500:
            # Fallback: find the longest section containing competency text
            for pat in ["ТРЕБОВАНИЯ К РЕЗУЛЬТАТАМ", "ПЛАНИРУЕМЫЕ РЕЗУЛЬТАТЫ", "КОМПЕТЕНЦИ"]:
                idx = full_text.find(pat)
                if idx >= 0:
                    # Find the section header that precedes this text
                    for j, m in enumerate(sec_matches):
                        if m.start() < idx:
                            continue
                        comp_start = m.end() if m.start() < idx + 200 else idx
                        comp_end = sec_matches[j+1].start() if j+1 < len(sec_matches) else idx + 10000
                        break
                    if comp_start is not None and comp_end is not None:
                        break

        if comp_start is None:
            continue

        comp_section = full_text[comp_start:comp_end]
        disc_name = fname.replace(".pdf", "").replace("РПД_", "").strip()

        # Extract bullet-point skills from the competency section
        lines = comp_section.split("\n")
        current_comp = None
        comp_skills = {}

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Check for KRM competency code
            cm = comp_pattern.search(line)
            if cm:
                code = cm.group(1).upper().replace(" ", "-")
                current_comp = code
                if code not in comp_skills:
                    comp_skills[code] = []
                continue

            # Check for Знания/Умения/Навыки header
            if ksa_header.search(line):
                continue

            # Extract bullet points (lines starting with bullet character)
            if current_comp and (line.startswith("\u2012") or line.startswith("\u2013") or line.startswith("- ") or line.startswith("\u2022")):
                skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", line).strip()
                skill = re.sub(r"\s+", " ", skill)
                skill = skill.strip(" ,;.:-")
                if len(skill) > 5:
                    if current_comp not in comp_skills:
                        comp_skills[current_comp] = []
                    if skill not in comp_skills[current_comp]:
                        comp_skills[current_comp].append(skill)

        if comp_skills:
            result["09.03.02"]["disciplines"][disc_name] = {
                "competencies": sorted(comp_skills.keys()),
                "skills": comp_skills
            }

    except Exception as e:
        print(f"  {fname}: {e}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

total_skills = sum(len(v) for d in result["09.03.02"]["disciplines"].values() for v in d["skills"].values())
print(f"\n{len(result['09.03.02']['disciplines'])} disciplines, {total_skills} skills")

for dname, info in sorted(result["09.03.02"]["disciplines"].items()):
    sc = sum(len(v) for v in info["skills"].values())
    print(f"  {dname}: {len(info['competencies'])} comps, {sc} skills")
