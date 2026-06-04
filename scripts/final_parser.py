import json, os, re, sys, urllib.request, urllib.parse
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

dst_dir = "temp/rpd_pdfs"
os.makedirs(dst_dir, exist_ok=True)

# Download if needed
src_dir = r"C:\forCode\temp\parsed_rpd\pdfs"
need_download = True
if os.path.exists(src_dir):
    pdfs = [f for f in os.listdir(src_dir) if f.endswith(".pdf")]
    if len(pdfs) > 40:
        import shutil
        for f in pdfs:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        need_download = False

if need_download:
    existing = [f for f in os.listdir(dst_dir) if f.endswith(".pdf")]
    if len(existing) < 40:
        print("Downloading PDFs...")
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

sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
comp_pattern = re.compile(r"(ОПК[-\s]?\d+(?:[.]\d+)?|ПК[-\s]?\d+(?:[.]\d+)?|УК[-\s]?\d+(?:[.]\d+)?|ППК[-\s]?[РС]\d+(?:[.]\d+)?|ИП[-\s]?\d+(?:[.]\d+)?)")
ksa_header = re.compile(r"(Знания|Умения|Навыки|Знать|Уметь|Владеть)\s*:", re.IGNORECASE)

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

        sec_matches = list(sec_pattern.finditer(full_text))
        comp_start = None
        comp_end = None

        # Find real III section (skip TOC)
        for i, m in enumerate(sec_matches):
            header = m.group().strip()
            if header.startswith("III"):
                end_pos = sec_matches[i+1].start() if i+1 < len(sec_matches) else len(full_text)
                clen = end_pos - m.end()
                if comp_start is None or clen > (comp_end - comp_start if comp_end else 0):
                    comp_start = m.end()
                    comp_end = end_pos

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

        if comp_start is None:
            # Fallback: parse the entire document for competencies
            lines = full_text.split("\n")
            fallback_comp_skills = {}
            fc = None
            fi = 0

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                cm = comp_pattern.search(line)
                if cm:
                    fc = cm.group(1).upper().replace(" ", "-")
                    if fc not in fallback_comp_skills:
                        fallback_comp_skills[fc] = []
                    fi = 0
                    continue
                ks = ksa_header.search(line)
                if ks:
                    fi = 2
                    continue
                if fc:
                    is_bullet = line.startswith("\u2012") or line.startswith("\u2013") or line.startswith("- ") or line.startswith("\u2022")
                    if fi > 0:
                        fi -= 1
                        if is_bullet:
                            skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", line).strip()
                        else:
                            skill = line
                        skill = re.sub(r"\s+", " ", skill).strip(" ,;.:-")
                        if len(skill) > 5 and skill not in fallback_comp_skills[fc]:
                            fallback_comp_skills[fc].append(skill)
                    elif is_bullet:
                        skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", line).strip()
                        skill = re.sub(r"\s+", " ", skill).strip(" ,;.:-")
                        if len(skill) > 5 and skill not in fallback_comp_skills[fc]:
                            fallback_comp_skills[fc].append(skill)

            for comp in list(fallback_comp_skills.keys()):
                if "." in comp:
                    parent = comp.rsplit(".", 1)[0]
                    if parent in fallback_comp_skills and parent != comp:
                        for s in fallback_comp_skills[comp]:
                            if s not in fallback_comp_skills[parent]:
                                fallback_comp_skills[parent].append(s)

            if fallback_comp_skills:
                disc_name = fname.replace(".pdf", "").replace("РПД_", "").strip()
                result["09.03.02"]["disciplines"][disc_name] = {
                    "competencies": sorted(fallback_comp_skills.keys()),
                    "skills": fallback_comp_skills
                }
            continue

        comp_section = full_text[comp_start:comp_end]
        disc_name = fname.replace(".pdf", "").replace("РПД_", "").strip()

        lines = comp_section.split("\n")
        current_comp = None
        comp_skills = {}
        in_ksa = 0

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # Check for KRM competency code
            cm = comp_pattern.search(line)
            if cm:
                current_comp = cm.group(1).upper().replace(" ", "-")
                if current_comp not in comp_skills:
                    comp_skills[current_comp] = []
                in_ksa = 0
                continue

            # Check for Знания/Умения/Навыки header
            ks = ksa_header.search(line)
            if ks:
                in_ksa = 2  # capture next 2 lines after header
                continue

            # Extract skills
            if current_comp:
                is_bullet = line.startswith("\u2012") or line.startswith("\u2013") or line.startswith("- ") or line.startswith("\u2022")

                if in_ksa > 0:
                    in_ksa -= 1
                    if is_bullet:
                        skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", line).strip()
                    else:
                        skill = line
                    skill = re.sub(r"\s+", " ", skill).strip(" ,;.:-")
                    if len(skill) > 5 and skill not in comp_skills[current_comp]:
                        comp_skills[current_comp].append(skill)
                elif is_bullet:
                    # Bullet points even without KSA header
                    skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", line).strip()
                    skill = re.sub(r"\s+", " ", skill).strip(" ,;.:-")
                    if len(skill) > 5 and skill not in comp_skills[current_comp]:
                        comp_skills[current_comp].append(skill)

        # Merge parent skills from indicators
        for comp in list(comp_skills.keys()):
            if "." in comp:
                parent = comp.rsplit(".", 1)[0]
                if parent in comp_skills and parent != comp:
                    for s in comp_skills[comp]:
                        if s not in comp_skills[parent]:
                            comp_skills[parent].append(s)

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
zeros = [(n, i) for n, i in result["09.03.02"]["disciplines"].items() if not any(len(v) > 0 for v in i["skills"].values())]
print(f"\n{len(result['09.03.02']['disciplines'])} disciplines, {total_skills} skills")
print(f"Zero-skill disciplines: {len(zeros)}")
for n, i in zeros:
    print(f"  {n}: {i['competencies']}")
