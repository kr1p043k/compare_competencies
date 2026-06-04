import json, os, re, sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

json_path = "data/reference/krm_disciplines_09.03.02.json"
pdf_dir = "temp/rpd_pdfs"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

disciplines = data["09.03.02"]["disciplines"]

# Find disciplines with 0 skills
zero_skill = []
for dname, info in disciplines.items():
    has_skills = any(len(v) > 0 for v in info["skills"].values())
    if not has_skills:
        zero_skill.append(dname)

print(f"Retrying {len(zero_skill)} disciplines with 0 skills")

comp_pattern = re.compile(
    r"(ОПК[-\s]?\d+(?:[.]\d+)?|"
    r"ПК[-\s]?\d+(?:[.]\d+)?|"
    r"УК[-\s]?\d+(?:[.]\d+)?|"
    r"ППК[-\s]?[РС]\d+(?:[.]\d+)?|"
    r"ИП[-\s]?\d+(?:[.]\d+)?)"
)

for dname in zero_skill:
    # Find matching PDF
    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        clean = fname.replace(".pdf", "").replace("РПД_", "").strip()
        if clean != dname:
            continue

        fpath = os.path.join(pdf_dir, fname)
        try:
            reader = PdfReader(fpath)
            full_text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"

            # Find the competency section
            sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
            sec_matches = list(sec_pattern.finditer(full_text))

            comp_start = None
            comp_end = None
            best_len = 0

            for i, m in enumerate(sec_matches):
                header = m.group().strip()
                if header.startswith("III"):
                    end_pos = sec_matches[i+1].start() if i+1 < len(sec_matches) else len(full_text)
                    clen = end_pos - m.end()
                    if clen > best_len:
                        best_len = clen
                        comp_start = m.end()
                        comp_end = end_pos

            if comp_start is None:
                # Look for any section with competency text
                for pat in ["ТРЕБОВАНИЯ К РЕЗУЛЬТАТАМ", "КОМПЕТЕНЦИ", "ИНДИКАТОРЫ"]:
                    idx = full_text.find(pat)
                    if idx >= 0:
                        for m in sec_matches:
                            if m.start() < idx:
                                continue
                            comp_start = m.end() if m.start() < idx + 200 else idx
                            comp_end = sec_matches[sec_matches.index(m)+1].start() if sec_matches.index(m)+1 < len(sec_matches) else idx + 10000
                            break
                        break

            if comp_start is None:
                print(f"  {dname}: no section found")
                continue

            comp_section = full_text[comp_start:comp_end]

            # Extract: look for any text near competency codes
            lines = comp_section.split("\n")
            current_comp = None
            comp_skills = {}

            for line in lines:
                s = line.strip()
                if not s:
                    continue

                cm = comp_pattern.search(s)
                if cm:
                    current_comp = cm.group(1).upper().replace(" ", "-")
                    if current_comp not in comp_skills:
                        comp_skills[current_comp] = []
                    # Extract text after the code
                    after = s[cm.end():].strip().lstrip(".:- ")
                    if after and len(after) > 5 and len(comp_skills[current_comp]) < 3:
                        pass  # Don't add the long description
                    continue

                if current_comp and len(s) > 5:
                    pass  # Don't add non-bullet lines

            info = disciplines[dname]
            sk_count = 0
            for comp, skills_list in comp_skills.items():
                if comp in info["skills"]:
                    info["skills"][comp] = skills_list
                    sk_count += len(skills_list)

            if sk_count > 0:
                print(f"  {dname}: recovered {sk_count} skills")
            else:
                # Try extracting ANY line between Знания/Умения/Навыки and next header
                in_ksa = False
                for line in lines:
                    s = line.strip()
                    if not s:
                        continue
                    if re.search(r"(Знания|Умения|Навыки)\s*:", s, re.IGNORECASE):
                        in_ksa = True
                        continue
                    if in_ksa and s.startswith("\u2012"):
                        skill = s[1:].strip()
                        skill = re.sub(r"\s+", " ", skill).strip(" ,;")
                        if len(skill) > 5 and current_comp:
                            if current_comp not in info["skills"]:
                                info["skills"][current_comp] = []
                            if skill not in info["skills"][current_comp]:
                                info["skills"][current_comp].append(skill)
                    if current_comp and not s.startswith("\u2012") and len(s) > 3:
                        in_ksa = False

                sk_count = sum(len(v) for v in info["skills"].values())
                if sk_count > 0:
                    print(f"  {dname}: recovered {sk_count} skills (v2)")

        except Exception as e:
            print(f"  {fname}: {e}")
        break

# Also handle missing English-named disciplines
print("\nTrying English-named PDFs...")
for fname in os.listdir(pdf_dir):
    if not fname.endswith(".pdf"):
        continue
    clean = fname.replace(".pdf", "").replace("РПД_", "").strip()
    if clean in disciplines:
        continue  # already parsed
    
    fpath = os.path.join(pdf_dir, fname)
    try:
        reader = PdfReader(fpath)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
        
        # Directly search for competency codes throughout document
        comp_skills = {}
        lines = full_text.split("\n")
        current_comp = None
        
        for line in lines:
            s = line.strip()
            if not s:
                continue
            cm = comp_pattern.search(s)
            if cm:
                current_comp = cm.group(1).upper().replace(" ", "-")
                if current_comp not in comp_skills:
                    comp_skills[current_comp] = []
                continue
            if current_comp and (s.startswith("\u2012") or s.startswith("- ") or s.startswith("\u2022")):
                skill = re.sub(r"^[\u2012\u2013\-\u2022]\s*", "", s).strip()
                skill = re.sub(r"\s+", " ", skill).strip(" ,;")
                if len(skill) > 5 and skill not in comp_skills[current_comp]:
                    comp_skills[current_comp].append(skill)
        
        if comp_skills:
            disciplines[clean] = {
                "competencies": sorted(comp_skills.keys()),
                "skills": comp_skills
            }
            skc = sum(len(v) for v in comp_skills.values())
            print(f"  {clean}: {len(comp_skills)} comps, {skc} skills (from English PDF)")
    except Exception as e:
        print(f"  {fname}: {e}")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

total_skills = sum(len(v) for d in disciplines.values() for v in d["skills"].values())
zeros = [(n, i) for n, i in disciplines.items() if not any(len(v) > 0 for v in i["skills"].values())]
print(f"\nFinal: {len(disciplines)} disciplines, {total_skills} skills")
print(f"Still 0 skills: {len(zeros)}")
for n, i in zeros:
    print(f"  {n}: {i['competencies']}")
