import json, re, sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("scripts/rebuild_with_ksa.py", "r", encoding="utf-8") as f:
    src = f.read()
exec(src)

# Now check which disciplines that previously had skills now have zero
with open("data/reference/krm_disciplines_09.03.02.json", "r", encoding="utf-8") as f:
    data = json.load(f)

zeros = [(n, d) for n, d in data["09.03.02"]["disciplines"].items() if not any(d["skills"].values())]
print(f"Zero-skill disciplines: {len(zeros)}")
for n, d in zeros:
    print(f"\n  {n}")
    print(f"  Comps: {d['competencies']}")
    # Try to find why no skills - look at PDF text
    for fname in [f"temp/rpd_pdfs/РПД_{n}.pdf"]:
        try:
            reader = PdfReader(fname)
            text = ""
            for p in reader.pages:
                t = p.extract_text()
                if t: text += t + "\n"
            # find section III or KSA headers
            ksa = re.findall(r"(Знания|Умения|Навыки|Знать|Уметь|Владеть)\s*:", text, re.IGNORECASE)
            comp = re.findall(r"(ОПК[-\s]?\d+|ПК[-\s]?\d+|УК[-\s]?\d+)", text)
            print(f"  KSA headers found: {len(ksa)}")
            print(f"  Competencies found: {len(set(comp))}")
            # Show first 100 chars of section III
            sec_pat = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
            for m in list(sec_pat.finditer(text)):
                if m.group().strip().startswith("III"):
                    sect = text[m.end():m.end()+300]
                    print(f"  III section snippet: {repr(sect[:200])}")
                    break
            else:
                print(f"  No section III found. Text length: {len(text)}")
                print(f"  First 100: {repr(text[:100])}")
        except:
            print(f"  Could not read PDF")
