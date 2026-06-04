from pypdf import PdfReader
import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

reader = PdfReader("temp/rpd_pdfs/РПД_Объектно-ориентированное программирование.pdf")
full_text = ""
for p in reader.pages:
    t = p.extract_text()
    if t:
        full_text += t + "\n"

sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
matches = list(sec_pattern.finditer(full_text))

best_start = 0
best_end = 0
for i, m in enumerate(matches):
    if m.group().strip().startswith("III"):
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        if end_pos - m.end() > best_end - best_start:
            best_start = m.end()
            best_end = end_pos

chunk = full_text[best_start:best_end]
lines = chunk.split("\n")

# Show context around competency codes and KSA headers
for i, line in enumerate(lines):
    s = line.strip()
    if not s:
        continue
    # Show competency code lines
    if re.search(r"(ОПК|ПК|УК|ППК|ИП)", s):
        ctx = lines[max(0,i-2):i+5]
        print(f"[Line {i}] COMP: {s[:100]}")
        for j, cl in enumerate(ctx):
            print(f"  ctx[{j}]: {cl.strip()[:100]}")
        print()
