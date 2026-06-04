from pypdf import PdfReader
import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

reader = PdfReader("temp/rpd_pdfs/РПД_Объектно-ориентированное программирование.pdf")
full_text = ""
for p in reader.pages:
    t = p.extract_text()
    if t:
        full_text += t + "\n"

# Find the competency section
sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
matches = list(sec_pattern.finditer(full_text))

# Find last III before IV
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

# Show KSA headers and bullet-like lines
for line in lines:
    s = line.strip()
    if not s:
        continue
    if re.search(r"(Знания|Умения|Навыки)\s*:", s, re.IGNORECASE):
        print(f"KSA: {s[:120]}")
    elif s.startswith("\u2012") or s.startswith("- "):
        print(f"  BULLET: {s[:120]}")
