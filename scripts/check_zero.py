import sys, re
from pypdf import PdfReader
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Check one zero-skill PDF for structure
fname = "temp/rpd_pdfs/РПД_Байесовские и генеративные модели в искусственном интеллекте.pdf"
reader = PdfReader(fname)

full_text = ""
for page in reader.pages:
    t = page.extract_text()
    if t:
        full_text += t + "\n"

# Find the competency section
sec_pattern = re.compile(r"\n\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[А-ЯЁ][А-ЯЁ\s]+", re.IGNORECASE)
sec_matches = list(sec_pattern.finditer(full_text))

print("Section headers:")
for m in sec_matches:
    print(f"  [{m.start()}] {m.group().strip()[:80]}")

# Find the longest III section
best_len = 0
best_start = 0
best_end = 0
for i, m in enumerate(sec_matches):
    if m.group().strip().startswith("III"):
        end_pos = sec_matches[i+1].start() if i+1 < len(sec_matches) else len(full_text)
        clen = end_pos - m.end()
        if clen > best_len:
            best_len = clen
            best_start = m.end()
            best_end = end_pos

if best_len > 0:
    chunk = full_text[best_start:best_end]
    print(f"\nIII section content ({best_len} chars):")
    # Show lines with bullet-like starts
    for line in chunk.split("\n"):
        s = line.strip()
        if s:
            first = s[0]
            if first in "\u2012\u2013\u2022\u2023\u25B8\u25C6\u2043\u25E6\u2024\u203B" or first == "-" or ord(first) > 127:
                print(f"  [{hex(ord(first))}] {s[:100]}")
