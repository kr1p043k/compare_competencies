from pypdf import PdfReader
import sys, os, json, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

f = "temp/РПД_Базы данных и СУБД.pdf"
reader = PdfReader(f)

for i in range(5, 12):
    text = reader.pages[i].extract_text()
    print(f"\n=== PAGE {i+1} ({len(text)} chars) ===")
    print(text[:2000])
