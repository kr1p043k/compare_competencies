# RPD Parsing — Result

## File
data/reference/krm_disciplines_09.03.02.json

## Stats
- PDFs: 49 total, 41 parsed
- Skills: 972 (after parent-child dedup)
- Zero-skill: 8 (foreign lang + PE - expected)
- KSA taxonomy: all 41 disciplines

## What was fixed
1. Multi-line bullet merging (continuation lines joined)
2. KSA structure (knowledge/abilities/skills per competency)
3. KSA prefix stripped ("Навыками работы" -> "работы")
4. Parent-child dedup (indicators removed if duplicate of parent)
5. Inline format capture (skills without bullet markers)

## Problems
- 8 PDFs with corrupt Cyrillic (pypdf), 2 unique lost
- Some over-merging (pypdf loses bullet markers between lines)
- Some truncation (PDF line-end cuts off skill text)
