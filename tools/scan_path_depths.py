"""Проверка путей вида Path(__file__).parent... / "data" / ... .

Ищем офф-бай-ван в глубине: путь должен существовать. Один такой уже был
(teacher.py: market-skills резолвился в src/data и всегда отдавал []).

Запуск: python tools/scan_path_depths.py  (из корня репозитория)
"""
from __future__ import annotations

import re
from pathlib import Path

PAT = re.compile(r'Path\(__file__\)(\.resolve\(\))?((?:\.parent)+)\s*/\s*"([^"]+)"')

root = Path.cwd()
problems = []
checked = 0

for py in Path("src").rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        for m in PAT.finditer(line):
            parents = m.group(2).count(".parent")
            first_seg = m.group(3)
            base = py
            for _ in range(parents):
                base = base.parent
            target = base / first_seg
            checked += 1
            if not target.exists():
                problems.append((str(py), i, parents, str(target)))

print(f"проверено выражений: {checked}")
print(f"несуществующих целей: {len(problems)}")
for p, i, parents, t in problems:
    print(f"  {p}:{i}  parents={parents}  -> {t}")
