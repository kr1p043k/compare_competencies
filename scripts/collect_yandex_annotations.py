"""Скачать аннотации/РПД направления с публичной ссылки Yandex Disk в файл года.

`scripts/sfu_annotations.py` пишет результат в `sfu_annotations/<код>.json` и
затирает предыдущий год: у направления теперь несколько наборов, поэтому файлы
разделены суффиксом года (`<код>_2025.json`). Логика листинга и разбора PDF не
дублируется — она импортируется из существующего скрипта.

    python scripts/collect_yandex_annotations.py --dir-code 09.03.02 \
        --url https://disk.360.yandex.ru/d/-5D0p0XfTwL5Qg --year 2025
    python scripts/collect_yandex_annotations.py --dir-code 09.03.02 --year 2024  # ссылка из TARGETS
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "data" / "reference"


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "sfu_annotations", ROOT / "scripts" / "sfu_annotations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-code", required=True)
    parser.add_argument("--url", default=None,
                        help="публичная ссылка Yandex Disk; без неё берётся TARGETS из sfu_annotations.py")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sfu = _load_collector()
    public_key = args.url or sfu.TARGETS.get(args.dir_code)
    if not public_key:
        print(f"нет ссылки для {args.dir_code}: передайте --url")
        return 2

    import tempfile
    cache = Path(tempfile.gettempdir()) / "sfu_pdfs" / f"{args.dir_code}_{args.year}"
    print(f"{args.dir_code} ({args.year}): {public_key}")
    anns = sfu.collect(public_key, cache)
    if not anns:
        print("ничего не скачано")
        return 1

    ok = [a for a in anns if a.get("ok")]
    with_comp = [a for a in ok if a.get("competencies")]
    out = REFERENCE / "sfu_annotations" / f"{args.dir_code}_{args.year}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anns, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"всего={len(anns)} разобрано={len(ok)} с_компетенциями={len(with_comp)}")
    print(f"файл: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
