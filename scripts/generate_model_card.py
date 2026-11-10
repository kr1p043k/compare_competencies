"""Генерирует docs/MODEL_REGISTRY.md из реестра моделей.

Читает `data/models/registry/registry_index.json` (ведётся ``ModelRegistry``) и
пишет человекочитаемую карту: версии артефактов, размеры, метрики и теги.

    python -m scripts.generate_model_card [--out docs/MODEL_REGISTRY.md] [--registry-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _fmt_kv(data: dict[str, Any] | None) -> str:
    if not data:
        return "-"
    return ", ".join(f"{k}={v}" for k, v in data.items())


def build_model_card(index: dict[str, list[dict[str, Any]]], generated_at: str | None = None) -> str:
    """Собирает markdown-карту по индексу реестра. Чистая функция."""
    stamp = generated_at or datetime.now(UTC).isoformat()
    lines = [
        "# Model Registry",
        "",
        f"_Сгенерировано: {stamp}_",
        "",
        "Реестр артефактов моделей (LTR / embeddings / clusters).",
        "Источник: `data/models/registry/registry_index.json` (ведётся `ModelRegistry`).",
        "",
    ]
    if not index:
        lines += [
            "Модели ещё не зарегистрированы. Зарегистрируйте артефакт через",
            "`ModelRegistry().register*`, затем перегенерируйте карту.",
            "",
        ]
        return "\n".join(lines)

    for model_type in sorted(index):
        lines += [
            f"## {model_type}",
            "",
            "| Версия | Размер, МБ | Зарегистрировано | Метрики | Теги |",
            "|---|---|---|---|---|",
        ]
        for entry in index[model_type]:
            size_mb = round(int(entry.get("size_bytes") or 0) / 1024 / 1024, 2)
            lines.append(
                f"| {entry.get('version', '?')} "
                f"| {size_mb} "
                f"| {entry.get('registered_at', '?')} "
                f"| {_fmt_kv(entry.get('metrics'))} "
                f"| {_fmt_kv(entry.get('tags'))} |"
            )
        lines.append("")

    lines += [
        "## Ограничения",
        "",
        "- Метрики приведены как записаны при регистрации; сплит/источник не гарантированы.",
        "- Артефакты лежат вне git (`data/models/`, `data/cache/`); карта описывает реестр, а не бинари.",
        "",
    ]
    return "\n".join(lines)


def main(out: Path | None = None, registry_dir: Path | None = None) -> Path:
    from src import config

    reg_dir = Path(registry_dir) if registry_dir else (config.MODELS_DIR / "registry")
    index_path = reg_dir / "registry_index.json"
    index: dict[str, list[dict[str, Any]]] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            index = {}
    target = Path(out) if out else Path("docs/MODEL_REGISTRY.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_model_card(index), encoding="utf-8")
    print(f"Wrote {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сгенерировать docs/MODEL_REGISTRY.md из реестра моделей")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--registry-dir", type=Path, default=None)
    args = parser.parse_args()
    main(out=args.out, registry_dir=args.registry_dir)
