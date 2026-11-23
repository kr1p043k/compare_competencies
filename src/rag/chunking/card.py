"""Синтетические карточки для KRM/таксономии.

Компетенции и навыки в JSON размазаны по дисциплинам: чтобы ответить «что
входит в УК-3» или «где учат Python», нужен один документ на сущность.
Карточка — детерминированный текст, поэтому retrieval по ней работает и без
эмбеддингов (через entity/FTS-ветку).
"""

from __future__ import annotations

from typing import Any

from src.rag.types import Chunk, Document, normalize_entity_key

_CARD_MIN_CHARS = 60


def competency_card_text(
    code: str,
    *,
    description: str = "",
    skills: list[str] | None = None,
    disciplines: list[str] | None = None,
) -> str:
    lines = [f"Компетенция {code}."]
    if description:
        lines.append(description.strip())
    if skills:
        lines.append("Навыки: " + "; ".join(skills) + ".")
    if disciplines:
        lines.append("Формируется в дисциплинах: " + "; ".join(disciplines) + ".")
    return " ".join(lines)


def skill_card_text(name: str, *, category: str = "", aliases: list[str] | None = None) -> str:
    lines = [f"Навык: {name}."]
    if category:
        lines.append(f"Категория: {category}.")
    if aliases:
        lines.append("Синонимы: " + ", ".join(aliases) + ".")
    return " ".join(lines)


def discipline_card_text(name: str, *, competencies: list[str] | None = None, skills: list[str] | None = None) -> str:
    lines = [f"Дисциплина: {name}."]
    if competencies:
        lines.append("Компетенции: " + ", ".join(competencies) + ".")
    if skills:
        lines.append("Навыки: " + "; ".join(skills) + ".")
    return " ".join(lines)


def _document(source: str, external_id: str, title: str, text: str, entities: list[tuple[str, str]],
              *, direction: str | None = None, meta: dict | None = None) -> Document | None:
    if len(text) < _CARD_MIN_CHARS:
        return None
    chunk = Chunk(text=text, ord=0, section_path=title, entities=entities, meta=dict(meta or {}))
    return Document(
        source=source,
        external_id=external_id,
        title=title,
        chunks=[chunk],
        acl_scope="internal",
        direction=direction,
        meta=dict(meta or {}),
    )


def iter_krm_cards(payload: dict, *, direction: str | None = None) -> list[Document]:
    """Собрать карточки компетенций/дисциплин из krm_disciplines_*.json.

    Ожидаемая структура: {dir_code: {"disciplines": {name: {"competencies": [...],
    "skills": {code: [skills]}}}}}.
    """
    docs: list[Document] = []
    competencies: dict[str, dict[str, Any]] = {}
    disciplines: dict[str, dict[str, Any]] = {}

    for direction_payload in (payload or {}).values():
        if not isinstance(direction_payload, dict):
            continue
        disc_map = direction_payload.get("disciplines")
        if not isinstance(disc_map, dict):
            continue
        for disc_name, info in disc_map.items():
            if not isinstance(info, dict):
                continue
            skills_map = info.get("skills") if isinstance(info.get("skills"), dict) else {}
            codes = [str(code).strip() for code in (info.get("competencies") or []) if str(code).strip()]
            disc_skills: list[str] = []
            for code in codes:
                for skill in skills_map.get(code) or []:
                    text = str(skill).strip()
                    if text and text not in disc_skills:
                        disc_skills.append(text)
                bucket = competencies.setdefault(code, {"skills": [], "disciplines": []})
                for skill in skills_map.get(code) or []:
                    text = str(skill).strip()
                    if text and text not in bucket["skills"]:
                        bucket["skills"].append(text)
                if disc_name not in bucket["disciplines"]:
                    bucket["disciplines"].append(disc_name)
            entry = disciplines.setdefault(str(disc_name), {"competencies": [], "skills": []})
            for code in codes:
                if code not in entry["competencies"]:
                    entry["competencies"].append(code)
            for skill in disc_skills:
                if skill not in entry["skills"]:
                    entry["skills"].append(skill)

    for code, bucket in competencies.items():
        text = competency_card_text(
            code,
            skills=bucket["skills"][:40],
            disciplines=bucket["disciplines"][:20],
        )
        doc = _document(
            "krm",
            f"competency:{normalize_entity_key(code)}",
            f"Компетенция {code}",
            text,
            [("competency", normalize_entity_key(code))]
            + [("skill", normalize_entity_key(skill)) for skill in bucket["skills"][:40]]
            + [("discipline", normalize_entity_key(name)) for name in bucket["disciplines"][:20]],
            direction=direction,
            meta={"kind": "competency", "code": code},
        )
        if doc:
            docs.append(doc)

    for name, entry in disciplines.items():
        text = discipline_card_text(name, competencies=entry["competencies"], skills=entry["skills"][:40])
        doc = _document(
            "krm",
            f"discipline:{normalize_entity_key(name)}",
            f"Дисциплина {name}",
            text,
            [("discipline", normalize_entity_key(name))]
            + [("competency", normalize_entity_key(code)) for code in entry["competencies"]]
            + [("skill", normalize_entity_key(skill)) for skill in entry["skills"][:40]],
            direction=direction,
            meta={"kind": "discipline", "name": name},
        )
        if doc:
            docs.append(doc)
    return docs


def iter_skill_cards(taxonomy: dict) -> list[Document]:
    """Карточки навыков из skill_taxonomy.json.

    Структура файла: ``{"categories": {cat: {"label": ..., "skills": [...]}}}``.
    Один навык = одна карточка; варианты написания в той же категории идут
    в синонимы (Python / python — одна сущность).
    """
    docs: list[Document] = []
    if not isinstance(taxonomy, dict):
        return docs
    categories = taxonomy.get("categories")
    if not isinstance(categories, dict):
        return docs

    seen: dict[str, dict[str, Any]] = {}
    for cat_key, info in categories.items():
        if not isinstance(info, dict):
            continue
        label = str(info.get("label") or cat_key)
        names = [str(skill).strip() for skill in (info.get("skills") or []) if str(skill).strip()]
        for name in names:
            key = normalize_entity_key(name)
            if not key:
                continue
            entry = seen.setdefault(key, {"name": name, "category": label, "aliases": []})
            for variant in names:
                if (
                    variant != name
                    and normalize_entity_key(variant) == key
                    and variant not in entry["aliases"]
                ):
                    entry["aliases"].append(variant)

    for key, entry in seen.items():
        text = skill_card_text(entry["name"], category=entry["category"], aliases=entry["aliases"])
        doc = _document(
            "krm",
            f"skill:{key}",
            f"Навык {entry['name']}",
            text,
            [("skill", key)],
            meta={"kind": "skill", "category": entry["category"]},
        )
        if doc:
            docs.append(doc)
    return docs
