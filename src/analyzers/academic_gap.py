"""Собственный (локальный) анализ разрыва компетенций по научной теме.

Не зависит от сервиса ЮФУ: тема → навыки → сравнение с рынком (it_skills)
и с компетенциями КРМ (эмбеддинги). Формат ответа совместим с
/academic/analyze-gap (GapResponse) — фронт не меняется.

Только чтение: использует диск-кэш эмбеддингов, не пишет в БД.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.parsing.skills.skill_parser import SkillParser, SkillSource

logger = structlog.get_logger(__name__)

# Порог cosine-близости «навык компетенции близок теме»
SIM_THRESHOLD = 0.55
# Порог близости рыночного навыка к навыкам компетенции (для suggested_skills)
COMP_MARKET_THRESHOLD = 0.45
# Пороги покрытия компетенции → статус
STATUS_THRESHOLDS = {"covered": 80, "partial": 40}
# Сколько уникальных навыков компетенций держим в памяти за раз
_TOP_REASON = 5
_TOP_NEAR = 5
_TOP_SUGGEST = 4


def _load_it_skills() -> list[str]:
    """Список рыночных навыков (it_skills.json)."""
    path = config.IT_SKILLS_PATH
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [s.strip() for s in data if isinstance(s, str) and s.strip()]
    except Exception as exc:
        logger.warning("it_skills_load_failed", error=str(exc))
        return []


def _load_krm(dir_code: str = "09.03.02") -> dict[str, dict[str, Any]]:
    """Возвращает {comp_code: {"skills": [...], "discipline": name}} из KRM-файла."""
    path = config.REFERENCE_DIR / f"krm_disciplines_{dir_code}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("krm_load_failed", error=str(exc))
        return {}
    sub = next(iter(data.values()), {}) if isinstance(data, dict) else {}
    discs = sub.get("disciplines", {}) if isinstance(sub, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for dname, disc in discs.items():
        if not isinstance(disc, dict):
            continue
        skills_map = disc.get("skills", {}) or {}
        for code, skills in skills_map.items():
            if not isinstance(skills, list):
                continue
            entry = out.setdefault(code, {"skills": [], "disciplines": set()})
            entry["skills"].extend(str(s) for s in skills if s)
            entry["disciplines"].add(dname)
    # дедупликация и сортировка по длине (короткие ключевые фразы информативнее)
    for entry in out.values():
        seen: set[str] = set()
        uniq: list[str] = []
        for s in entry["skills"]:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                uniq.append(s.strip())
        uniq.sort(key=len)
        entry["skills"] = uniq
        entry["disciplines"] = sorted(entry["disciplines"])
    return out


class AcademicGapAnalyzer:
    """Локальный анализ разрыва. Потокобезопасен на чтение (кэш общий)."""

    def __init__(self, dir_code: str = "09.03.02"):
        self.dir_code = dir_code
        self._comparator = None

    def _get_comparator(self):
        if self._comparator is None:
            from src.analyzers.comparison.embedding_comparator import EmbeddingComparator

            self._comparator = EmbeddingComparator(similarity_threshold=SIM_THRESHOLD)
        return self._comparator

    # ── шаг 1: тема → навыки ──────────────────────────────────────────────

    def topic_to_skills(self, topic: str) -> list[str]:
        import re

        parser = SkillParser()
        extracted = parser._extract_from_text(topic, SkillSource.DESCRIPTION)
        if extracted.is_err():
            extracted = []
        else:
            extracted = extracted.unwrap()
        skills = [s.text.strip() for s in extracted if s.text and s.text.strip()]
        # сама тема как единый фрагмент
        if topic and topic.strip():
            skills.append(topic.strip())
        # отдельные значимые слова/словосочетания темы (для семантики)
        words = re.findall(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z-]{2,}", topic)
        stop = {
            "методы", "метод", "анализ", "разработка", "системы", "система",
            "средства", "технологии", "технология", "и", "для", "в", "на", "по",
            "применение", "применения", "использование", "использования", "процессы",
            "процесс", "исследование", "исследования", "задачи", "задач", "основы",
        }
        for w in words:
            lw = w.lower()
            if lw not in stop and len(lw) >= 4 and lw not in {s.lower() for s in skills}:
                skills.append(w)
        seen: set[str] = set()
        uniq: list[str] = []
        for s in skills:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        logger.info("gap_topic_skills", topic=topic, count=len(uniq), skills=uniq[:20])
        return uniq

    # ── шаг 2: сравнение с рынком (it_skills) ─────────────────────────────

    def _market_stats(self, topic_skills: list[str]) -> dict[str, Any]:
        """Cosine-близость темы к рыночным навыкам (top-N). Возвращает также market_embs."""
        comp = self._get_comparator()
        market = _load_it_skills()
        if not market:
            return {"top_market": [], "market_avg_sim": 0.0, "market_embs": None}
        try:
            topic_embs = comp.embed_skills(topic_skills)
            if len(topic_embs) == 0:
                return {"top_market": [], "market_avg_sim": 0.0, "market_embs": None}
            topic_emb = np.mean(topic_embs, axis=0, keepdims=True)
            market_embs = comp.embed_skills(market)
            sims = cosine_similarity(topic_emb, market_embs)[0]
            order = np.argsort(sims)[::-1]
            top = [
                {"skill": market[i], "similarity": round(float(sims[i]), 4)}
                for i in order[:15]
                if float(sims[i]) >= 0.3
            ]
            return {
                "top_market": top,
                "market_avg_sim": round(float(sims.mean()), 4),
                "market_embs": market_embs,
            }
        except Exception as exc:
            logger.warning("gap_market_compare_failed", error=str(exc))
            return {"top_market": [], "market_avg_sim": 0.0, "market_embs": None}

    # ── шаг 3: сравнение с компетенциями КРМ ──────────────────────────────

    def _competency_analysis(self, topic_skills: list[str], market_top: list[dict], market_embs, topic_lower: str = "") -> list[dict]:
        comp = self._get_comparator()
        krm = _load_krm(self.dir_code)
        if not krm:
            return []

        topic_embs = comp.embed_skills(topic_skills)
        if len(topic_embs) == 0:
            return []

        market = _load_it_skills()

        results: list[dict] = []
        for code, entry in krm.items():
            skills = entry["skills"]
            if not skills:
                results.append({
                    "code": code,
                    "status": "gap",
                    "coverage_percent": 0,
                    "disciplines": entry["disciplines"],
                    "skills_count": 0,
                    "near_skills": [],
                    "missing_topic_skills": [],
                    "suggested_skills": [],
                    "reason": "У компетенции нет навыков в KRM.",
                    "recommendation": "Добавьте ЗУН для компетенции в РПД.",
                })
                continue
            try:
                skill_embs = comp.embed_skills(skills)
            except Exception as exc:
                logger.warning("gap_comp_embed_failed", code=code, error=str(exc))
                results.append({
                    "code": code,
                    "status": "gap",
                    "coverage_percent": 0,
                    "disciplines": entry["disciplines"],
                    "skills_count": len(skills),
                    "near_skills": [],
                    "missing_topic_skills": [],
                    "suggested_skills": [],
                    "reason": "Не удалось вычислить эмбеддинги навыков.",
                    "recommendation": "",
                })
                continue

            skills_lower = {s.lower() for s in skills}

            # для каждого навыка компетенции — макс. близость к любому навыку темы
            sims = cosine_similarity(skill_embs, topic_embs)  # (n_skills, n_topic)
            per_skill = sims.max(axis=1)
            coverage = round(float(per_skill.mean()) * 100)

            # статус
            if coverage >= STATUS_THRESHOLDS["covered"]:
                status = "covered"
            elif coverage >= STATUS_THRESHOLDS["partial"]:
                status = "partial"
            else:
                status = "gap"

            # near_skills: топ навыков компетенции, близких к теме
            idx = np.argsort(per_skill)[::-1][:_TOP_NEAR]
            near_skills = [
                {"skill": skills[i], "similarity": round(float(per_skill[i]), 3)}
                for i in idx if float(per_skill[i]) >= 0.4
            ]
            reason = (
                "Близкие к теме навыки: " + "; ".join(n["skill"] for n in near_skills[:3])
                if near_skills
                else "Навыки компетенции слабо связаны с темой."
            )

            # missing_topic_skills: навыки темы, которых нет в компетенции
            topic_to_comp = cosine_similarity(topic_embs, skill_embs)  # (n_topic, n_skills)
            topic_best = topic_to_comp.max(axis=1)
            missing_topic_skills = [
                topic_skills[i] for i in range(len(topic_skills))
                if float(topic_best[i]) < SIM_THRESHOLD
                and topic_skills[i].lower() not in skills_lower
                and topic_skills[i].lower() != topic_lower
                and len(topic_skills[i].strip()) >= 5
            ][:_TOP_SUGGEST]

            # suggested_skills: уникальные для компетенции + по теме
            suggested: list[dict] = []
            if market_embs is not None and len(market_embs):
                # рыночные навыки, близкие к навыкам ЭТОЙ компетенции
                comp_market = cosine_similarity(skill_embs, market_embs)  # (n_skills, n_market)
                market_best = comp_market.max(axis=0)
                order = np.argsort(market_best)[::-1]
                for i in order:
                    if len(suggested) >= _TOP_SUGGEST:
                        break
                    if float(market_best[i]) < COMP_MARKET_THRESHOLD:
                        break
                    if market[i].lower() in skills_lower or market[i].lower() in {s["skill"].lower() for s in suggested}:
                        continue
                    suggested.append({
                        "skill": market[i],
                        "similarity": round(float(market_best[i]), 3),
                        "source": "competency",
                    })
                # рыночные навыки, близкие к теме, которых нет в компетенции
                for m in market_top:
                    if len(suggested) >= _TOP_SUGGEST + 2:
                        break
                    if m["skill"].lower() in skills_lower or m["skill"].lower() in {s["skill"].lower() for s in suggested}:
                        continue
                    suggested.append({
                        "skill": m["skill"],
                        "similarity": m["similarity"],
                        "source": "topic",
                    })

            # рекомендация: уникальный текст с похожестью
            if suggested:
                parts = []
                for s in suggested:
                    src = "близок к вашим навыкам" if s["source"] == "competency" else "по теме"
                    parts.append(f"{s['skill']} ({s['similarity']:.2f}, {src})")
                recommendation = "Рекомендуется дополнить: " + ", ".join(parts) + "."
            elif coverage >= STATUS_THRESHOLDS["covered"]:
                recommendation = "Компетенция достаточно покрывает тему."
            else:
                recommendation = "Рекомендуется усилить навыки, связанные с темой."

            results.append({
                "code": code,
                "status": status,
                "coverage_percent": coverage,
                "disciplines": entry["disciplines"],
                "skills_count": len(skills),
                "near_skills": near_skills,
                "missing_topic_skills": missing_topic_skills,
                "suggested_skills": suggested,
                "reason": reason,
                "recommendation": recommendation,
            })

        results.sort(key=lambda r: r["coverage_percent"])
        return results

    # ── шаг 4: сводка ─────────────────────────────────────────────────────

    def _build_summary(self, results: list[dict], market_stats: dict[str, Any]) -> str:
        if not results:
            return "По направлению КРМ не найдено компетенций для анализа."
        total = len(results)
        covered = sum(1 for r in results if r["status"] == "covered")
        partial = sum(1 for r in results if r["status"] == "partial")
        gap = total - covered - partial
        top_market = market_stats.get("top_market", [])
        parts = [
            f"Проанализировано {total} компетенций КРМ: "
            f"покрыто {covered}, частично {partial}, не покрыто {gap}.",
        ]
        if top_market:
            names = ", ".join(m["skill"] for m in top_market[:5])
            parts.append(f"Наиболее близкие к теме рыночные навыки: {names}.")
        return " ".join(parts)

    # ── публичный метод ───────────────────────────────────────────────────

    def analyze(self, topic: str) -> dict[str, Any]:
        if not topic or not topic.strip():
            return {
                "overall_score": 0,
                "detailed_analysis": [],
                "summary": "Тема не задана.",
            }
        topic_skills = self.topic_to_skills(topic)
        market_stats = self._market_stats(topic_skills)
        results = self._competency_analysis(
            topic_skills,
            market_stats.get("top_market", []),
            market_stats.get("market_embs"),
            topic_lower=topic.strip().lower(),
        )
        overall = round(
            sum(r["coverage_percent"] for r in results) / len(results) / 100, 4
        ) if results else 0.0
        summary = self._build_summary(results, market_stats)
        return {
            "overall_score": overall,
            "detailed_analysis": results,
            "summary": summary,
        }
