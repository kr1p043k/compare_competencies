"""Vacancy skill co-occurrence for polysemy disambiguation (v10)."""
from __future__ import annotations

from collections import Counter


class SkillCooccurrence:
    """P(ref | skill) lookup over vacancy skill sets."""

    def __init__(self) -> None:
        self.freq: dict[str, int] = {}
        self.top: dict[str, dict[str, int]] = {}
        self.vocab = None

    def build(self, vacancy_skill_sets, vocab=None, top_n: int = 40):
        freq: Counter = Counter()
        pair: Counter = Counter()
        for raw in vacancy_skill_sets or []:
            skills = {str(s).strip().lower() for s in (raw or []) if str(s).strip()
                      and (len(str(s).strip()) > 1 or str(s).strip().lower() in ("r", "c"))}
            if vocab is not None:
                skills &= set(vocab)
            for s in skills:
                freq[s] += 1
            if len(skills) < 2:
                continue
            ordered = sorted(skills)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pair[(ordered[i], ordered[j])] += 1
        grouped: dict[str, Counter] = {}
        for (a, b), c in pair.items():
            grouped.setdefault(a, Counter())[b] = c
            grouped.setdefault(b, Counter())[a] = c
        self.freq = dict(freq)
        self.top = {s: dict(c.most_common(top_n)) for s, c in grouped.items()}
        self.vocab = set(vocab) if vocab is not None else None
        return self

    def cond(self, ref: str, skill: str) -> float:
        """P(ref | skill): share of `skill` vacancies also mentioning `ref`."""
        s = (skill or "").strip().lower()
        r = (ref or "").strip().lower()
        f = self.freq.get(s, 0)
        if f <= 0 or r == s:
            return 0.0
        return self.top.get(s, {}).get(r, 0) / f

    def link(self, skill: str, ref_skills) -> float:
        """Max P(ref | skill) over reference skills (0.0 when unknown)."""
        best = 0.0
        for ref in ref_skills or []:
            p = self.cond(ref, skill)
            if p > best:
                best = p
        return best

    def top_partners(self, skill: str, n: int = 3) -> list:
        """Top-n co-occurring skills by pair count (for sense context, v13)."""
        s = (skill or "").strip().lower()
        return list(self.top.get(s, {}).keys())[:n]

    def to_cache(self) -> dict:
        payload = {"freq": self.freq, "top": self.top}
        if self.vocab is not None:
            payload["vocab"] = sorted(self.vocab)
        return payload

    @classmethod
    def from_cache(cls, payload: dict):
        obj = cls()
        obj.freq = dict(payload.get("freq", {}))
        obj.top = {s: {o: c for o, c in v.items() if (o or "").strip()}
                   for s, v in payload.get("top", {}).items() if (s or "").strip()}
        obj.vocab = {s for s in payload["vocab"] if (s or "").strip()} if payload.get("vocab") else None
        return obj
