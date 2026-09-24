"""Trend analyzer: skill demand trends over time."""
from __future__ import annotations


import structlog

from src.result import Ok, Err, Result
from src.errors import TrendError

logger = structlog.get_logger(__name__)


from src.utils import skill_words


class SnapshotTrendAnalyzer:
    def __init__(self, snapshot_records: list[dict] | None = None):
        self.snapshots: list[dict] = snapshot_records or []

    def set_snapshots(self, records: list[dict]) -> Result[None, TrendError]:
        if not records:
            logger.warning("snapshots_empty")
            return Err(TrendError(message="No snapshot records provided", reason="empty"))
        self.snapshots = records
        logger.info("snapshots_set", count=len(records))
        return Ok(None)

    def _normalize_freq(self, freq: dict[str, int]) -> dict[str, int]:
        """Normalize freq dict: merge aliases, apply user overrides."""
        freq = dict(freq)
        CUR_MERGE = {"linux": ["администрирование linux"]}
        for target, sources in CUR_MERGE.items():
            for src in sources:
                if src in freq:
                    freq[target] = freq.get(target, 0) + freq.pop(src)
        return freq

    def _pair_for_compare(self) -> tuple[dict, dict] | None:
        """Пара снимков для честного сравнения: два СВЕЖАЙШИХ одного source
        (обычно hh_vacancies). Смежность в общем списке не требуется: между
        рыночными лежат снимки профессий — иначе свежие данные игнорируются
        в пользу древней смежной пары, а сравнение идёт поперёк разрядов."""
        if len(self.snapshots) < 2:
            return None
        newest_source = (self.snapshots[-1].get("source") or "hh_vacancies")
        same = [s for s in self.snapshots
                if (s.get("source") or "hh_vacancies") == newest_source]
        if len(same) >= 2:
            return same[-2], same[-1]
        # Fallback: два последних любых (старое поведение).
        return self.snapshots[-2], self.snapshots[-1]

    @staticmethod
    def _shares(freq: dict[str, int]) -> tuple[dict[str, float], float]:
        total = sum(v for v in freq.values() if isinstance(v, (int, float))) or 1.0
        return ({k: v / total for k, v in freq.items() if isinstance(v, (int, float))}, total)

    def get_rising(self, top_n: int = 10) -> Result[list[dict], TrendError]:
        if len(self.snapshots) < 2:
            logger.warning("insufficient_snapshots_for_rising", count=len(self.snapshots))
            return Err(TrendError(
                message=f"Need ≥2 snapshots, got {len(self.snapshots)}",
                reason="insufficient_data",
            ))
        if top_n < 1:
            logger.warning("invalid_top_n", top_n=top_n)
            return Err(TrendError(message="top_n must be ≥1", reason="invalid_args"))

        latest = self._normalize_freq(self.snapshots[-1].get("skill_freq", {}))
        previous = self._normalize_freq(self.snapshots[-2].get("skill_freq", {}))

        pair = self._pair_for_compare()
        if pair is not None:
            prev_rec, cur_rec = pair
            previous = self._normalize_freq(prev_rec.get("skill_freq", {}))
            latest = self._normalize_freq(cur_rec.get("skill_freq", {}))

        # token-subset alias for renamed skills
        for ck in list(latest.keys()):
            if ck in previous or len(ck) < 3:
                continue
            ck_words = skill_words(ck)
            if not ck_words:
                continue
            for ok in list(previous.keys()):
                if ok != ck and ck_words <= skill_words(ok):
                    previous[ck] = previous[ok]
                    break

        prev_shares, _ = self._shares(previous)
        cur_shares, _ = self._shares(latest)

        changes = []
        for skill in cur_shares:
            prev_share = prev_shares.get(skill, 0)
            if prev_share <= 0:
                continue
            if previous.get(skill, 0) < 10:
                continue
            raw_change = (cur_shares[skill] - prev_share) / prev_share * 100
            capped = max(min(raw_change, 200), -200)
            changes.append({
                "skill": skill,
                "change_pct": round(capped, 1),
                "change_pct_raw": round(raw_change, 1),
                "capped": abs(raw_change) > 200,
                "frequency": latest.get(skill, 0),
            })
        # Только рост: без фильтра сюда попадали отрицательные («+-87%» в UI).
        rising = [c for c in changes if c["change_pct"] > 0]
        result = sorted(rising, key=lambda x: -x["change_pct"])[:top_n]
        logger.info("rising_skills_found", count=len(result))
        return Ok(result)

    def get_declining(self, top_n: int = 10) -> Result[list[dict], TrendError]:
        """Навыки с падающей частотой, включая исчезнувшие (-100%)."""
        if len(self.snapshots) < 2:
            logger.warning("insufficient_snapshots_for_declining", count=len(self.snapshots))
            return Err(TrendError(
                message=f"Need ≥2 snapshots, got {len(self.snapshots)}",
                reason="insufficient_data",
            ))
        if top_n < 1:
            logger.warning("invalid_top_n", top_n=top_n)
            return Err(TrendError(message="top_n must be ≥1", reason="invalid_args"))

        latest = self._normalize_freq(self.snapshots[-1].get("skill_freq", {}))
        previous = self._normalize_freq(self.snapshots[-2].get("skill_freq", {}))

        pair = self._pair_for_compare()
        if pair is not None:
            prev_rec, cur_rec = pair
            previous = self._normalize_freq(prev_rec.get("skill_freq", {}))
            latest = self._normalize_freq(cur_rec.get("skill_freq", {}))

        # token-subset alias for renamed skills
        for ck in list(latest.keys()):
            if ck in previous or len(ck) < 3:
                continue
            ck_words = skill_words(ck)
            if not ck_words:
                continue
            for ok in list(previous.keys()):
                if ok != ck and ck_words <= skill_words(ok):
                    previous[ck] = previous[ok]
                    break

        prev_shares, _ = self._shares(previous)
        cur_shares, _ = self._shares(latest)

        changes = []
        for skill in cur_shares:
            prev_share = prev_shares.get(skill, 0)
            if prev_share <= 0:
                continue
            if previous.get(skill, 0) < 10:
                continue
            raw_change = (cur_shares[skill] - prev_share) / prev_share * 100
            capped = max(min(raw_change, 200), -200)
            changes.append({
                "skill": skill,
                "change_pct": round(capped, 1),
                "change_pct_raw": round(raw_change, 1),
                "capped": abs(raw_change) > 200,
                "frequency": latest.get(skill, 0),
            })
        # Include skills that disappeared (in previous, not in latest)
        for skill, prev_freq in previous.items():
            if skill not in latest and prev_freq >= 10:
                changes.append({"skill": skill, "change_pct": -100.0, "frequency": 0})
        # Самые падающие — первые (по возрастанию change_pct), только отрицательные.
        declining = [c for c in changes if c["change_pct"] < 0]
        result = sorted(declining, key=lambda x: x["change_pct"])[:top_n]
        logger.info("declining_skills_found", count=len(result))
        return Ok(result)
