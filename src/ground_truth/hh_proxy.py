"""Proxy ground truth: validates recommended skills against HH market demand."""

import json
from pathlib import Path
from typing import Optional

import structlog


logger = structlog.get_logger(__name__)


class HHGroundTruth:
    def __init__(self, history_dir: Optional[Path] = None, top_k: int = 100):
        from src.config import HISTORY_DIR
        self.history_dir = history_dir or HISTORY_DIR
        self.top_k = top_k
        self._market_skills: set[str] = set()

    def load(self) -> None:
        freq_path = self.history_dir / "freq_latest.json"
        if not freq_path.exists():
            logger.warning("hh_freq_not_found", path=str(freq_path))
            self._market_skills = set()
            return
        with open(freq_path, encoding="utf-8") as f:
            data = json.load(f)
        sorted_skills = sorted(data.items(), key=lambda x: x[1], reverse=True)
        self._market_skills = {skill for skill, _ in sorted_skills[:self.top_k]}
        logger.info("hh_ground_truth_loaded", skills=len(self._market_skills), top_k=self.top_k)

    def validate(self, recommended_skills: list[str]) -> dict[str, bool]:
        if not self._market_skills:
            self.load()
        market_lower = {ms.lower() for ms in self._market_skills}
        return {s: s.lower() in market_lower for s in recommended_skills}

    def precision_at_k(self, recommended_skills: list[str]) -> float:
        if not recommended_skills:
            return 0.0
        labels = self.validate(recommended_skills)
        hits = sum(1 for v in labels.values() if v)
        return hits / len(recommended_skills)
