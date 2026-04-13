"""
Анализ трендов: какие навыки растут/падают в спросе
Поддерживает сравнение с предыдущими снимками рынка.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src import config

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Анализирует тренды в спросе на навыки.
    Может сравнивать спрос по времени (если есть исторические снимки).
    """

    def __init__(self, current_frequencies: Dict[str, float], historical_dir: Optional[Path] = None):
        """
        Args:
            current_frequencies: Текущие веса/частоты навыков
            historical_dir: Папка с историческими снимками (по умолчанию config.HISTORY_DIR)
        """
        self.current = current_frequencies or {}
        self.history_dir = historical_dir or config.HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, frequencies: Dict[str, float], label: str = None):
        """Сохраняет текущий снимок рынка."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"freq_{label or timestamp}.json"
        path = self.history_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        logger.info(f"Снимок рынка сохранён: {path}")

    def _load_previous_snapshot(self) -> Dict[str, float]:
        """Загружает самый последний снимок (кроме текущего)."""
        files = sorted(self.history_dir.glob("freq_*.json"), reverse=True)
        if len(files) >= 2:
            with open(files[1], 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_trending_skills(self, top_n: int = 20, min_change_percent: float = 10.0) -> Dict[str, List[Dict]]:
        """
        Возвращает растущие и падающие навыки на основе сравнения с предыдущим снимком.
        Если исторических данных нет, возвращает пустые списки.
        """
        prev = self._load_previous_snapshot()
        if not prev:
            logger.warning("Нет исторических данных для анализа трендов")
            return {"rising": [], "falling": []}

        rising = []
        falling = []
        for skill, current_freq in self.current.items():
            prev_freq = prev.get(skill, 0)
            if prev_freq == 0:
                continue
            change_pct = ((current_freq - prev_freq) / prev_freq) * 100
            if change_pct >= min_change_percent:
                rising.append({
                    "skill": skill,
                    "current_freq": current_freq,
                    "prev_freq": prev_freq,
                    "change_pct": round(change_pct, 1),
                    "trend": "RISING"
                })
            elif change_pct <= -min_change_percent:
                falling.append({
                    "skill": skill,
                    "current_freq": current_freq,
                    "prev_freq": prev_freq,
                    "change_pct": round(change_pct, 1),
                    "trend": "FALLING"
                })

        rising.sort(key=lambda x: x["change_pct"], reverse=True)
        falling.sort(key=lambda x: x["change_pct"])

        return {
            "rising": rising[:top_n],
            "falling": falling[:top_n]
        }

    def get_emerging_skills(self, min_weight: float = 0.01, top_n: int = 20) -> List[Dict]:
        """Возвращает навыки с низкой частотой, которые могут быть на подъёме."""
        max_weight = max(self.current.values()) if self.current else 1.0
        emerging = []
        for skill, weight in self.current.items():
            norm_weight = weight / max_weight
            if 0 < norm_weight < min_weight:
                potential = "STABLE"
                if any(kw in skill.lower() for kw in ["ai", "machine", "llm", "gpt", "mlops"]):
                    potential = "RISING"
                elif any(kw in skill.lower() for kw in ["cloud", "aws", "azure", "k8s"]):
                    potential = "RISING"
                emerging.append({
                    "skill": skill,
                    "weight": weight,
                    "normalized": round(norm_weight, 4),
                    "potential": potential
                })
        emerging.sort(key=lambda x: x["weight"], reverse=True)
        return emerging[:top_n]

    def get_stable_skills(self, top_n: int = 20) -> List[Dict]:
        """Возвращает навыки с высокой стабильной частотой."""
        if not self.current:
            return []
        avg = sum(self.current.values()) / len(self.current)
        stable = []
        for skill, weight in self.current.items():
            if weight >= avg:
                stability = "CRITICAL" if weight >= avg * 2 else "STABLE"
                stable.append({
                    "skill": skill,
                    "weight": weight,
                    "stability": stability
                })
        stable.sort(key=lambda x: x["weight"], reverse=True)
        return stable[:top_n]


# ----------------------------------------------------------------------
# Блок для отладки и тестирования (__main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
    if not freq_path.exists():
        logger.error(f"Файл {freq_path} не найден. Сначала выполните сбор вакансий.")
        sys.exit(1)

    with open(freq_path, 'r', encoding='utf-8') as f:
        current_freq = json.load(f)

    analyzer = TrendAnalyzer(current_freq)

    # Сохраняем текущий снимок (если нужно)
    analyzer.save_snapshot(current_freq)

    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ ТРЕНДОВ НАВЫКОВ")
    print("=" * 70)

    trends = analyzer.get_trending_skills(top_n=10)
    if trends["rising"]:
        print("\n📈 РАСТУЩИЕ НАВЫКИ (по сравнению с предыдущим снимком):")
        for t in trends["rising"]:
            print(f"  • {t['skill']:<30} +{t['change_pct']:.1f}% (было {t['prev_freq']}, стало {t['current_freq']})")
    else:
        print("\n📈 РАСТУЩИЕ НАВЫКИ: нет данных (нужен предыдущий снимок)")

    if trends["falling"]:
        print("\n📉 ПАДАЮЩИЕ НАВЫКИ:")
        for t in trends["falling"]:
            print(f"  • {t['skill']:<30} {t['change_pct']:.1f}% (было {t['prev_freq']}, стало {t['current_freq']})")

    emerging = analyzer.get_emerging_skills(min_weight=0.05, top_n=10)
    if emerging:
        print("\n🌱 ПОТЕНЦИАЛЬНО РАСТУЩИЕ НАВЫКИ (низкая частота, но могут быть в тренде):")
        for e in emerging:
            print(f"  • {e['skill']:<30} вес: {e['weight']} (потенциал: {e['potential']})")

    stable = analyzer.get_stable_skills(top_n=10)
    if stable:
        print("\n💪 СТАБИЛЬНЫЕ ВЫСОКОЧАСТОТНЫЕ НАВЫКИ:")
        for s in stable:
            print(f"  • {s['skill']:<30} вес: {s['weight']} ({s['stability']})")

    print("\n" + "=" * 70)