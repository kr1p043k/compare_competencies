"""
Анализ трендов: какие навыки растут/падают в спросе.
Поддерживает сравнение с предыдущими снимками рынка и построение графиков.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src import config

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Анализатор трендов с поддержкой множественных исторических снимков."""

    def __init__(self, current_frequencies: Dict[str, float], 
                 historical_dir: Optional[Path] = None):
        self.current = current_frequencies or {}
        self.history_dir = historical_dir or config.HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Работа со снимками
    # ------------------------------------------------------------------
    def save_snapshot(self, frequencies: Dict[str, float], label: str = None,
                      apply_whitelist: bool = True) -> Path:
        """Сохраняет текущий снимок и возвращает путь к файлу."""
        if apply_whitelist:
            from src.parsing.skill_validator import SkillValidator
            validator = SkillValidator()
            filtered = {skill: freq for skill, freq in frequencies.items()
                       if validator.validate(skill).is_valid}
            frequencies = filtered
            logger.info(f"После фильтрации осталось {len(frequencies)} навыков")

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"freq_{label or timestamp}.json"
        path = self.history_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(frequencies, f, ensure_ascii=False, indent=2)
        logger.info(f"Снимок сохранён: {path}")
        return path

    def load_all_snapshots(self) -> List[Tuple[datetime, Path, Dict[str, float]]]:
        """
        Загружает все снимки из history_dir.
        Возвращает список (datetime, путь, данные), отсортированный по дате.
        """
        files = sorted(self.history_dir.glob("freq_*.json"))
        snapshots = []
        for fpath in files:
            try:
                dt = self._extract_date(fpath)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                snapshots.append((dt, fpath, data))
            except Exception as e:
                logger.warning(f"Не удалось загрузить {fpath.name}: {e}")

        snapshots.sort(key=lambda x: x[0])
        return snapshots

    @staticmethod
    def _extract_date(filepath: Path) -> datetime:
        """Извлекает дату из имени файла вида freq_2026-04-15.json или freq_2026-04-15-120000.json."""
        stem = filepath.stem.replace("freq_", "")
        # Пробуем с временем
        formats = ["%Y-%m-%d-%H%M%S", "%Y-%m-%d", "%Y-%m-%d %H%M%S"]
        for fmt in formats:
            try:
                return datetime.strptime(stem, fmt)
            except ValueError:
                continue
        # Fallback — дата модификации файла
        return datetime.fromtimestamp(filepath.stat().st_mtime)

    @staticmethod
    def load_file(filepath: Path) -> Dict[str, float]:
        """Загружает один файл снимка."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Выбор снимков по параметру --history
    # ------------------------------------------------------------------
    def get_snapshots_for_analysis(self, n: int = None) -> List[Tuple[datetime, Path, Dict[str, float]]]:
        """
        Возвращает список снимков для анализа:
        - Если n=None — все доступные снимки
        - Если n указано — последние n снимков
        """
        all_snapshots = self.load_all_snapshots()
        if n is None or n >= len(all_snapshots):
            return all_snapshots
        return all_snapshots[-n:]

    # ------------------------------------------------------------------
    # Анализ трендов
    # ------------------------------------------------------------------
    def get_trending_skills(self, top_n: int = 20, min_change_percent: float = 5.0,
                            previous_snapshot: Dict = None, prev_label: str = "предыдущий") -> Dict[str, List[Dict]]:
        """
        Сравнивает текущий снимок с предыдущим.
        """
        if previous_snapshot is None:
            snapshots = self.load_all_snapshots()
            if len(snapshots) < 2:
                logger.warning("Слишком мало снимков для анализа трендов")
                return {"rising": [], "falling": []}
            prev_dt, _, prev_data = snapshots[-2]
            prev_label = prev_dt.strftime("%Y-%m-%d")

        prev_data = previous_snapshot
        rising, falling = [], []

        for skill, current_freq in self.current.items():
            prev_freq = prev_data.get(skill, 0)
            if prev_freq == 0:
                continue
            change_pct = ((current_freq - prev_freq) / prev_freq) * 100
            entry = {
                "skill": skill,
                "current_freq": current_freq,
                "prev_freq": prev_freq,
                "change_pct": round(change_pct, 1),
                "prev_label": prev_label
            }
            if change_pct >= min_change_percent:
                entry["trend"] = "RISING"
                rising.append(entry)
            elif change_pct <= -min_change_percent:
                entry["trend"] = "FALLING"
                falling.append(entry)

        rising.sort(key=lambda x: x["change_pct"], reverse=True)
        falling.sort(key=lambda x: x["change_pct"])
        return {"rising": rising[:top_n], "falling": falling[:top_n]}

    def get_emerging_skills(self, min_weight: float = 0.01, top_n: int = 20) -> List[Dict]:
        max_weight = max(self.current.values()) if self.current else 1.0
        emerging = []
        for skill, weight in self.current.items():
            norm_weight = weight / max_weight
            if 0 < norm_weight < min_weight:
                potential = "STABLE"
                if any(kw in skill.lower() for kw in ["ai", "llm", "gpt", "mlops", "langchain", "rag"]):
                    potential = "RISING"
                elif any(kw in skill.lower() for kw in ["cloud", "aws", "azure", "k8s", "kubernetes"]):
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
        if not self.current:
            return []
        avg = sum(self.current.values()) / max(len(self.current), 1)
        stable = []
        for skill, weight in self.current.items():
            if weight >= avg:
                stability = "CRITICAL" if weight >= avg * 2 else "STABLE"
                stable.append({"skill": skill, "weight": weight, "stability": stability})
        stable.sort(key=lambda x: x["weight"], reverse=True)
        return stable[:top_n]

    # ------------------------------------------------------------------
    # Временные ряды
    # ------------------------------------------------------------------
    def get_skill_timeline(self, skills: List[str],
                           snapshots: List[Tuple[datetime, Path, Dict]]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Строит временной ряд для каждого навыка по переданным снимкам."""
        timeline = {skill: [] for skill in skills}
        for dt, _, data in snapshots:
            for skill in skills:
                timeline[skill].append((dt, data.get(skill, 0)))
        return timeline

    # ------------------------------------------------------------------
    # Графики
    # ------------------------------------------------------------------
    def plot_timeline(self, skills: List[str], snapshots: List[Tuple[datetime, Path, Dict]],
                      save_path: Optional[Path] = None,
                      title: str = "Динамика спроса на навыки"):
        """Линейный график изменения частот навыков по всем выбранным снимкам."""
        if not snapshots or len(snapshots) < 2:
            logger.warning("Недостаточно снимков для графика")
            return None

        timeline = self.get_skill_timeline(skills, snapshots)

        fig, ax = plt.subplots(figsize=(18, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(skills)))
        for idx, (skill, points) in enumerate(timeline.items()):
            if not points:
                continue
            dates = [p[0] for p in points]
            freqs = [p[1] for p in points]
            ax.plot(dates, freqs, marker='o', linewidth=2.5, markersize=10,
                    color=colors[idx], label=skill, alpha=0.85)

        ax.set_title(title, fontsize=18, pad=20)
        ax.set_xlabel("Дата снимка", fontsize=14)
        ax.set_ylabel("Частота (количество вакансий)", fontsize=14)
        ax.legend(loc='upper left', fontsize=11, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ График временных рядов сохранён: {save_path}")
        plt.close(fig)
        return fig

    def plot_trending(self, top_n: int = 15, save_path: Optional[Path] = None,
                      previous_snapshot: Dict = None):
        """Горизонтальный бар-чарт растущих и падающих навыков."""
        trends = self.get_trending_skills(top_n=top_n, min_change_percent=3.0,
                                         previous_snapshot=previous_snapshot)
        rising = trends['rising'][:top_n]
        falling = trends['falling'][:top_n]

        if not rising and not falling:
            logger.info("Нет значимых трендов для отображения (порог 3%)")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, (len(rising) + len(falling)) * 0.4)))

        # Растущие
        if rising:
            skills_r = [r['skill'] for r in rising][::-1]
            values_r = [r['change_pct'] for r in rising][::-1]
            colors_r = ['#2ca02c' if v > 0 else '#ff7f0e' for v in values_r]
            bars = ax1.barh(skills_r, values_r, color=colors_r, edgecolor='white')
            ax1.set_title('Растущие навыки ↑', fontsize=14, pad=10)
            ax1.set_xlabel('Изменение (%)')
            ax1.axvline(x=0, color='black', linewidth=0.8)
            for bar, val in zip(bars, values_r):
                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Изменения меньше порога', ha='center', va='center', fontsize=14)
            ax1.set_title('Растущие навыки ↑', fontsize=14)

        # Падающие
        if falling:
            skills_f = [f['skill'] for f in falling][::-1]
            values_f = [f['change_pct'] for f in falling][::-1]
            colors_f = ['#d62728' if v < 0 else '#ff7f0e' for v in values_f]
            bars = ax2.barh(skills_f, values_f, color=colors_f, edgecolor='white')
            ax2.set_title('Падающие навыки ↓', fontsize=14, pad=10)
            ax2.set_xlabel('Изменение (%)')
            ax2.axvline(x=0, color='black', linewidth=0.8)
            for bar, val in zip(bars, values_f):
                ax2.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', ha='right' if val < 0 else 'left', fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Изменения меньше порога', ha='center', va='center', fontsize=14)
            ax2.set_title('Падающие навыки ↓', fontsize=14)

        plt.suptitle("Тренды навыков (сравнение двух снимков)", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ График трендов сохранён: {save_path}")
        plt.close(fig)
        return fig


# ============================== CLI ==============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Анализ трендов навыков по историческим снимкам")
    parser.add_argument('--current', '-c', type=Path, help="Путь к файлу текущего снимка")
    parser.add_argument('--previous', '-p', type=Path, help="Путь к файлу предыдущего снимка для сравнения")
    parser.add_argument('--history', '-n', type=int, default=None,
                        help="Количество последних снимков для анализа (если не указано — все)")
    parser.add_argument('--no-save', action='store_true', help="Не сохранять новый снимок")
    parser.add_argument('--top', type=int, default=10, help="Количество выводимых навыков (default 10)")
    parser.add_argument('--min-change', type=float, default=3.0,
                        help="Минимальный процент изменения для тренда (default 3.0)")
    args = parser.parse_args()

    # === Загрузка текущего снимка ===
    if args.current:
        current_freq = TrendAnalyzer.load_file(args.current)
        logger.info(f"Загружен текущий снимок: {args.current}")
    else:
        freq_path = config.DATA_PROCESSED_DIR / "competency_frequency.json"
        if not freq_path.exists():
            logger.error(f"Файл {freq_path} не найден. Сначала выполните сбор вакансий.")
            sys.exit(1)
        current_freq = TrendAnalyzer.load_file(freq_path)

    analyzer = TrendAnalyzer(current_freq)

    # === Сохранение (опционально) ===
    if not args.no_save:
        analyzer.save_snapshot(current_freq)

    # === Выбор снимков для анализа ===
    snapshots = analyzer.get_snapshots_for_analysis(args.history)
    logger.info(f"Для анализа выбрано {len(snapshots)} снимков из {len(analyzer.load_all_snapshots())} доступных")

    # === Предыдущий снимок ===
    previous_snapshot = None
    if args.previous:
        previous_snapshot = TrendAnalyzer.load_file(args.previous)
        logger.info(f"Загружен предыдущий снимок: {args.previous}")
    elif len(snapshots) >= 2:
        # Берём предпоследний из выбранных (или самый последний исторический)
        previous_snapshot = snapshots[-2][2]
        logger.info(f"Автоматически выбран предыдущий снимок: {snapshots[-2][0].strftime('%Y-%m-%d')}")

    # === Вывод в консоль ===
    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ ТРЕНДОВ НАВЫКОВ")
    print(f"   Анализируется снимков: {len(snapshots)}")
    print("=" * 70)

    # Тренды
    trends = analyzer.get_trending_skills(top_n=args.top, min_change_percent=args.min_change,
                                         previous_snapshot=previous_snapshot)
    if trends["rising"]:
        print(f"\n📈 РАСТУЩИЕ НАВЫКИ:")
        for t in trends["rising"]:
            print(f"  • {t['skill']:<35} +{t['change_pct']:.1f}% (было {t['prev_freq']}, стало {t['current_freq']})")
    else:
        print(f"\n📈 РАСТУЩИЕ НАВЫКИ: нет (изменения меньше {args.min_change}%)")

    if trends["falling"]:
        print(f"\n📉 ПАДАЮЩИЕ НАВЫКИ:")
        for t in trends["falling"]:
            print(f"  • {t['skill']:<35} {t['change_pct']:.1f}% (было {t['prev_freq']}, стало {t['current_freq']})")
    else:
        print(f"\n📉 ПАДАЮЩИЕ НАВЫКИ: нет (изменения меньше {args.min_change}%)")

    # Emerging
    emerging = analyzer.get_emerging_skills(min_weight=0.03, top_n=args.top)
    if emerging:
        print(f"\n🌱 ПОТЕНЦИАЛЬНО РАСТУЩИЕ НАВЫКИ:")
        for e in emerging:
            print(f"  • {e['skill']:<35} вес: {e['weight']} (потенциал: {e['potential']})")

    # Stable
    stable = analyzer.get_stable_skills(top_n=args.top)
    if stable:
        print(f"\n💪 СТАБИЛЬНЫЕ ВЫСОКОЧАСТОТНЫЕ НАВЫКИ:")
        for s in stable:
            print(f"  • {s['skill']:<35} вес: {s['weight']} ({s['stability']})")

    # === Графики ===
    output_dir = config.DATA_DIR / "result" / "trends"
    output_dir.mkdir(parents=True, exist_ok=True)

    # График трендов
    analyzer.plot_trending(top_n=args.top, save_path=output_dir / "trending_skills.png",
                          previous_snapshot=previous_snapshot)

    # Временные ряды для топ-N навыков по всем выбранным снимкам
    if len(snapshots) >= 2:
        top_skills_list = [s['skill'] for s in stable[:min(args.top, 10)]]
        analyzer.plot_timeline(
            top_skills_list,
            snapshots=snapshots,
            save_path=output_dir / "skill_timeline.png",
            title=f"Динамика спроса на ключевые навыки ({len(snapshots)} снимков)"
        )
        print(f"\n📊 Графики сохранены в {output_dir}")
    else:
        print("\n⚠️  Недостаточно снимков для построения графика (нужно ≥ 2)")

    print("=" * 70)