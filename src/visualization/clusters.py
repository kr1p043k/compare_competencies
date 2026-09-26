"""Визуализация ближайших кластеров вакансий."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog

from ._config import EMOJI_TO_TEXT

logger = structlog.get_logger(__name__)


def plot_cluster_insights(results: dict[str, Any], output_dir: Path):
    """Для каждого профиля отображает ближайшие кластеры и покрытие навыков."""
    for profile_name, eval_dict in results.items():
        cluster_ctx = eval_dict.get("cluster_context")
        if not cluster_ctx:
            logger.info("no_cluster_context_for_profile", profile=profile_name)
            continue

        # Кросс-уровневый пул даёт до 30 кандидатов — показываем топ-15,
        # иначе подписи сливаются. Горизонтальные бары: длинные имена не налезают.
        closest = (cluster_ctx.get("closest_clusters", []) or [])[:15]
        if not closest:
            continue

        logger.info("plotting_cluster_insights", profile=profile_name, clusters=len(closest))

        student_skills = set(s.lower() for s in eval_dict.get("student_skills", []))
        cluster_skills_map = cluster_ctx.get("skills", {})
        cluster_skills_set = set(cluster_skills_map.keys())

        cluster_names = []
        for c in closest:
            name = c.get("name", f"Cluster {c['id']}")
            if ":" in name:
                name = name.split(":")[0].strip()
            for emoji, text in EMOJI_TO_TEXT.items():
                name = name.replace(emoji, text)
            if len(name) > 42:
                name = name[:41] + "…"
            cluster_names.append(name)

        similarities = [c["similarity"] * 100 for c in closest]

        if student_skills and cluster_skills_set:
            coverage = len(student_skills & cluster_skills_set) / len(student_skills) * 100
        else:
            coverage = 0.0

        fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(closest))))
        y = np.arange(len(closest))

        bars = ax.barh(y, similarities, 0.55, color="#1f77b4", alpha=0.85, label="Близость к профилю")
        ax.axvline(x=coverage, color="#2ca02c", linestyle="--", linewidth=2, label=f"Покрытие навыков: {coverage:.1f}%")

        ax.set_title(f"Ближайшие кластеры вакансий: {profile_name} (топ-{len(closest)})", pad=15, fontsize=14)
        ax.set_yticks(y)
        ax.set_yticklabels(cluster_names, fontsize=10)
        ax.set_xlabel("Сходство (%)", fontsize=12)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        ax.legend(fontsize=11)

        for bar, val in zip(bars, similarities):
            width = bar.get_width()
            ax.text(
                width + 1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        save_path = output_dir / profile_name / f"cluster_insights_{profile_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("cluster_insights_saved", path=str(save_path))
