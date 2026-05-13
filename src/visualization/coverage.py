"""Графики покрытия рынка."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog

from ._utils import load_skill_weights

logger = structlog.get_logger(__name__)


def plot_coverage_comparison(results: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    logger.info("plotting_coverage_comparison", profiles=len(results))
    profiles = list(results.keys())
    market_cov = [results[p].get("market_coverage_score", 0) for p in profiles]
    skill_cov = [results[p].get("skill_coverage", 0) for p in profiles]
    readiness = [results[p].get("readiness_score", 0) for p in profiles]
    real_cov = [results[p].get("market_skill_coverage", 0) for p in profiles]

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(profiles))
    width = 0.2

    bars1 = ax.bar(x - 1.5 * width, skill_cov, width, label="Покрытие навыков %", color="#2ca02c", alpha=0.9)
    bars2 = ax.bar(x - 0.5 * width, market_cov, width, label="Общее покрытие рынка %", color="#9467bd", alpha=0.9)
    bars3 = ax.bar(x + 0.5 * width, readiness, width, label="Готовность к уровню %", color="#ff7f0e", alpha=0.9)
    bars4 = ax.bar(x + 1.5 * width, real_cov, width, label="Реальное покрытие рынка %", color="#1f77b4", alpha=0.9)

    ax.set_title("Сравнение профилей: покрытие и готовность", pad=20)
    ax.set_ylabel("Процент")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=12)

    for bars in (bars1, bars2, bars3, bars4):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    if save_path:
        plt.savefig(save_path)
        logger.info("coverage_comparison_saved", path=str(save_path))
    plt.close(fig)
    return fig


def plot_skills_heatmap(results: dict[str, Any], top_n: int = 20, save_path: Path | None = None) -> plt.Figure:
    logger.info("plotting_skills_heatmap", profiles=len(results), top_n=top_n)
    skill_weights = load_skill_weights()
    if not skill_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет данных о весах", ha="center")
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)
        return fig

    top_skills = [s for s, _ in sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    profiles = list(results.keys())
    data = []
    for profile in profiles:
        eval_dict = results[profile]
        student_set = set(s.lower() for s in eval_dict.get("student_skills", []))
        row = [1 if skill.lower() in student_set else 0 for skill in top_skills]
        data.append(row)

    df = pd.DataFrame(data, index=profiles, columns=top_skills)
    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.4), len(profiles) * 0.8))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Наличие навыка"}, ax=ax)
    ax.set_title(f"Покрытие топ-{top_n} рыночных навыков", pad=20)
    ax.set_xlabel("Навыки")
    ax.set_ylabel("Профиль")
    plt.xticks(rotation=45, ha="right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("skills_heatmap_saved", path=str(save_path))
    plt.close(fig)
    return fig
