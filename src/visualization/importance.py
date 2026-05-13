"""Графики важности навыков."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import structlog

from ._utils import load_ml_recommendations

logger = structlog.get_logger(__name__)


def plot_ml_importance(profile: str, top_n: int = 10, save_path: Path | None = None) -> plt.Figure:
    """Горизонтальный барплот важности недостающих навыков по ML-модели."""
    logger.info("plotting_ml_importance", profile=profile, top_n=top_n)
    recs = load_ml_recommendations(profile)
    if not recs:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            f"Нет ML-рекомендаций для '{profile}'\n(модель не обучена или нет данных)",
            ha="center",
            va="center",
            fontsize=16,
        )
        if save_path:
            plt.savefig(save_path)
        logger.warning("no_ml_recommendations_for_plot", profile=profile)
        plt.close(fig)
        return fig

    skills = [r[0] for r in recs[:top_n]]
    scores = [r[1] for r in recs[:top_n]]

    fig, ax = plt.subplots()
    bars = ax.barh(skills[::-1], scores[::-1], color=sns.color_palette("viridis", len(skills)))

    ax.set_title(f"ТОП важных навыков (LTR XGBoost) — {profile}", pad=20)
    ax.set_xlabel("Важность навыка (%)")
    ax.set_xlim(0, 105)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    if save_path:
        plt.savefig(save_path)
        logger.info("ml_importance_saved", path=str(save_path))
    plt.close(fig)
    return fig


def plot_weight_distribution(
    weights: dict[str, float],
    title: str = "Топ-15 навыков по рыночному весу",
    save_path: Path | None = None,
) -> plt.Figure:
    """Распределение весов навыков (горизонтальный барплот топ-15)."""
    if not weights:
        logger.warning("no_weights_for_plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет весов навыков", ha="center")
        plt.close(fig)
        return fig

    df = pd.DataFrame(list(weights.items()), columns=["skill", "weight"])
    df = df.sort_values("weight", ascending=False).head(15)

    fig, ax = plt.subplots()
    sns.barplot(data=df, y="skill", x="weight", hue="skill", palette="viridis", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Вес (нормализованный)")

    if save_path:
        plt.savefig(save_path)
        logger.info("weight_distribution_saved", path=str(save_path))
    plt.close(fig)
    return fig
