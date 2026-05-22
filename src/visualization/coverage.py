"""Графики покрытия рынка и профессий."""

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
    prof_cov = [results[p].get("profession_coverage", 0) for p in profiles]

    fig, ax = plt.subplots(figsize=(18, 9))
    x = np.arange(len(profiles))
    width = 0.15

    bars1 = ax.bar(x - 2 * width, skill_cov, width, label="Покрытие навыков %", color="#2ca02c", alpha=0.9)
    bars2 = ax.bar(x - width, market_cov, width, label="Общее покрытие рынка %", color="#9467bd", alpha=0.9)
    bars3 = ax.bar(x, prof_cov, width, label="Покрытие по профессии %", color="#e74c3c", alpha=0.9)
    bars4 = ax.bar(x + width, readiness, width, label="Готовность к уровню %", color="#ff7f0e", alpha=0.9)
    bars5 = ax.bar(x + 2 * width, real_cov, width, label="Реальное покрытие рынка %", color="#1f77b4", alpha=0.9)

    ax.set_title("Сравнение профилей: покрытие, готовность и соответствие профессии", pad=20)
    ax.set_ylabel("Процент")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=11)

    for bars in (bars1, bars2, bars3, bars4, bars5):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
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


def plot_profession_coverage(results: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    """График покрытия целевой профессии по доменам."""
    logger.info("plotting_profession_coverage", profiles=len(results))

    all_domains = set()
    profile_data = {}
    for pname, ev in results.items():
        target = ev.get("target_profession", pname)
        detail = ev.get("profession_coverage_detail", {})
        if detail:
            profile_data[pname] = {"target": target, "domains": detail}
            all_domains.update(detail.keys())

    if not profile_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет данных о покрытии профессий", ha="center", transform=ax.transAxes)
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)
        return fig

    all_domains_sorted = sorted(all_domains)
    profiles_sorted = list(profile_data.keys())
    data = []
    for pname in profiles_sorted:
        row = [profile_data[pname]["domains"].get(d, 0) for d in all_domains_sorted]
        data.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(all_domains_sorted) * 1.2), len(profiles_sorted) * 1.2))
    df = pd.DataFrame(data, index=profiles_sorted, columns=all_domains_sorted)
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                cbar_kws={"label": "Покрытие домена %"}, ax=ax)
    targets = [profile_data[p]["target"] for p in profiles_sorted]
    ax.set_title("Покрытие доменов целевой профессии", pad=20)
    ax.set_ylabel("Профиль (цель)")
    ax.set_xlabel("Домен")
    ax.set_yticklabels([f"{p} ({t})" for p, t in zip(profiles_sorted, targets)], rotation=0)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("profession_coverage_saved", path=str(save_path))
    plt.close(fig)
    return fig


def plot_domain_skill_gaps(results: dict[str, Any], save_path: Path | None = None) -> plt.Figure:
    """График распределения gaps (strong/weak/missing) для целевого домена."""
    logger.info("plotting_domain_skill_gaps", profiles=len(results))

    profiles = list(results.keys())
    strong_vals = []
    weak_vals = []
    missing_vals = []

    for p in profiles:
        cats = results[p].get("skill_categories", {})
        total = cats.get("total", 1)
        strong_vals.append(cats.get("strong", 0) / total * 100)
        weak_vals.append(cats.get("weak", 0) / total * 100)
        missing_vals.append(cats.get("missing", 0) / total * 100)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(profiles))
    width = 0.4

    bars1 = ax.bar(x, strong_vals, width, label="Сильные (gap < 20%)", color="#27ae60", alpha=0.85)
    bars2 = ax.bar(x, weak_vals, width, bottom=strong_vals, label="Слабые (gap 20-60%)", color="#f39c12", alpha=0.85)
    bars3 = ax.bar(x, missing_vals, width, bottom=[s + w for s, w in zip(strong_vals, weak_vals)],
                   label="Отсутствуют (gap > 60%)", color="#e74c3c", alpha=0.85)

    ax.set_title("Распределение навыков по категориям gap'а (целевой домен)", pad=15)
    ax.set_ylabel("Процент навыков")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.legend(loc="upper right", fontsize=11)

    for i, (s, w, m) in enumerate(zip(strong_vals, weak_vals, missing_vals)):
        ax.text(i, s + w + m + 2, f"{s:.0f}/{w:.0f}/{m:.0f}%", ha="center", fontsize=10, fontweight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("domain_skill_gaps_saved", path=str(save_path))
    plt.close(fig)
    return fig
