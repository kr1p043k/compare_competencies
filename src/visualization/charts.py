# src/visualization/charts.py
"""
Модуль визуализации результатов gap-анализа и рекомендаций.
Генерирует презентационные графики в высоком качестве (DPI 300).
Включает запуск Jupyter ноутбуков и вывод контекстной информации.
"""

import subprocess
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Добавляем корень проекта в путь для импорта config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config

# ==================== КРАСИВЫЙ СТИЛЬ ДЛЯ ПРЕЗЕНТАЦИИ ====================
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.4)
plt.rcParams.update({
    'figure.figsize': (14, 9),
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.family': 'sans-serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Функции загрузки данных
# ----------------------------------------------------------------------
def load_skill_weights() -> Dict[str, float]:
    """Загружает skill_weights из data/processed/skill_weights.json."""
    path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить {path}: {e}")
    return {}


def load_hybrid_weights() -> Dict[str, float]:
    """Загружает гибридные веса (если есть)."""
    path = config.DATA_PROCESSED_DIR / "hybrid_weights.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить {path}: {e}")
    return {}


def load_ml_recommendations(profile: str) -> List[Tuple[str, float, str]]:
    """Загружает ML-рекомендации для профиля из любого подходящего файла."""
    possible_names = [
        f"ltr_recommendations_{profile}.json",
        f"ml_recommendations_{profile}.json",
        f"full_recommendations_{profile}.json",
    ]
    for fname in possible_names:
        path = config.DATA_DIR / "result" / profile / fname
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                recs = []
                for r in data.get("recommendations", []):
                    skill = r.get("skill", "")
                    # пробуем разные ключи важности
                    score = r.get("importance_score", r.get("importance", r.get("score", 0.0)))
                    explanation = r.get("explanation", r.get("why_important", ""))
                    recs.append((skill, score, explanation))
                return recs
            except Exception as e:
                logger.warning(f"Ошибка загрузки ML-рекомендаций из {path}: {e}")
    return []

# ----------------------------------------------------------------------
# Основные графики
# ----------------------------------------------------------------------

def plot_coverage_comparison(results: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    profiles = list(results.keys())
    simple_cov = [results[p].get('simple_coverage', 0) for p in profiles]
    hybrid_cov = [results[p].get('weighted_hybrid_coverage', results[p].get('weighted_coverage', 0)) for p in profiles]
    readiness = [results[p].get('readiness_score', 0) for p in profiles]

    fig, ax = plt.subplots(figsize=(14, 9))
    x = np.arange(len(profiles))
    width = 0.25

    bars1 = ax.bar(x - width, simple_cov, width, label='Покрытие (простое) %', color='#2ca02c', alpha=0.9)
    bars2 = ax.bar(x, hybrid_cov, width, label='Покрытие (гибридное) %', color='#9467bd', alpha=0.9)
    bars3 = ax.bar(x + width, readiness, width, label='Готовность к уровню %', color='#ff7f0e', alpha=0.9)

    ax.set_title("Сравнение профилей: покрытие и готовность", pad=20)
    ax.set_ylabel("Процент")
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    if save_path:
        plt.savefig(save_path)
        logger.info(f"✅ coverage_comparison сохранён → {save_path}")
    plt.close(fig)
    return fig


def plot_skill_comparison_radar(
    student_skills: List[str],
    market_top: List[str],
    student_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Радарная диаграмма: навыки студента против топ-навыков рынка.
    """
    # Ограничиваем количество навыков для читаемости
    all_skills = list(dict.fromkeys(market_top[:12] + student_skills))
    if len(all_skills) > 15:
        all_skills = all_skills[:15]

    num_vars = len(all_skills)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

    # Студент
    student_vals = [1 if skill in student_skills else 0 for skill in all_skills]
    student_vals += student_vals[:1]
    ax.plot(angles, student_vals, linewidth=3, linestyle="solid", label=student_name, color="#1f77b4")
    ax.fill(angles, student_vals, color="#1f77b4", alpha=0.25)

    # Рынок
    market_vals = [1 if skill in market_top else 0 for skill in all_skills]
    market_vals += market_vals[:1]
    ax.plot(angles, market_vals, linewidth=3, linestyle="solid", label="Топ рынка", color="#d62728")
    ax.fill(angles, market_vals, color="#d62728", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), all_skills, fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Сравнение навыков\n{student_name} vs Требования рынка", pad=30, fontsize=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=14)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"✅ radar сохранён → {save_path}")
    plt.close(fig)
    return fig


def plot_ml_importance(profile: str, top_n: int = 10, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Горизонтальный барплот важности недостающих навыков по ML-модели.
    """
    recs = load_ml_recommendations(profile)
    if not recs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Нет ML-рекомендаций для '{profile}'\n(модель не обучена или нет данных)",
                ha="center", va="center", fontsize=16)
        if save_path:
            plt.savefig(save_path)
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
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                va="center", fontsize=13, fontweight="bold")

    if save_path:
        plt.savefig(save_path)
        logger.info(f"✅ ml_importance сохранён → {save_path}")
    plt.close(fig)
    return fig


def plot_weight_distribution(weights: Dict[str, float], title: str = "Топ-15 навыков по рыночному весу",
                             save_path: Optional[Path] = None) -> plt.Figure:
    """Распределение весов навыков (горизонтальный барплот топ-15)."""
    if not weights:
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
    plt.close(fig)
    return fig
def plot_skills_heatmap(results: Dict[str, Any], top_n: int = 20, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Тепловая карта покрытия топ-N рыночных навыков разными профилями.
    """
    # Загружаем топ-N навыков из skill_weights
    skill_weights = load_skill_weights()
    if not skill_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет данных о весах", ha='center')
        if save_path:
            plt.savefig(save_path)
        return fig

    top_skills = [s for s, _ in sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    profiles = list(results.keys())
    
    # Строим матрицу: 1 если навык есть у студента, иначе 0
    data = []
    for profile in profiles:
        student_skills = results[profile].get('student_skills', [])
        student_set = set(s.lower() for s in student_skills)
        row = [1 if skill.lower() in student_set else 0 for skill in top_skills]
        data.append(row)
    
    df = pd.DataFrame(data, index=profiles, columns=top_skills)
    
    fig, ax = plt.subplots(figsize=(max(12, top_n*0.4), len(profiles)*0.8))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Наличие навыка'}, ax=ax)
    ax.set_title(f"Покрытие топ-{top_n} рыночных навыков", pad=20)
    ax.set_xlabel("Навыки")
    ax.set_ylabel("Профиль")
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"✅ heatmap сохранён → {save_path}")
    plt.close(fig)
    return fig

def plot_cluster_insights(results: Dict[str, Any], output_dir: Path):
    """
    Для каждого профиля отображает ближайшие кластеры и покрытие в них.
    """
    for profile_name, data in results.items():
        cluster_info = data.get('cluster_info')
        if not cluster_info:
            continue

        clusters = cluster_info.get('clusters', [])
        if not clusters:
            continue

        cluster_ids = [f"Кластер {c['cluster_id']}" for c in clusters]
        similarities = [c['similarity'] * 100 for c in clusters]
        coverages = [c['coverage'] * 100 for c in clusters]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(clusters))
        width = 0.35

        bars1 = ax.bar(x - width/2, similarities, width, label='Сходство (%)', color='#1f77b4')
        bars2 = ax.bar(x + width/2, coverages, width, label='Покрытие (%)', color='#2ca02c')

        ax.set_title(f"Ближайшие кластеры вакансий — {profile_name}", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cluster_ids, rotation=45, ha='right')
        ax.set_ylabel('Процент')
        ax.legend()

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        save_path = output_dir / profile_name / f"cluster_insights_{profile_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ cluster_insights сохранён → {save_path}")

def save_all_charts(results: Dict[str, Any], output_dir: Path, use_ml: bool = True):
    """
    Главная функция — сохраняет все презентационные графики в указанную папку.
    :param results: словарь с данными профилей (см. plot_coverage_comparison)
    :param output_dir: директория для сохранения (например, data/result/)
    :param use_ml: генерировать ли ML-графики важности
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🚀 Генерация презентационных графиков в {output_dir}")

    # 1. Общее сравнение покрытия
    plot_coverage_comparison(results, output_dir / "coverage_comparison.png")

    # Загружаем топ навыков рынка (для радаров)
    skill_weights = load_skill_weights()
    market_top = list(skill_weights.keys())[:15] if skill_weights else []

    # 2. Для каждого профиля
    for profile_name, data in results.items():
        prof_dir = output_dir / profile_name
        prof_dir.mkdir(exist_ok=True)

        student_skills = data.get("covered_skills", []) + data.get("student_skills", [])
        if market_top:
            plot_skill_comparison_radar(student_skills, market_top, profile_name.capitalize(),
                                        prof_dir / f"radar_{profile_name}.png")

        if use_ml:
            plot_ml_importance(profile_name, save_path=prof_dir / f"ml_importance_{profile_name}.png")

        plot_weight_distribution(skill_weights, save_path=prof_dir / f"weights_{profile_name}.png")

        if "high_demand_gaps" in data:
            deficits = data["high_demand_gaps"][:10]
            if deficits:
                skills = [d["skill"] for d in deficits]
                freqs = [d.get("frequency", 0) for d in deficits]
                fig, ax = plt.subplots()
                ax.barh(skills, freqs, color="salmon")
                ax.set_title(f"Топ-10 дефицитов высокого спроса — {profile_name}")
                ax.set_xlabel("Частота в вакансиях")
                ax.invert_yaxis()
                fig.savefig(prof_dir / f"deficits_{profile_name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

    # 3. Тепловая карта покрытия (вызывается один раз после цикла)
    plot_skills_heatmap(results, top_n=20, save_path=output_dir / "skills_heatmap.png")

    # 4. Инсайты по кластерам (вызывается один раз после цикла)
    plot_cluster_insights(results, output_dir)

    logger.info("✅ Все графики готовы для презентации (DPI 300, современный стиль, русские подписи)")
    


# ----------------------------------------------------------------------
# Утилиты для запуска ноутбуков и вывода контекстной информации
# ----------------------------------------------------------------------
def run_notebook(notebook_name: str, output_dir: Optional[Path] = None) -> bool:
    """
    Выполняет Jupyter ноутбук с помощью nbconvert и сохраняет результат.
    """
    base_dir = Path(__file__).parent.parent.parent
    notebook_path = base_dir / "notebook_jypiter" / notebook_name
    if not notebook_path.exists():
        logger.error(f"Ноутбук не найден: {notebook_path}")
        return False

    if output_dir is None:
        output_path = notebook_path
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / notebook_name

    try:
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(notebook_path),
            "--output", str(output_path),
            "--ExecutePreprocessor.timeout=600"
        ]
        logger.info(f"Запуск выполнения ноутбука: {notebook_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Ноутбук {notebook_name} успешно выполнен. Результат: {output_path}")
            return True
        else:
            logger.error(f"Ошибка при выполнении {notebook_name}: {result.stderr}")
            return False
    except Exception as e:
        logger.exception(f"Исключение при запуске ноутбука: {e}")
        return False


def show_context_info() -> None:
    """Выводит в консоль сводную информацию о текущих данных и рекомендации по запуску."""
    print("\n" + "=" * 80)
    print("КОНТЕКСТНАЯ ИНФОРМАЦИЯ О ПРОЕКТЕ")
    print("=" * 80)

    # Рыночные навыки
    market_file = config.DATA_PROCESSED_DIR / "competency_frequency.json"
    if market_file.exists():
        try:
            with open(market_file, 'r', encoding='utf-8') as f:
                market_skills = json.load(f)
            print(f"Рыночные навыки (частота): {len(market_skills)} уникальных")
            top_skills = sorted(market_skills.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"Топ-5 востребованных: {', '.join([s for s, _ in top_skills])}")
        except Exception as e:
            print(f"Не удалось загрузить рыночные навыки: {e}")
    else:
        print("⚠️ Файл с рыночными навыками не найден. Сначала соберите данные.")

    # Маппинг компетенций
    mapping_file = config.COMPETENCY_MAPPING_FILE
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"Компетенций в маппинге: {len(mapping)}")
        except Exception as e:
            print(f"Не удалось загрузить маппинг компетенций: {e}")
    else:
        print(f"⚠️ Файл маппинга не найден: {mapping_file}")

    # Профили студентов
    students_dir = config.STUDENTS_DIR
    students = list(students_dir.glob("*_competency.json"))
    print(f"Профили студентов (JSON): {len(students)}")
    for student_file in students:
        try:
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                skills = data.get("навыки", [])
                print(f"  - {student_file.stem.replace('_competency', '')}: {len(skills)} компетенций")
        except Exception as e:
            print(f"  - Ошибка чтения {student_file.name}: {e}")

    # Результаты анализа
    results_dir = config.DATA_DIR / "result"
    reports_found = 0
    if results_dir.exists():
        for student_dir in results_dir.glob("*"):
            if student_dir.is_dir():
                report_file = student_dir / f"comparison_report_{student_dir.name}.json"
                if report_file.exists():
                    reports_found += 1
    print(f"Готовые отчёты gap-анализа: {reports_found}")

    # Рекомендуемые действия
    print("\n📋 РЕКОМЕНДАЦИИ ПО ЗАПУСКУ:")
    if not market_file.exists():
        print("  - Соберите рыночные данные: python main.py --it-sector --excel")
    elif not mapping_file.exists():
        print("  - Создайте файл competency_mapping.json в data/processed/")
    elif not students:
        print("  - Загрузите профили студентов через student_loader.py или вручную в data/students/")
    elif reports_found == 0:
        print("  - Запустите gap-анализ: python main.py --run-gap-analysis")
    else:
        print("  - Все данные в порядке. Графики уже можно сгенерировать:")
        print("    python -c \"from src.visualization.charts import save_all_charts; ...\"")
    print("=" * 80 + "\n")


# ----------------------------------------------------------------------
# Точка входа (демонстрация)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    show_context_info()

    # Демонстрация графиков, если есть данные
    skill_weights = load_skill_weights()
    if not skill_weights:
        print("\n❌ skill_weights.json не найден. Невозможно построить графики.")
        sys.exit(1)

    # Строим распределение весов
    fig1 = plot_weight_distribution(skill_weights)
    plt.show()

    # Пытаемся собрать данные по профилям из готовых отчётов
    results_for_charts = {}
    for profile_name in ["base", "dc", "top_dc"]:
        report_path = config.DATA_DIR / "result" / profile_name / f"comparison_report_{profile_name}.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            results_for_charts[profile_name] = {
                'coverage_percent': report.get('coverage', {}).get('raw', 0),
                'readiness_score': report.get('readiness_score', 0),
                'covered_skills': report.get('student', {}).get('skills', [])
            }

    if results_for_charts:
        fig2 = plot_coverage_comparison(results_for_charts)
        plt.show()

    # ML-рекомендации для base
    recs = load_ml_recommendations("base")
    if recs:
        fig3 = plot_ml_importance("base", top_n=10)
        plt.show()
    else:
        print("\nℹ️  ML-рекомендации для 'base' не найдены.")
        print("   Обучите модель командой:")
        print("   python -m src.predictors.ltr_recommendation_engine --load-raw --train")

    # Предложение запустить ноутбуки
    run_nb = input("\nЗапустить ноутбуки анализа? (y/n): ").strip().lower()
    if run_nb == 'y':
        run_notebook("01_hh_analysis.ipynb")
        run_notebook("02_competency_matching.ipynb")