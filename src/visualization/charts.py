# src/visualization/charts.py (обновлённая версия)
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

logger = logging.getLogger(__name__)


def load_skill_weights() -> Dict[str, float]:
    """Загружает skill_weights из data/processed/skill_weights.json."""
    path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_hybrid_weights() -> Dict[str, float]:
    """Загружает гибридные веса (если есть)."""
    path = config.DATA_PROCESSED_DIR / "hybrid_weights.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_ml_recommendations(profile_name: str) -> List[Tuple[str, float, str]]:
    """Загружает ML-рекомендации для профиля."""
    rec_file = config.DATA_DIR / "result" / profile_name / f"ml_recommendations_{profile_name}.json"
    if rec_file.exists():
        with open(rec_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [(item['skill'], item['score'], item['explanation']) for item in data.get('recommendations', [])]
    return []


def plot_coverage_comparison(results: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    """Сравнение покрытия (coverage) для нескольких профилей."""
    data = []
    for name, rep in results.items():
        row = {'Профиль': name}
        if 'coverage_percent' in rep:
            row['Покрытие (%)'] = rep['coverage_percent']
        elif 'coverage_details' in rep:
            det = rep['coverage_details']
            row['Покрытие (%)'] = (det['covered_weight'] / det['total_weight'] * 100) if det['total_weight'] > 0 else 0
        if 'readiness_score' in rep:
            row['Готовность (%)'] = rep['readiness_score']
        data.append(row)

    if not data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет данных", ha='center', va='center')
        return fig

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x='Профиль', kind='bar', ax=ax)
    ax.set_title('Сравнение профилей: покрытие и готовность')
    ax.set_ylabel('Проценты')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ml_importance(profile_name: str, top_n: int = 10, save_path: Optional[Path] = None) -> plt.Figure:
    """Визуализирует топ-N ML-рекомендаций для профиля."""
    recs = load_ml_recommendations(profile_name)
    if not recs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Нет ML-рекомендаций для '{profile_name}'", ha='center', va='center')
        return fig

    skills = [r[0] for r in recs[:top_n]]
    scores = [r[1] for r in recs[:top_n]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(skills, scores, color='skyblue')
    ax.set_title(f"ML-рекомендации для '{profile_name}': важность навыков")
    ax.set_xlabel('Важность (%)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Добавим подписи с процентом
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{score:.1f}%', va='center')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_skill_comparison_radar(
    student_skills: List[str],
    market_top_skills: List[str],
    student_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Радарная диаграмма: профиль студента против топ-навыков рынка."""
    data = [1 if skill in student_skills else 0 for skill in market_top_skills]
    angles = np.linspace(0, 2 * np.pi, len(market_top_skills), endpoint=False).tolist()
    data += data[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, data, 'o-', linewidth=2)
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(market_top_skills, size=9)
    ax.set_title(f"Профиль '{student_name}' vs Топ-{len(market_top_skills)} рынка", pad=20)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_weight_distribution(weights: Dict[str, float], title: str = "Распределение весов навыков", save_path: Optional[Path] = None) -> plt.Figure:
    """Гистограмма распределения весов (skill_weights или hybrid_weights)."""
    if not weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Нет данных о весах", ha='center', va='center')
        return fig

    values = list(weights.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Вес')
    ax.set_ylabel('Количество навыков')
    ax.grid(axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def save_all_charts(results: Dict[str, Any], output_dir: Path, use_ml: bool = True):
    """Сохраняет основные графики в output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сводное сравнение покрытия
    plot_coverage_comparison(results, save_path=output_dir / "coverage_comparison.png")

    # Загружаем рыночные топ-навыки для радаров
    skill_weights = load_skill_weights()
    stop_skills = {"язык", "английский", "английский язык", "frontend", "backend"}
    top_skills = [s for s, _ in sorted(skill_weights.items(), key=lambda x: x[1], reverse=True)[:40]
                  if s not in stop_skills][:20]

    for name, rep in results.items():
        profile_dir = output_dir / name
        profile_dir.mkdir(exist_ok=True)

        # Радарная диаграмма
        if 'covered_skills' in rep and top_skills:
            plot_skill_comparison_radar(rep['covered_skills'], top_skills, name,
                                        save_path=profile_dir / f"radar_{name}.png")

        # ML-рекомендации
        if use_ml:
            plot_ml_importance(name, save_path=profile_dir / f"ml_importance_{name}.png")

        # Дефициты (если есть в отчёте)
        if 'high_demand_gaps' in rep:
            deficits = rep['high_demand_gaps'][:10]
            if deficits:
                skills = [d['skill'] for d in deficits]
                freqs = [d['frequency'] for d in deficits]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(skills, freqs, color='salmon')
                ax.set_title(f"Топ-10 дефицитов высокого спроса — {name}")
                ax.set_xlabel("Частота")
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                fig.savefig(profile_dir / f"deficits_{name}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

# ----------------------------------------------------------------------
# Утилиты для запуска ноутбуков и вывода контекстной информации
# ----------------------------------------------------------------------

def run_notebook(notebook_name: str, output_dir: Optional[Path] = None) -> bool:
    """
    Выполняет Jupyter ноутбук с помощью nbconvert и сохраняет результат.
    :param notebook_name: имя файла ноутбука (например, "01_hh_analysis.ipynb")
    :param output_dir: папка для сохранения выполненного ноутбука (если не указана, сохраняется в ту же папку)
    :return: True, если выполнение успешно, иначе False
    """
    # Корень проекта: src/visualization/charts.py -> src/ -> корень
    base_dir = Path(__file__).parent.parent.parent
    notebook_path = base_dir / "notebook_jypiter" / notebook_name
    if not notebook_path.exists():
        logger.error(f"Ноутбук не найден: {notebook_path}")
        return False

    # Определяем выходной файл
    if output_dir is None:
        output_path = notebook_path
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / notebook_name

    try:
        # Выполняем ноутбук с помощью jupyter nbconvert --execute
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
    """
    Выводит в консоль сводную информацию о текущих данных проекта.
    Использует пути из config.py.
    """

    print("\n" + "=" * 80)
    print("КОНТЕКСТНАЯ ИНФОРМАЦИЯ О ПРОЕКТЕ")
    print("=" * 80)

    # Рыночные навыки (файл competency_frequency.json)
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
        print("Файл с рыночными навыками не найден. Сначала соберите данные: python main.py --it-sector --excel")

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
        print(f"Файл маппинга не найден: {mapping_file}")

    # Профили учеников
    students_dir = config.STUDENTS_DIR
    students = list(students_dir.glob("*_competency.json"))
    print(f"Профили учеников (JSON): {len(students)}")
    for student_file in students:
        try:
            with open(student_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                skills = data.get("навыки", [])
                print(f"  - {student_file.stem.replace('_competency', '')}: {len(skills)} компетенций")
        except Exception as e:
            print(f"  - Ошибка чтения {student_file.name}: {e}")

    # Результаты анализа (отчёты в data/result/)
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
    print("\nРЕКОМЕНДАЦИИ:")
    if not market_file.exists():
        print("  - Сначала соберите рыночные данные: python main.py --it-sector --excel")
    elif not mapping_file.exists():
        print("  - Создайте файл competency_mapping.json в data/processed/")
    elif not students:
        print("  - Загрузите профили студентов через student_loader.py или вручную в data/students/")
    elif reports_found == 0:
        print("  - Запустите gap-анализ: python main.py (или выполните 02_competency_matching.ipynb)")
    else:
        print("  - Все данные в порядке. Вы можете запускать визуализацию или веб-интерфейс.")
    print("=" * 80 + "\n")


# ----------------------------------------------------------------------
# Точка входа для демонстрации (если запустить файл напрямую)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    logging.basicConfig(level=logging.INFO)
    show_context_info()

    # --------------------------------------------------------------
    # ДЕМОНСТРАЦИЯ ГРАФИКОВ (автоматически при запуске)
    # --------------------------------------------------------------
    skill_weights = load_skill_weights()
    if not skill_weights:
        print("\n" + "=" * 60)
        print("❌ skill_weights.json не найден.")
        print("   Соберите данные командой: python main.py --it-sector")
        print("=" * 60)
        sys.exit(1)

    # 1. Распределение весов навыков
    fig1 = plot_weight_distribution(skill_weights, "Распределение очищенных весов навыков")
    plt.show()

    # 2. Попробуем построить сравнение покрытия по профилям
    #    Загрузим данные из готовых отчётов или создадим минимальный словарь
    results_for_charts = {}
    profiles_found = False
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
            profiles_found = True

    if profiles_found:
        fig2 = plot_coverage_comparison(results_for_charts)
        plt.show()

    # 3. ML-рекомендации для профиля 'base' (если есть)
    recs = load_ml_recommendations("base")
    if recs:
        fig3 = plot_ml_importance("base", top_n=10)
        plt.show()
    else:
        print("\n" + "-" * 60)
        print("ℹ️  ML-рекомендации для 'base' не найдены.")
        print("   Обучите модель командой:")
        print("   python -m src.predictors.ml_recommendation_engine --load-raw --train")
        print("-" * 60)

    # 4. Опциональный запуск ноутбуков
    run_analysis = input("\nЗапустить ноутбуки анализа? (y/n): ").strip().lower()
    if run_analysis == 'y':
        run_notebook("01_hh_analysis.ipynb")
        run_notebook("02_competency_matching.ipynb")