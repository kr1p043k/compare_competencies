import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Добавляем корень проекта в sys.path (2 уровня вверх)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import config

# ----------------------------------------------------------------------
# Настройка стилей
# ----------------------------------------------------------------------
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Базовые функции визуализации
# ----------------------------------------------------------------------

def plot_coverage_comparison(results: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
    """
    Строит столбчатую диаграмму сравнения покрытия для нескольких учеников.
    :param results: словарь {student_name: report_dict} с полями 'coverage_percent' и 'weighted_coverage_percent'
    """
    df = pd.DataFrame([
        {
            'Ученик': name,
            'Покрытие (доля)': rep['coverage_percent'],
            'Взвешенное покрытие': rep['weighted_coverage_percent']
        }
        for name, rep in results.items()
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(x='Ученик', y=['Покрытие (доля)', 'Взвешенное покрытие'], kind='bar', ax=ax)
    ax.set_title('Сравнение покрытия рыночных навыков')
    ax.set_ylabel('Процент')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_top_deficits(deficits: List[Dict], student_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Строит горизонтальную столбчатую диаграмму топ-дефицитов.
    :param deficits: список словарей с ключами 'skill' и 'frequency'
    """
    if not deficits:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Нет дефицитов высокого спроса", ha='center', va='center')
        ax.set_title(f"{student_name}")
        return fig

    skills = [d['skill'] for d in deficits[:10]]
    freqs = [d['frequency'] for d in deficits[:10]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(skills, freqs, color='salmon')
    ax.set_title(f"Топ-10 дефицитов высокого спроса — {student_name}")
    ax.set_xlabel("Частота в вакансиях")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_radar_chart(student_skills: List[str], market_top: List[str], student_name: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Лепестковая диаграмма: сравнение профиля ученика с топ-навыками рынка.
    """
    # Создаём массив: 1, если навык есть у студента, иначе 0
    data = [1 if skill in student_skills else 0 for skill in market_top]
    angles = np.linspace(0, 2 * np.pi, len(market_top), endpoint=False).tolist()
    # Замыкаем круг
    data += data[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, data, 'o-', linewidth=2)
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(market_top, size=10)
    ax.set_title(f"Профиль ученика {student_name} vs топ-навыки рынка")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_skill_trends(history_df: pd.DataFrame, skill: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Строит график изменения частоты навыка во времени.
    """
    data = history_df[history_df['skill'] == skill].sort_values('date')
    if data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Нет данных по навыку '{skill}'", ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['date'], data['frequency'], marker='o', linestyle='-')
    ax.set_title(f"Тренд востребованности: {skill}")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Частота в вакансиях")
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_heatmap(skill_matrix: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Тепловая карта соответствия учеников навыкам.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(skill_matrix, annot=True, cmap='Blues', cbar_kws={'label': 'Наличие'}, ax=ax)
    ax.set_title("Матрица соответствия учеников навыкам")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def save_all_charts(results: Dict[str, Any], output_dir: Path):
    """Сохраняет основные графики для каждого ученика и сводные."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем skill_weights для получения топ-навыков рынка
    skill_weights_path = config.DATA_PROCESSED_DIR / "skill_weights.json"
    top_market_skills = []
    if skill_weights_path.exists():
        try:
            with open(skill_weights_path, 'r', encoding='utf-8') as f:
                skill_weights = json.load(f)

            # Стоп-слова, которые не должны попадать в радиарную диаграмму
            stop_skills = {
                "язык", "английский", "английский язык", "язык frontend",
                "frontend английский", "frontend английский язык"
            }

            def filter_top_skills(weights, top_n=20):
                # Получаем топ-N навыков по весу, исключая стоп-слова
                sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                filtered_items = [(skill, w) for skill, w in sorted_items if skill not in stop_skills]
                top_items = filtered_items[:top_n]
                skills = [skill for skill, _ in top_items]

                # Удаляем навыки, которые являются подстрокой другого навыка (оставляем более длинные)
                result = []
                for skill in skills:
                    if not any(skill != other and skill in other for other in skills):
                        result.append(skill)

                # Если после фильтрации осталось слишком мало, возвращаем первые top_n без подстрок
                if len(result) < top_n // 2:
                    result = [skill for skill, _ in filtered_items[:top_n]]
                return result

            top_market_skills = filter_top_skills(skill_weights, top_n=20)
            logger.info(f"Загружены топ-{len(top_market_skills)} навыков рынка для радиарной диаграммы: {top_market_skills}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить skill_weights.json: {e}")
    else:
        logger.warning("skill_weights.json не найден, радиарные диаграммы не будут построены")

    # Общий график покрытия
    plot_coverage_comparison(results, save_path=output_dir / "coverage_comparison.png")

    for name, rep in results.items():
        student_dir = output_dir / name
        student_dir.mkdir(parents=True, exist_ok=True)

        # Дефициты
        if 'high_demand_gaps' in rep:
            plot_top_deficits(rep['high_demand_gaps'], name,
                              save_path=student_dir / f"top_deficits_{name}.png")

        # Радар (если есть covered_skills и топ-рыночные навыки)
        if 'covered_skills' in rep and top_market_skills:
            plot_radar_chart(rep['covered_skills'], top_market_skills, name,
                             save_path=student_dir / f"radar_{name}.png")

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
    import json

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
    logging.basicConfig(level=logging.INFO)

    # Показываем контекстную информацию
    show_context_info()

    # Запустить ноутбуки (опционально)
    run_analysis = input("Запустить анализ через ноутбуки? (y/n): ").strip().lower()
    if run_analysis == 'y':
        # Запускаем 01_hh_analysis.ipynb (анализ рынка)
        run_notebook("01_hh_analysis.ipynb")
        # Запускаем 02_competency_matching.ipynb (сравнение профилей)
        run_notebook("02_competency_matching.ipynb")