import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import shutil
import traceback
from typing import List, Optional, Dict

import pandas as pd

from src.models.student import StudentProfile   # ← ИСПРАВЛЕНО: импортируем StudentProfile
from src.utils import get_logger
from src.config import STUDENTS_DIR, LAST_UPLOADED_DIR, PROFILES_DISCIPLINES, DATA_RAW_DIR

logger = get_logger(__name__)

class StudentLoader:
    """Загрузчик данных учеников из JSON-файлов."""

    def __init__(self, students_dir: Path = STUDENTS_DIR):
        self.students_dir = students_dir

    def load_student(self, profile_name: str) -> Optional[StudentProfile]:
        """Загружает данные ученика по имени профиля (base, dc, top_dc)."""
        file_path = self.students_dir / f"{profile_name}_competency.json"
        if not file_path.exists():
            logger.error(f"Файл {file_path} не найден")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        skills = data.get("навыки", [])
        # Создаём объект StudentProfile с правильными полями
        return StudentProfile(
            student_id=profile_name,
            name=profile_name,
            competencies=skills,          # ← поле competencies, а не skills
            target_role="Data Scientist"
        )

    def load_all_students(self) -> List[StudentProfile]:
        students = []
        for profile in ["base", "dc", "top_dc"]:
            student = self.load_student(profile)
            if student:
                students.append(student)
        return students


def generate_profiles_from_csv(
    csv_path: Path = DATA_RAW_DIR / "competency_matrix.csv",
    output_dir: Path = STUDENTS_DIR,
    save_copy: bool = True
) -> Dict[str, List[str]]:
    logger.info(f"Начало обработки CSV-файла: {csv_path}")

    if not csv_path.exists():
        logger.error(f"Файл {csv_path} не найден.")
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    # Чтение CSV
    try:
        try:
            df = pd.read_csv(csv_path, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, header=None, encoding='cp1251')
        logger.debug(f"CSV загружен, форма: {df.shape}")
    except Exception as e:
        logger.exception("Ошибка при чтении CSV")
        raise

    # Индикаторы компетенций (вторая строка)
    indicators_raw = df.iloc[1, 1:].tolist()
    indicator_codes = []
    for raw in indicators_raw:
        if pd.isna(raw):
            continue
        code = str(raw).split(' ', 1)[0]
        indicator_codes.append(code)
    logger.info(f"Найдено индикаторов компетенций: {len(indicator_codes)}")

    # Данные дисциплин (строки с 3-й)
    disciplines_df = df.iloc[2:, :].copy()
    disciplines_df.columns = ['№', 'Дисциплина'] + indicator_codes
    disciplines_df['№'] = pd.to_numeric(disciplines_df['№'], errors='coerce').fillna(0).astype(int)

    # Сбор навыков по профилям
    profiles_skills = {}
    for profile_name, discipline_ids in PROFILES_DISCIPLINES.items():
        logger.debug(f"Обработка профиля {profile_name}, дисциплины: {discipline_ids}")
        profile_df = disciplines_df[disciplines_df['№'].isin(discipline_ids)]

        if len(profile_df) == 0:
            logger.warning(f"Для профиля {profile_name} не найдено ни одной дисциплины.")
            profiles_skills[profile_name] = []
            continue

        skills = set()
        for _, row in profile_df.iterrows():
            for indicator in indicator_codes:
                val = row.get(indicator)
                # Защита от случая, когда val оказывается Series (ошибка неоднозначности)
                if isinstance(val, pd.Series):
                    logger.warning(f"Пропуск индикатора {indicator}: получен Series, ожидался скаляр")
                    continue
                if pd.notna(val) and str(val).strip() in ('Б', 'П'):
                    skills.add(indicator)

        profile_skills = sorted(skills)
        profiles_skills[profile_name] = profile_skills
        logger.info(f"Профиль {profile_name}: получено {len(profile_skills)} навыков")

        # Сохраняем JSON
        json_path = output_dir / f"{profile_name}_competency.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"навыки": profile_skills}, f, ensure_ascii=False, indent=2)
        logger.debug(f"Сохранён JSON: {json_path}")

    # Копия CSV
    if save_copy:
        LAST_UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
        last_csv_path = LAST_UPLOADED_DIR / "competency_matrix.csv"
        shutil.copy2(csv_path, last_csv_path)
        logger.info(f"Сохранена копия загруженного CSV: {last_csv_path}")

    logger.info("Обработка CSV завершена успешно")
    return profiles_skills


if __name__ == "__main__":
    print("Запуск генерации профилей из CSV...")
    try:
        profiles = generate_profiles_from_csv()
        print("Готово! Созданы файлы:")
        for name, skills in profiles.items():
            print(f"  {name}: {len(skills)} навыков")
        print(f"JSON-файлы сохранены в: {STUDENTS_DIR}")
        print(f"Копия CSV сохранена в: {LAST_UPLOADED_DIR / 'competency_matrix.csv'}")
    except Exception as e:
        print(f"Ошибка: {e}")
        traceback.print_exc()