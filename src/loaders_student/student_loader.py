import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import shutil
import traceback

import pandas as pd
import structlog

from src import DomainError, Err, Ok, Result
from src.config import DATA_RAW_DIR, LAST_UPLOADED_DIR, PROFILES_DISCIPLINES, STUDENTS_DIR
from src.models.student import StudentProfile

logger = structlog.get_logger(__name__)


class StudentLoader:
    """Загрузчик данных учеников из JSON-файлов."""

    def __init__(self, students_dir: Path = STUDENTS_DIR):
        self.students_dir = students_dir

    def load_student(self, profile_name: str) -> Result[StudentProfile, DomainError]:
        """Загрузить JSON профиля: компетенции + уровни навыков + целевой уровень."""
        """Загружает данные ученика по имени профиля (base, dc, top_dc)."""
        file_path = self.students_dir / f"{profile_name}_competency.json"
        if not file_path.exists():
            return Err(DomainError(message=f"Файл студента не найден: {file_path}"))

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return Err(DomainError(message=f"Ошибка чтения файла студента: {e}"))

        skills = data.get("навыки", [])
        # C2 fix: fallback to other keys for backward compat
        if not skills:
            skills = data.get("компетенции", [])
        if not skills:
            for _k, _v in data.items():
                if isinstance(_v, list) and _v and isinstance(_v[0], str):
                    skills = _v
                    break
        # C2+C3 fix: load per-skill levels, correct target_level per profile
        skill_levels = data.get("skill_levels", {})
        _level_map = {"base": "junior", "dc": "middle", "top_dc": "senior"}
        return Ok(StudentProfile(
            profile_name=profile_name,
            competencies=skills,
            skills=skills,
            skill_levels=skill_levels,
            target_level=_level_map.get(profile_name, "middle"),
        ))

    def load_all_students(self) -> list[StudentProfile]:
        """Загружает все три профиля."""
        students = []
        for profile in ["base", "dc", "top_dc"]:
            match self.load_student(profile):
                case Ok(student):
                    students.append(student)
                case Err(e):
                    logger.warning("student_load_skipped", profile=profile, error=str(e))
        return students


# ====================== Создание профилей из CSV ======================
def generate_profiles_from_csv(
    csv_path: Path = DATA_RAW_DIR / "competency_matrix.csv", output_dir: Path = STUDENTS_DIR, save_copy: bool = True
) -> Result[dict[str, list[str]], DomainError]:
    """Сгенерировать JSON профилей (+skill_levels) из CSV-матрицы."""
    logger.info("csv_processing_started", path=str(csv_path))

    if not csv_path.exists():
        return Err(DomainError(message=f"CSV файл не найден: {csv_path}"))

    try:
        try:
            df = pd.read_csv(csv_path, header=None, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, header=None, encoding="cp1251")
        logger.debug("csv_loaded", shape=df.shape)
    except Exception as e:
        logger.exception("csv_read_error")
        return Err(DomainError(message=f"Ошибка чтения CSV: {e}"))

    try:
        indicators_raw = df.iloc[1, 1:].tolist()
        indicator_mapping = {}
        col_idx = 2
        for raw in indicators_raw:
            if pd.isna(raw):
                continue
            code = str(raw).split(" ", 1)[0]
            indicator_mapping[col_idx] = code
            col_idx += 1
        logger.info("indicators_found", count=len(indicator_mapping))

        disciplines_df = df.iloc[2:, :].copy()
        disciplines_df.columns = list(range(disciplines_df.shape[1]))
        disciplines_df[0] = pd.to_numeric(disciplines_df[0], errors="coerce").fillna(0).astype(int)

        profiles_skills = {profile: set() for profile in PROFILES_DISCIPLINES}
        # C2 fix: track per-skill mastery level (Б/П/Э/X or checkmark variants)
        profiles_levels: dict[str, dict[str, str]] = {profile: {} for profile in PROFILES_DISCIPLINES}

        for _, row in disciplines_df.iterrows():
            discipline_id = int(row[0])
            if discipline_id == 0:
                continue

            for profile_name, discipline_ids in PROFILES_DISCIPLINES.items():
                if discipline_id not in discipline_ids:
                    continue

                for col_idx, indicator_code in indicator_mapping.items():
                    val = row.iloc[col_idx]
                    if pd.notna(val):
                        _v = str(val).strip()
                        if _v in ("\u2713", "\u0445", "\u0425", "X", "\u0411", "\u041f", "\u042d", "B", "P", "E"):
                            profiles_skills[profile_name].add(indicator_code)
                            # C2 fix: preserve mastery level (normalize Cyrillic to Latin)
                            _lvl = {"\u0411": "B", "\u041f": "P", "\u042d": "E", "\u0445": "X", "\u0425": "X"}.get(_v, _v)
                            profiles_levels[profile_name][indicator_code] = _lvl

        result = {}
        for profile_name, skills in profiles_skills.items():
            sorted_skills = sorted(skills)
            result[profile_name] = sorted_skills
            logger.info("profile_skills_extracted", profile=profile_name, skills_count=len(sorted_skills))

            json_path = output_dir / f"{profile_name}_competency.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"навыки": sorted_skills, "skill_levels": profiles_levels[profile_name]}, f, ensure_ascii=False, indent=2)

        if save_copy:
            LAST_UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
            last_csv_path = LAST_UPLOADED_DIR / "competency_matrix.csv"
            shutil.copy2(csv_path, last_csv_path)

        logger.info("csv_processing_completed")
        return Ok(result)
    except Exception as e:
        logger.exception("csv_processing_error")
        return Err(DomainError(message=f"Ошибка обработки CSV: {e}"))


if __name__ == "__main__":
    print("Запуск генерации профилей из CSV...")
    match generate_profiles_from_csv():
        case Ok(profiles):
            print("Готово! Созданы файлы:")
            for name, skills in profiles.items():
                print(f"  {name}: {len(skills)} навыков")
            print(f"JSON-файлы сохранены в: {STUDENTS_DIR}")
            print(f"Копия CSV сохранена в: {LAST_UPLOADED_DIR / 'competency_matrix.csv'}")
        case Err(e):
            print(f"Ошибка: {e.message}")
            traceback.print_exc()
